"""Generation orchestration for Qwen AdaLoop readouts.

The callbacks in this module isolate cache mechanics from rollout semantics:
the Qwen wrapper supplies M-RoPE-aware prefill/append operations.

Decode policy
-------------
Every released decode path is greedy and pinned in code.  Nothing is read from
a checkpoint's ``generation_config.json`` (Qwen2.5-VL-3B-Instruct ships
``do_sample=true``, ``temperature=1e-6``, ``repetition_penalty=1.05`` and
``eos_token_id=[151645, 151643]``).  The policy reproduces the evaluation
that produced the paper's numbers:

================================  =======  =============  ==============  ======
path                              penalty  stop token     suppressed IDs  budget
================================  =======  =============  ==============  ======
SFT, AdaLN, Variant A, Variant B  1.05     tokenizer EOS  none            64
(fixed depth and every p=1 depth)
Variant C drafts (rounds < t)     1.0      tokenizer EOS  other specials  24
Variant C terminal answer         1.0      tokenizer EOS  other specials  64
================================  =======  =============  ==============  ======

* The paper decoded SFT, AdaLN, A and B with Hugging Face ``generate``
  (``do_sample=False``, ``num_beams=1``, ``eos_token_id=tokenizer.eos_token_id``)
  under the 1.05 penalty inherited from the base model.  The release passes
  that policy explicitly to ``generate`` (SFT, AdaLN, fixed-depth A), and
  :func:`greedy_decode` reproduces ``RepetitionPenaltyLogitsProcessor``
  exactly for the cached readouts (B, every p=1 depth): once per token ID
  present anywhere in the context (prompt text, chat-template and image
  placeholder tokens, and the tokens generated so far), a positive float32
  logit is divided by the penalty and a negative one is multiplied by it.
  ``<|endoftext|>`` (151643) is an ordinary token on these paths; it is emitted
  and later removed by ``skip_special_tokens``.
* Variant C used a dedicated greedy loop with no penalty that masks every
  special token except EOS, including ``<|endoftext|>``, exactly as during
  training-time draft rollout.
* p=1 selects among the same per-depth decodes as fixed-depth generation; the
  paper applies the stopping rule to separate fixed-depth evaluations.
* EOS is ``tokenizer.eos_token_id`` (``<|im_end|>``, 151645) alone and is
  excluded from returned token IDs.  Budgets are the recipe's
  ``generation.max_new_tokens`` and ``readout.draft_max_tokens``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar

import torch
from adaloop_common.halting import normalize_for_stability

from .readouts import (
    InterleavedDraftReadout,
    PreparedVariantC,
    build_interleaved_draft_readout,
)


# Repetition penalty of the SFT/AdaLN/A/B decode paths (see module docstring).
REPETITION_PENALTY = 1.05


@dataclass
class DecodeState:
    """Opaque runtime cache plus logits for the next autoregressive token."""

    logits: torch.Tensor
    context_length: int
    cache: Any = None


class DecodeRuntime(Protocol):
    """Model-specific operations required by the framework-free greedy loop."""

    def prefill(
        self,
        readout: InterleavedDraftReadout,
        step_features: Sequence[torch.Tensor],
    ) -> DecodeState: ...

    def append(
        self,
        state: DecodeState,
        token_ids: torch.LongTensor,
        token_mask: torch.Tensor,
    ) -> DecodeState: ...


@dataclass(frozen=True)
class GreedyDecodeOutput:
    """Right-padded generated IDs; EOS is represented only as a stop reason."""

    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    stop_reasons: tuple[str, ...]


@dataclass(frozen=True)
class VariantCGenerationOutput:
    sequences: torch.LongTensor
    sequence_attention_mask: torch.Tensor
    drafts: tuple[torch.LongTensor, ...]
    draft_attention_masks: tuple[torch.Tensor, ...]
    stop_reasons: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class VariantCDraftRollout:
    """Detached nonterminal drafts generated for a Variant-C prefix."""

    drafts: tuple[torch.LongTensor, ...]
    draft_attention_masks: tuple[torch.Tensor, ...]
    stop_reasons: tuple[tuple[str, ...], ...]


def _token_ids(value: int | Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().flatten().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        values = (value,)
    result: list[int] = []
    for token_id in values:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token IDs must be integers")
        if token_id not in result:
            result.append(int(token_id))
    return tuple(result)


def suppressed_special_token_ids(
    special_token_ids: Sequence[int] | torch.Tensor,
    *,
    eos_token_ids: int | Sequence[int] | torch.Tensor,
    vocab_size: int | None = None,
) -> tuple[int, ...]:
    """Return sorted non-EOS special/control IDs suitable for logit masking."""

    eos = set(_token_ids(eos_token_ids))
    special = set(_token_ids(special_token_ids))
    special.difference_update(eos)
    if vocab_size is not None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        special = {token_id for token_id in special if 0 <= token_id < vocab_size}
    elif any(token_id < 0 for token_id in special):
        raise ValueError("special token IDs must be non-negative")
    return tuple(sorted(special))


def _validate_state(state: DecodeState, *, previous_length: int | None = None) -> None:
    if not isinstance(state, DecodeState):
        raise TypeError("DecodeRuntime methods must return DecodeState")
    if not isinstance(state.logits, torch.Tensor) or state.logits.ndim != 2:
        shape = None if not isinstance(state.logits, torch.Tensor) else tuple(state.logits.shape)
        raise ValueError(f"DecodeState.logits must have shape [B, V], got {shape}")
    if not state.logits.is_floating_point():
        raise TypeError("DecodeState.logits must use a floating dtype")
    if (
        isinstance(state.context_length, bool)
        or not isinstance(state.context_length, int)
        or state.context_length < 1
    ):
        raise ValueError("DecodeState.context_length must be a positive integer")
    if previous_length is not None and state.context_length != previous_length + 1:
        raise RuntimeError(
            "DecodeRuntime.append must advance the cache by one position; got "
            f"{previous_length} -> {state.context_length}"
        )


def greedy_decode(
    runtime: DecodeRuntime,
    state: DecodeState,
    *,
    max_new_tokens: int,
    eos_token_ids: int | Sequence[int] | torch.Tensor,
    pad_token_id: int,
    suppressed_token_ids: Sequence[int] | torch.Tensor = (),
    max_context_length: int | None = None,
    repetition_penalty: float = 1.0,
    context_token_ids: torch.Tensor | None = None,
) -> GreedyDecodeOutput:
    """Greedily decode while excluding EOS and suppressed specials from output.

    A ``repetition_penalty`` other than 1.0 follows Hugging Face
    ``RepetitionPenaltyLogitsProcessor`` on float32 logits.  It requires
    ``context_token_ids`` ``[B, T]``, the prompt whose IDs count as previous
    tokens together with every token generated so far.
    """

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TypeError("pad_token_id must be an integer")
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or not math.isfinite(repetition_penalty)
        or repetition_penalty <= 0
    ):
        raise ValueError("repetition_penalty must be a positive finite number")
    _validate_state(state)
    if max_context_length is not None:
        if (
            isinstance(max_context_length, bool)
            or not isinstance(max_context_length, int)
            or max_context_length < 1
        ):
            raise ValueError("max_context_length must be a positive integer or None")
        requested = state.context_length + max_new_tokens
        if requested > max_context_length:
            raise ValueError(
                f"Decode budget exceeds max_context_length={max_context_length}: "
                f"{state.context_length} + {max_new_tokens} = {requested}"
            )

    eos_ids = _token_ids(eos_token_ids)
    if not eos_ids:
        raise ValueError("eos_token_ids must contain at least one ID")
    vocab_size = int(state.logits.shape[1])
    if any(token_id < 0 or token_id >= vocab_size for token_id in eos_ids):
        raise ValueError("EOS token IDs must lie inside the logits vocabulary")
    suppressed = suppressed_special_token_ids(
        suppressed_token_ids,
        eos_token_ids=eos_ids,
        vocab_size=vocab_size,
    )
    suppression_index = (
        torch.tensor(suppressed, dtype=torch.long, device=state.logits.device)
        if suppressed
        else None
    )
    eos_tensor = torch.tensor(eos_ids, dtype=torch.long, device=state.logits.device)
    batch_size = int(state.logits.shape[0])
    penalty = float(repetition_penalty)
    seen: torch.Tensor | None = None
    if penalty != 1.0:
        if not isinstance(context_token_ids, torch.Tensor):
            raise ValueError("repetition_penalty requires context_token_ids")
        if (
            context_token_ids.ndim != 2
            or context_token_ids.shape[0] != batch_size
            or context_token_ids.dtype == torch.bool
            or context_token_ids.is_floating_point()
        ):
            raise ValueError(
                "context_token_ids must be an integer [B, T] tensor matching "
                f"the logits batch {batch_size}; got "
                f"{tuple(context_token_ids.shape)} {context_token_ids.dtype}"
            )
        context = context_token_ids.to(device=state.logits.device, dtype=torch.long)
        if context.numel() and (
            int(context.min()) < 0 or int(context.max()) >= vocab_size
        ):
            raise ValueError("context_token_ids must lie inside the logits vocabulary")
        # Tokens already present in the context, counted once per ID.
        seen = torch.zeros(
            (batch_size, vocab_size), dtype=torch.bool, device=state.logits.device
        )
        seen.scatter_(1, context, True)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=state.logits.device)
    eos_steps = torch.full(
        (batch_size,), -1, dtype=torch.long, device=state.logits.device
    )
    token_columns: list[torch.Tensor] = []
    mask_columns: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        # Do not mutate runtime-owned logits. Some cache adapters retain this
        # tensor for diagnostics or reuse it when every row stops immediately.
        # Like Hugging Face generate, score in at least float32.
        logits = state.logits.to(
            dtype=torch.promote_types(state.logits.dtype, torch.float32), copy=True
        )
        if seen is not None:
            penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
            logits = torch.where(seen, penalized, logits)
        if suppression_index is not None:
            logits.index_fill_(1, suppression_index, torch.finfo(logits.dtype).min)
        selected = logits.argmax(dim=-1)
        is_eos = (selected[:, None] == eos_tensor[None, :]).any(dim=1)
        newly_finished = ~finished & is_eos
        eos_steps = torch.where(
            newly_finished, torch.full_like(eos_steps, step), eos_steps
        )
        emit = ~finished & ~is_eos
        stored = torch.where(
            emit, selected, torch.full_like(selected, int(pad_token_id))
        )
        token_columns.append(stored)
        mask_columns.append(emit)
        finished = finished | is_eos
        if seen is not None:
            seen.scatter_(1, selected[:, None], True)

        if step + 1 >= max_new_tokens or not bool(torch.any(emit)):
            break
        previous_length = state.context_length
        state = runtime.append(state, stored[:, None], emit[:, None])
        _validate_state(state, previous_length=previous_length)
        if state.logits.shape[0] != batch_size:
            raise RuntimeError("DecodeRuntime.append changed the batch size")

    generated = torch.stack(token_columns, dim=1)
    generated_mask = torch.stack(mask_columns, dim=1)
    reasons = tuple(
        "eos" if step >= 0 else "max_tokens"
        for step in eos_steps.detach().cpu().tolist()
    )
    return GreedyDecodeOutput(
        input_ids=generated,
        attention_mask=generated_mask,
        stop_reasons=reasons,
    )


def _draft_limit(draft_max_tokens: int | None) -> int:
    value = draft_max_tokens
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("draft_max_tokens must be a positive integer")
    return value


@torch.no_grad()
def rollout_variant_c_drafts_reprefill(
    runtime: DecodeRuntime,
    prepared: PreparedVariantC,
    step_features: Sequence[torch.Tensor],
    *,
    target_depth: int | None = None,
    draft_max_tokens: int | None = None,
    eos_token_ids: int | Sequence[int] | torch.Tensor,
    pad_token_id: int,
    suppressed_token_ids: Sequence[int] | torch.Tensor,
    max_context_length: int | None = None,
) -> VariantCDraftRollout:
    """Generate all nonterminal C drafts using detached fresh reprefills.

    This is also the training-time rollout primitive: callers may subsequently
    assemble supervised trajectories with the returned exact token IDs. No KV
    cache or autograd edge crosses a recurrent round boundary.
    """

    if isinstance(step_features, torch.Tensor) or not isinstance(step_features, Sequence):
        raise TypeError("step_features must be a sequence of tensors")
    depth = len(step_features) if target_depth is None else target_depth
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("target_depth must be a positive integer")
    if len(step_features) < depth:
        raise ValueError(
            f"target_depth={depth} requires {depth} step features, got "
            f"{len(step_features)}"
        )
    draft_limit = _draft_limit(draft_max_tokens)
    detached_features = tuple(feature.detach() for feature in step_features[:depth])
    drafts: list[torch.LongTensor] = []
    draft_masks: list[torch.Tensor] = []
    reasons: list[tuple[str, ...]] = []

    for round_depth in range(1, depth):
        trajectory = build_interleaved_draft_readout(
            prepared,
            drafts,
            draft_masks,
            target_depth=round_depth,
            pad_token_id=pad_token_id,
            draft_padding_side="right",
            padding_side="left",
            max_length=max_context_length,
            include_answer=False,
        )
        state = runtime.prefill(trajectory, detached_features[:round_depth])
        _validate_state(state)
        decoded = greedy_decode(
            runtime,
            state,
            max_new_tokens=draft_limit,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
            suppressed_token_ids=suppressed_token_ids,
            max_context_length=max_context_length,
        )
        drafts.append(decoded.input_ids.detach())
        draft_masks.append(decoded.attention_mask.detach())
        reasons.append(decoded.stop_reasons)

    return VariantCDraftRollout(
        drafts=tuple(drafts),
        draft_attention_masks=tuple(draft_masks),
        stop_reasons=tuple(reasons),
    )


@torch.no_grad()
def generate_variant_c_reprefill(
    runtime: DecodeRuntime,
    prepared: PreparedVariantC,
    step_features: Sequence[torch.Tensor],
    *,
    target_depth: int | None = None,
    draft_max_tokens: int | None = None,
    max_new_tokens: int,
    eos_token_ids: int | Sequence[int] | torch.Tensor,
    pad_token_id: int,
    suppressed_token_ids: Sequence[int] | torch.Tensor,
    max_context_length: int | None = None,
) -> VariantCGenerationOutput:
    """Run released Variant C, including a fresh terminal full reprefill."""

    if isinstance(step_features, torch.Tensor) or not isinstance(step_features, Sequence):
        raise TypeError("step_features must be a sequence of tensors")
    depth = len(step_features) if target_depth is None else target_depth
    rollout = rollout_variant_c_drafts_reprefill(
        runtime,
        prepared,
        step_features,
        target_depth=depth,
        draft_max_tokens=draft_max_tokens,
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
        suppressed_token_ids=suppressed_token_ids,
        max_context_length=max_context_length,
    )
    detached_features = tuple(feature.detach() for feature in step_features[:depth])
    drafts = list(rollout.drafts)
    draft_masks = list(rollout.draft_attention_masks)

    final_trajectory = build_interleaved_draft_readout(
        prepared,
        drafts,
        draft_masks,
        target_depth=depth,
        pad_token_id=pad_token_id,
        draft_padding_side="right",
        padding_side="left",
        max_length=max_context_length,
        include_answer=False,
    )
    final_state = runtime.prefill(final_trajectory, detached_features)
    _validate_state(final_state)
    final = greedy_decode(
        runtime,
        final_state,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
        suppressed_token_ids=suppressed_token_ids,
        max_context_length=max_context_length,
    )
    return VariantCGenerationOutput(
        sequences=final.input_ids,
        sequence_attention_mask=final.attention_mask,
        drafts=rollout.drafts,
        draft_attention_masks=rollout.draft_attention_masks,
        stop_reasons=(*rollout.stop_reasons, final.stop_reasons),
    )


OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class FixedDepthOutput(Generic[OutputT]):
    depth: int
    output: OutputT


@dataclass(frozen=True)
class P1Output(Generic[OutputT]):
    """Online one-step answer-stability selection and evaluated prefixes."""

    answer: str
    depth: int
    stabilized: bool
    output: OutputT
    evaluated: tuple[FixedDepthOutput[OutputT], ...]


def generate_p1_stable(
    max_depth: int,
    generate_at_depth: Callable[[int], OutputT],
    *,
    answer_from_output: Callable[[OutputT], str] | None = None,
    fallback_depth: int | None = None,
) -> P1Output[OutputT]:
    """Decode successive depths and stop at the first adjacent answer match.

    This applies ``adaloop_common.halting.select_p1_answer`` online: no depth
    is decoded after the rule fires.
    """

    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 2:
        raise ValueError("p=1 stability requires max_depth >= 2")
    if fallback_depth is None:
        fallback_depth = max_depth
    if (
        isinstance(fallback_depth, bool)
        or not isinstance(fallback_depth, int)
        or not 1 <= fallback_depth <= max_depth
    ):
        raise ValueError("fallback_depth must be between 1 and max_depth")
    if not callable(generate_at_depth):
        raise TypeError("generate_at_depth must be callable")

    def answer_of(output: OutputT) -> str:
        if answer_from_output is not None:
            return answer_from_output(output)
        if not isinstance(output, str):
            raise TypeError(
                "Non-string generation outputs require answer_from_output"
            )
        return output

    evaluated: list[FixedDepthOutput[OutputT]] = []
    answers: list[str] = []
    normalized: list[str] = []
    for depth in range(1, max_depth + 1):
        item = FixedDepthOutput(depth=depth, output=generate_at_depth(depth))
        answer = answer_of(item.output)
        evaluated.append(item)
        answers.append(answer)
        normalized.append(normalize_for_stability(answer))
        if depth > 1 and normalized[-1] == normalized[-2]:
            return P1Output(
                answer=answer,
                depth=depth,
                stabilized=True,
                output=item.output,
                evaluated=tuple(evaluated),
            )

    fallback = evaluated[fallback_depth - 1]
    return P1Output(
        answer=answers[fallback_depth - 1],
        depth=fallback_depth,
        stabilized=False,
        output=fallback.output,
        evaluated=tuple(evaluated),
    )


__all__ = [
    "DecodeRuntime",
    "DecodeState",
    "FixedDepthOutput",
    "GreedyDecodeOutput",
    "P1Output",
    "REPETITION_PENALTY",
    "VariantCDraftRollout",
    "VariantCGenerationOutput",
    "generate_p1_stable",
    "generate_variant_c_reprefill",
    "greedy_decode",
    "rollout_variant_c_drafts_reprefill",
    "suppressed_special_token_ids",
]
