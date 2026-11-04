"""Variant-C interleaved draft trajectory for PaliGemma.

The only released layout is::

  [I1 | P] D1 SEP [I2 | P] D2 SEP ... [It | P] A

``P`` is PaliGemma's native language prefix (including its normal BOS/cue),
whereas vision conditioning always uses the separate bare question.  Drafts
are greedy, detached, capped at 24 tokens, stop on EOS without storing EOS,
and suppress PAD/BOS/UNK/control IDs.  Each visual round starts from a fresh
full-trajectory prefill; tokens within that draft then use exact cached KV
extension.  No cross-round KV-append approximation is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from ..constants import PAPER_DRAFT_MAX_TOKENS, PAPER_MAX_SEQUENCE_LENGTH


PAPER_TRAJECTORY_MAX_LENGTH = PAPER_MAX_SEQUENCE_LENGTH


@dataclass(frozen=True)
class TrajectoryBatch:
    embeddings: Any
    input_mask: Any
    mask_ar: Any
    current_text_offset: Any
    current_text_positions: Any
    last_active_index: Any
    depth: int


def _image_mask_ar(batch: int, tokens: int, *, starts_new_block: bool):
    index = jnp.arange(tokens)
    row = (starts_new_block & (index == 0)).astype(jnp.int32)
    return jnp.broadcast_to(row[None], (batch, tokens))


def pack_trajectory(
    image_readouts: Sequence[Any],
    text_embeddings,
    text_input_mask,
    text_mask_ar,
    draft_embeddings,
    draft_mask,
    separator_embeddings,
    *,
    depth: int,
    terminal: str,
    terminal_draft_embeddings=None,
    terminal_draft_mask=None,
    max_length: int = PAPER_MAX_SEQUENCE_LENGTH,
) -> TrajectoryBatch:
    """Pack a fixed-depth Variant-C prompt or supervised answer trajectory.

    Args:
      terminal: ``"answer"`` includes the full final text input, while
        ``"draft"`` includes only its native prefix and an optional partial
        terminal draft used during greedy rollout.
    """

    if terminal not in {"answer", "draft"}:
        raise ValueError("terminal must be 'answer' or 'draft'")
    if depth < 1 or depth > len(image_readouts):
        raise ValueError("depth exceeds available image readouts")
    if text_embeddings.ndim != 3:
        raise ValueError("text_embeddings must be [B,L,D]")
    if text_input_mask.shape != text_mask_ar.shape:
        raise ValueError("text masks must align")
    batch, text_width, embed_width = text_embeddings.shape
    prefix_mask = text_input_mask.astype(bool) & (text_mask_ar == 0)
    if jnp.shape(draft_embeddings)[:2] != jnp.shape(draft_mask)[:2]:
        raise ValueError("draft embeddings and masks do not align")

    # As in the paper runs, capacity is static, but every active source is
    # scattered contiguously from the left for each row.  Leaving masked answer
    # slots between P and D would change the later tokens' RoPE positions.
    image_tokens = int(image_readouts[0].shape[1])
    draft_width = int(draft_embeddings.shape[2])
    separator_width = int(separator_embeddings.shape[1])
    terminal_draft_width = (
        0
        if terminal_draft_embeddings is None
        else int(terminal_draft_embeddings.shape[1])
    )
    capacity = (
        depth * (image_tokens + text_width)
        + (depth - 1) * (draft_width + separator_width)
        + terminal_draft_width
    )
    if capacity > max_length:
        raise ValueError(
            f"Variant-C trajectory capacity {capacity} exceeds "
            f"max_length={max_length}"
        )
    embeddings = jnp.zeros((batch, capacity, embed_width), text_embeddings.dtype)
    input_mask = jnp.zeros((batch, capacity), dtype=bool)
    packed_ar = jnp.zeros((batch, capacity), dtype=jnp.int32)
    cursor = jnp.zeros((batch,), dtype=jnp.int32)
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    current_text_offset = jnp.zeros((batch,), dtype=jnp.int32)
    current_text_positions = jnp.full(
        (batch, text_width), -1, dtype=jnp.int32
    )

    def scatter(output, valid_out, ar_out, cursor_in, values, valid, ar_values):
        if values.ndim != 3 or valid.shape != values.shape[:2]:
            raise ValueError(
                "trajectory segment must be embeddings [B,T,D] + mask [B,T]"
            )
        if ar_values.shape != valid.shape:
            raise ValueError("trajectory segment mask_ar must match its mask")
        valid = valid.astype(bool)
        rank = jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1
        positions = cursor_in[:, None] + rank
        destination = jnp.where(valid, positions, capacity)
        output = output.at[batch_index, destination].set(values, mode="drop")
        valid_out = valid_out.at[batch_index, destination].set(valid, mode="drop")
        ar_out = ar_out.at[batch_index, destination].set(
            jnp.where(valid, ar_values, 0).astype(jnp.int32), mode="drop"
        )
        return (
            output,
            valid_out,
            ar_out,
            cursor_in + jnp.sum(valid, axis=1, dtype=jnp.int32),
            positions,
        )

    for round_index in range(depth):
        image = image_readouts[round_index]
        if image.ndim != 3 or image.shape[0] != batch or image.shape[-1] != embed_width:
            raise ValueError("all image readouts must be [B,N,D] matching text")
        image_valid = jnp.ones(image.shape[:2], dtype=bool)
        image_ar = _image_mask_ar(
            batch,
            image.shape[1],
            starts_new_block=(round_index > 0),
        )
        embeddings, input_mask, packed_ar, cursor, _ = scatter(
            embeddings,
            input_mask,
            packed_ar,
            cursor,
            image,
            image_valid,
            image_ar,
        )

        is_terminal = round_index == depth - 1
        prompt_valid = (
            text_input_mask.astype(bool)
            if is_terminal and terminal == "answer"
            else prefix_mask
        )
        prompt_ar = (
            text_mask_ar.astype(jnp.int32)
            if is_terminal and terminal == "answer"
            else jnp.zeros_like(text_mask_ar, dtype=jnp.int32)
        )
        if is_terminal:
            current_text_offset = cursor
        embeddings, input_mask, packed_ar, cursor, text_positions = scatter(
            embeddings,
            input_mask,
            packed_ar,
            cursor,
            text_embeddings,
            prompt_valid,
            prompt_ar,
        )
        if is_terminal:
            current_text_positions = jnp.where(
                prompt_valid, text_positions, -1
            )

        if not is_terminal:
            draft = draft_embeddings[:, round_index]
            valid = draft_mask[:, round_index].astype(bool)
            embeddings, input_mask, packed_ar, cursor, _ = scatter(
                embeddings,
                input_mask,
                packed_ar,
                cursor,
                draft,
                valid,
                valid.astype(jnp.int32),
            )
            separator_valid = jnp.ones(
                separator_embeddings.shape[:2], dtype=bool
            )
            embeddings, input_mask, packed_ar, cursor, _ = scatter(
                embeddings,
                input_mask,
                packed_ar,
                cursor,
                separator_embeddings,
                separator_valid,
                jnp.ones_like(separator_valid, dtype=jnp.int32),
            )

    if terminal == "draft" and terminal_draft_embeddings is not None:
        if terminal_draft_mask is None:
            raise ValueError("terminal_draft_mask is required with its embeddings")
        valid = terminal_draft_mask.astype(bool)
        embeddings, input_mask, packed_ar, cursor, _ = scatter(
            embeddings,
            input_mask,
            packed_ar,
            cursor,
            terminal_draft_embeddings,
            valid,
            valid.astype(jnp.int32),
        )
    physical = jnp.arange(embeddings.shape[1])[None]
    last = jnp.max(jnp.where(input_mask, physical, -1), axis=1)
    return TrajectoryBatch(
        embeddings=embeddings,
        input_mask=input_mask,
        mask_ar=packed_ar,
        current_text_offset=current_text_offset,
        current_text_positions=current_text_positions,
        last_active_index=last,
        depth=int(depth),
    )


def gather_last_slot(states, trajectory: TrajectoryBatch):
    """Read each row's last physical active slot (the next-token position).

    Works for any per-slot output ``[B,T,...]``, e.g. Gemma pre-logits.
    """

    if states.shape[:2] != trajectory.embeddings.shape[:2]:
        raise ValueError("outputs do not align with the packed trajectory")
    return jnp.take_along_axis(
        states,
        trajectory.last_active_index[:, None, None],
        axis=1,
    )[:, 0]


def gather_text_slots(states, trajectory: TrajectoryBatch):
    """Gather terminal-round outputs back into original text-slot order.

    The model gathers Gemma pre-logits and projects only these text slots.
    """

    if states.shape[:2] != trajectory.embeddings.shape[:2]:
        raise ValueError("outputs do not align with the packed trajectory")
    positions = jnp.maximum(trajectory.current_text_positions, 0)
    gathered = jnp.take_along_axis(states, positions[..., None], axis=1)
    return jnp.where(
        (trajectory.current_text_positions >= 0)[..., None], gathered, 0
    )


def suppress_logits(logits, suppressed_token_ids: Sequence[int]):
    """Mask invalid draft tokens while leaving EOS available as a stop event."""

    for token_id in tuple(int(value) for value in suppressed_token_ids):
        if token_id < 0 or token_id >= logits.shape[-1]:
            raise ValueError(f"suppressed token id {token_id} is out of range")
        logits = logits.at[:, token_id].set(jnp.asarray(-jnp.inf, logits.dtype))
    return logits


def greedy_reprefill_drafts(
    *,
    image_readouts: Sequence[Any],
    text_embeddings,
    text_input_mask,
    text_mask_ar,
    separator_embeddings,
    embed_tokens: Callable[[Any], Any],
    logits_for_trajectory: Callable[[TrajectoryBatch], Any],
    eos_token_id: int,
    suppressed_token_ids: Sequence[int],
    draft_max_tokens: int = PAPER_DRAFT_MAX_TOKENS,
    max_length: int = PAPER_MAX_SEQUENCE_LENGTH,
):
    """Generate every nonterminal draft with exact full-KV reprefill.

    ``logits_for_trajectory`` must perform a fresh language-model forward over
    the supplied full prompt.  Passing a cache-append implementation would
    violate the public Variant-C contract.  It returns each row's next-token
    logits ``[B,V]``, projecting only the last active slot (see
    :func:`gather_last_slot`) as the model does, or per-slot logits
    ``[B,T,V]``, from which that slot is read.  The per-slot form builds
    full-vocabulary logits for every packed slot: at paper size 2.5 GiB per
    example and draft token at training depth 3 and 6.25 GiB at depth 6.
    """

    steps = len(image_readouts)
    if steps < 1 or draft_max_tokens < 1:
        raise ValueError("steps and draft_max_tokens must be positive")
    batch = text_embeddings.shape[0]
    drafts = jnp.zeros((batch, steps - 1, draft_max_tokens), jnp.int32)
    draft_mask = jnp.zeros_like(drafts, dtype=bool)

    for depth in range(1, steps):
        alive = jnp.ones((batch,), dtype=bool)
        for token_index in range(draft_max_tokens):
            embedded_drafts = embed_tokens(drafts)
            current_embeddings = embedded_drafts[:, depth - 1]
            current_mask = draft_mask[:, depth - 1]
            trajectory = pack_trajectory(
                image_readouts,
                text_embeddings,
                text_input_mask,
                text_mask_ar,
                embedded_drafts,
                draft_mask,
                separator_embeddings,
                depth=depth,
                terminal="draft",
                terminal_draft_embeddings=current_embeddings,
                terminal_draft_mask=current_mask,
                max_length=max_length,
            )
            logits = logits_for_trajectory(trajectory)
            if logits.ndim == 3:
                logits = gather_last_slot(logits, trajectory)
            token = jnp.argmax(
                suppress_logits(logits, suppressed_token_ids), axis=-1
            ).astype(jnp.int32)
            retain = alive & (token != int(eos_token_id))
            drafts = drafts.at[:, depth - 1, token_index].set(
                jnp.where(retain, token, 0)
            )
            draft_mask = draft_mask.at[:, depth - 1, token_index].set(retain)
            alive = retain
    return jax.lax.stop_gradient(drafts), jax.lax.stop_gradient(draft_mask)


def rollout_reprefill_with_apply(
    model,
    params,
    *,
    image,
    text,
    mask_ar,
    input_mask,
    condition_input_ids,
    condition_attention_mask,
    steps: int,
    draft_max_tokens: int,
    eos_token_id: int,
    suppressed_token_ids: Sequence[int],
):
    """Efficient exact Variant-C rollout using one fresh prefill per round.

    This is the training/inference orchestration path used for the paper.
    Each round re-prefills the complete accumulated
    trajectory, then uses an exact causal KV cache while producing that
    round's draft.  The cache is discarded before the next round.
    """

    if steps < 1 or draft_max_tokens < 1:
        raise ValueError("steps and draft_max_tokens must be positive")
    image_steps, text_embeddings, separator_embeddings = model.apply(
        {"params": params},
        image,
        text,
        mask_ar,
        condition_input_ids=condition_input_ids,
        condition_attention_mask=condition_attention_mask,
        steps=int(steps),
        method=model.variant_c_sources,
    )
    batch = text.shape[0]
    drafts = jnp.zeros((batch, steps - 1, draft_max_tokens), jnp.int32)
    draft_mask = jnp.zeros_like(drafts, dtype=bool)
    for depth in range(1, steps):
        trajectory = model.apply(
            {"params": params},
            image_steps,
            text_embeddings,
            input_mask,
            mask_ar,
            drafts,
            draft_mask,
            separator_embeddings,
            depth=depth,
            method=model.variant_c_prompt,
        )
        logits, mutable = model.apply(
            {"params": params},
            trajectory.embeddings,
            trajectory.input_mask,
            trajectory.mask_ar,
            cache_size=trajectory.embeddings.shape[1] + draft_max_tokens,
            method=model.prefill_cache,
            mutable=("cache",),
        )
        cache = mutable["cache"]
        alive = jnp.ones((batch,), dtype=bool)
        for token_index in range(draft_max_tokens):
            token_logits = logits[:, -1] if logits.ndim == 3 else logits
            token = jnp.argmax(
                suppress_logits(token_logits, suppressed_token_ids), axis=-1
            ).astype(jnp.int32)
            retain = alive & (token != int(eos_token_id))
            drafts = drafts.at[:, depth - 1, token_index].set(
                jnp.where(retain, token, 0)
            )
            draft_mask = draft_mask.at[:, depth - 1, token_index].set(retain)
            alive = retain
            if token_index + 1 < draft_max_tokens:
                logits, mutable = model.apply(
                    {"params": params, "cache": cache},
                    token[:, None],
                    method=model.extend_cache,
                    mutable=("cache",),
                )
                cache = mutable["cache"]
    return jax.lax.stop_gradient(drafts), jax.lax.stop_gradient(draft_mask)


__all__ = [
    "PAPER_TRAJECTORY_MAX_LENGTH",
    "TrajectoryBatch",
    "gather_last_slot",
    "gather_text_slots",
    "greedy_reprefill_drafts",
    "pack_trajectory",
    "rollout_reprefill_with_apply",
    "suppress_logits",
]
