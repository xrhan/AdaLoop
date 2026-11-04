"""Role-free prompt tokenization for the Qwen2.5-VL AdaLoop paths.

The vision conditioner and Variant-C language trajectory deliberately bypass
Qwen's chat template.  Training (:class:`~adaloop_qwen.data.QwenImageQADataset`)
and inference both tokenize these fragments here, so their token IDs agree by
construction and can be tested without constructing a Transformers model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import torch

from ..config import CONDITION_MAX_TOKENS


PaddingSide = Literal["left", "right"]
VARIANT_C_ANSWER_CUE = "\nAnswer:"
VARIANT_C_SEPARATOR = "\n"


class TokenizerLike(Protocol):
    """The tokenizer surface used by this module."""

    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None
    padding_side: str

    def encode(
        self, text: str, *, add_special_tokens: bool = False
    ) -> Sequence[int]: ...


@dataclass(frozen=True)
class TokenBatch:
    """A padded batch with an explicit mask."""

    input_ids: torch.LongTensor
    attention_mask: torch.Tensor

    @property
    def active_lengths(self) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self.attention_mask.sum(dim=1).detach().cpu().tolist()
        )


@dataclass(frozen=True)
class VariantCPromptFragments:
    """Static, role-free fragments used to assemble Variant-C trajectories."""

    question: TokenBatch
    cue: TokenBatch
    separator: TokenBatch
    answer: TokenBatch | None = None


def _require_texts(texts: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence of strings")
    values = tuple(texts)
    if not values:
        raise ValueError(f"{name} must contain at least one string")
    if any(not isinstance(text, str) for text in values):
        raise TypeError(f"Every {name} value must be a string")
    return values


def _padding_side(tokenizer: TokenizerLike, override: PaddingSide | None) -> PaddingSide:
    value = getattr(tokenizer, "padding_side", "right") if override is None else override
    if value not in {"left", "right"}:
        raise ValueError(f"padding_side must be 'left' or 'right'; got {value!r}")
    return value


def _pad_token_id(tokenizer: TokenizerLike, override: int | None) -> int:
    value = getattr(tokenizer, "pad_token_id", None) if override is None else override
    if value is None:
        raise ValueError("Role-free Qwen tokenization requires a pad_token_id")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"pad_token_id must be a non-negative integer; got {value!r}")
    return int(value)


def _encode_nonempty(
    tokenizer: TokenizerLike,
    text: str,
    *,
    name: str,
    allow_eos: bool = False,
) -> list[int]:
    ids = [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    if not ids:
        raise ValueError(f"{name} tokenized to an empty sequence")
    if any(token_id < 0 for token_id in ids):
        raise ValueError(f"{name} produced a negative token ID")
    forbidden = {
        int(token_id)
        for token_id in (
            getattr(tokenizer, "bos_token_id", None),
            None if allow_eos else getattr(tokenizer, "eos_token_id", None),
        )
        if token_id is not None
    }
    present = forbidden.intersection(ids)
    if present:
        raise ValueError(
            f"{name} unexpectedly contains BOS/EOS token IDs despite "
            f"add_special_tokens=False: {sorted(present)}"
        )
    return ids


def encode_question_condition(
    tokenizer: TokenizerLike,
    text: str,
    *,
    max_tokens: int = CONDITION_MAX_TOKENS,
) -> list[int]:
    """Token IDs of one bare AdaLN condition (no BOS, EOS, or chat wrapper).

    The first ``max_tokens`` tokens are kept.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Question condition must be a non-empty string")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    return _encode_nonempty(tokenizer, text, name="Question condition")[:max_tokens]


def encode_variant_c_fragments(
    tokenizer: TokenizerLike,
    question: str,
    answer: str | None = None,
) -> dict[str, list[int]]:
    """Token IDs of one row's Variant-C fragments, keyed ``question``,
    ``answer`` (only when given), ``cue``, and ``separator``.

    Questions, ``"\nAnswer:"``, and the newline separator are role-free and
    contain no automatically inserted special tokens.  The answer is
    whitespace-normalized at its outer boundary, keeps the leading space it
    has after the cue, and ends in exactly one EOS token.
    """

    fragments = {
        "question": _encode_nonempty(tokenizer, question, name="Variant-C question")
    }
    if answer is not None:
        content = answer.strip()
        if not content:
            raise ValueError("Variant-C answer must contain non-whitespace content")
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("Variant-C answer tokenization requires eos_token_id")
        eos_id = int(eos_id)
        ids = _encode_nonempty(
            tokenizer, f" {content}", name="Variant-C answer", allow_eos=True
        )
        while ids and ids[-1] == eos_id:
            ids.pop()
        if not ids:
            raise ValueError("Variant-C answer must contain a token before EOS")
        if eos_id in ids:
            raise ValueError(
                "Variant-C answer contains EOS before its final token; the "
                "trajectory contract requires exactly one terminal EOS"
            )
        fragments["answer"] = [*ids, eos_id]
    fragments["cue"] = _encode_nonempty(
        tokenizer, VARIANT_C_ANSWER_CUE, name="Variant-C answer cue"
    )
    fragments["separator"] = _encode_nonempty(
        tokenizer, VARIANT_C_SEPARATOR, name="Variant-C newline separator"
    )
    return fragments


def pad_token_rows(
    rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    padding_side: PaddingSide = "right",
    device: torch.device | str | None = None,
) -> TokenBatch:
    """Pad non-empty integer token rows without adding special tokens."""

    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    if not rows:
        raise ValueError("rows must contain at least one token sequence")
    normalized = [tuple(int(token_id) for token_id in row) for row in rows]
    if any(not row for row in normalized):
        raise ValueError("Every token row must be non-empty")
    if any(token_id < 0 for row in normalized for token_id in row):
        raise ValueError("Token IDs must be non-negative")

    width = max(len(row) for row in normalized)
    input_ids = torch.full(
        (len(normalized), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(normalized), width), dtype=torch.bool, device=device
    )
    for index, row in enumerate(normalized):
        values = torch.tensor(row, dtype=torch.long, device=device)
        start = width - len(row) if padding_side == "left" else 0
        input_ids[index, start : start + len(row)] = values
        attention_mask[index, start : start + len(row)] = True
    return TokenBatch(input_ids=input_ids, attention_mask=attention_mask)


def tokenize_question_conditions(
    tokenizer: TokenizerLike,
    texts: Sequence[str],
    *,
    max_tokens: int = CONDITION_MAX_TOKENS,
    pad_token_id: int | None = None,
    padding_side: PaddingSide | None = None,
    device: torch.device | str | None = None,
) -> TokenBatch:
    """Pad :func:`encode_question_condition` rows into a batch.

    Each text is the row's question, or the recipe's
    ``conditioner.constant_text`` for a constant-question recipe.
    """

    rows = [
        encode_question_condition(tokenizer, text, max_tokens=max_tokens)
        for text in _require_texts(texts, name="texts")
    ]
    return pad_token_rows(
        rows,
        pad_token_id=_pad_token_id(tokenizer, pad_token_id),
        padding_side=_padding_side(tokenizer, padding_side),
        device=device,
    )


def tokenize_variant_c_fragments(
    tokenizer: TokenizerLike,
    questions: Sequence[str],
    *,
    pad_token_id: int | None = None,
    padding_side: PaddingSide | None = None,
    device: torch.device | str | None = None,
) -> VariantCPromptFragments:
    """Pad the inference-time (answer-free) Variant-C fragments into batches."""

    rows = [
        encode_variant_c_fragments(tokenizer, question)
        for question in _require_texts(questions, name="questions")
    ]
    side = _padding_side(tokenizer, padding_side)
    pad_id = _pad_token_id(tokenizer, pad_token_id)

    def batch(name: str) -> TokenBatch:
        return pad_token_rows(
            [row[name] for row in rows],
            pad_token_id=pad_id,
            padding_side=side,
            device=device,
        )

    return VariantCPromptFragments(
        question=batch("question"), cue=batch("cue"), separator=batch("separator")
    )


__all__ = [
    "PaddingSide",
    "TokenBatch",
    "TokenizerLike",
    "VARIANT_C_ANSWER_CUE",
    "VARIANT_C_SEPARATOR",
    "VariantCPromptFragments",
    "encode_question_condition",
    "encode_variant_c_fragments",
    "pad_token_rows",
    "tokenize_question_conditions",
    "tokenize_variant_c_fragments",
]
