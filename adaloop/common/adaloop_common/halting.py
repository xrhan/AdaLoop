"""Answer-stability inference used in the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def normalize_for_stability(answer: str) -> str:
    """Normalize case and whitespace before comparing adjacent answers."""

    if not isinstance(answer, str):
        raise TypeError(f"answer must be a string; got {type(answer).__name__}")
    return " ".join(answer.lower().split())


@dataclass(frozen=True)
class P1Selection:
    """The answer chosen by one-step (``p=1``) stability stopping."""

    answer: str
    depth: int
    stabilized: bool


def select_p1_answer(
    predictions: Sequence[str], *, fallback_depth: int | None = None
) -> P1Selection:
    """Select the first answer repeated at two consecutive depths.

    ``predictions[i]`` is the greedy answer at depth ``i + 1``. If no adjacent
    pair matches after case/whitespace normalization, the answer at
    ``fallback_depth`` is returned. The final available depth is the default
    fallback. Task scorers are intentionally not used in this stopping rule.
    """

    if len(predictions) < 2:
        raise ValueError("p=1 answer stability requires at least two predictions")
    if any(not isinstance(answer, str) for answer in predictions):
        raise TypeError("all predictions must be strings")

    if fallback_depth is None:
        fallback_depth = len(predictions)
    if (
        isinstance(fallback_depth, bool)
        or not isinstance(fallback_depth, int)
        or not 1 <= fallback_depth <= len(predictions)
    ):
        raise ValueError(
            "fallback_depth must index an available 1-based depth; "
            f"got {fallback_depth!r} for {len(predictions)} predictions"
        )

    normalized = tuple(normalize_for_stability(answer) for answer in predictions)
    for index in range(1, len(normalized)):
        if normalized[index] == normalized[index - 1]:
            return P1Selection(
                answer=predictions[index], depth=index + 1, stabilized=True
            )

    return P1Selection(
        answer=predictions[fallback_depth - 1],
        depth=fallback_depth,
        stabilized=False,
    )
