"""Symbolic language-prefix layouts for AdaLoop Variants A, B, and C."""

from __future__ import annotations

from .config import Variant


def layout_notation(variant: Variant, depth: int) -> str:
    """Return the language prefix at ``depth``, such as ``[I1 | I2 | Q]``.

    ``In`` is the connector output for the visual state of round ``n``, ``Q``
    the question, ``Dn`` the tokens greedily generated in round ``n`` and
    ``Sn`` the separator after them. The target answer follows the prefix and
    is therefore not part of it.
    """

    if not isinstance(variant, Variant):
        raise ValueError("variant must be a Variant enum value")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError(f"depth must be a positive integer; got {depth!r}")
    if variant is Variant.A:
        segments = [f"I{depth}", "Q"]
    elif variant is Variant.B:
        segments = [f"I{index}" for index in range(1, depth + 1)] + ["Q"]
    else:
        segments = []
        for index in range(1, depth + 1):
            segments += [f"I{index}", "Q"]
            if index < depth:
                segments += [f"D{index}", f"S{index}"]
    return "[" + " | ".join(segments) + "]"
