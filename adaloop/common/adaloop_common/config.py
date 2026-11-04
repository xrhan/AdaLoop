"""Framework-neutral loop vocabulary shared by the AdaLoop backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Variant(str, Enum):
    """Language interfaces defined in the AdaLoop paper."""

    A = "A"
    B = "B"
    C = "C"


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return value


@dataclass(frozen=True)
class LoopSpan:
    """A zero-indexed, half-open recurrent block interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        _require_int("loop_span.start", self.start, minimum=0)
        _require_int("loop_span.end", self.end, minimum=1)
        if self.end <= self.start:
            raise ValueError(
                "loop_span.end must be greater than loop_span.start; "
                f"got [{self.start}, {self.end})"
            )
