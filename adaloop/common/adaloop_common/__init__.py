"""Framework-neutral public contracts for AdaLoop."""

from .config import LoopSpan, Variant
from .depth_weights import geometric_depth_weights
from .halting import P1Selection, normalize_for_stability, select_p1_answer
from .layouts import layout_notation

__all__ = [
    "LoopSpan",
    "P1Selection",
    "Variant",
    "geometric_depth_weights",
    "layout_notation",
    "normalize_for_stability",
    "select_p1_answer",
]
