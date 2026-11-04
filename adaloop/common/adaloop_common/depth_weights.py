"""Depth-weight schedules shared by the backend implementations."""

from __future__ import annotations

import math


def geometric_depth_weights(max_depth: int, ratio: float) -> tuple[float, ...]:
    """Return normalized weights ``w_t proportional to ratio**(T - t)``.

    Entries are ordered from depth 1 through ``max_depth``. Ratios in ``(0, 1]``
    assign at least as much weight to a deeper readout as to an earlier one.
    """

    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError(f"max_depth must be a positive integer; got {max_depth!r}")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError(f"ratio must be a finite number in (0, 1]; got {ratio!r}")
    ratio = float(ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be a finite number in (0, 1]; got {ratio!r}")

    unnormalized = tuple(ratio ** (max_depth - depth) for depth in range(1, max_depth + 1))
    total = math.fsum(unnormalized)
    return tuple(weight / total for weight in unnormalized)
