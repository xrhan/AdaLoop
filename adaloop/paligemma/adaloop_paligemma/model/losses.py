"""Paper-facing objectives for PaliGemma AdaLoop."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp

from adaloop_common.depth_weights import geometric_depth_weights


def causal_cross_entropy_per_example(logits: Any, targets: Any, loss_mask: Any):
    """Average answer-token CE within rows, then average examples.

    The PaliGemma paper runs use this historical reduction, including for the
    SFT/AdaLN controls.  Rows with an empty mask contribute zero; public data
    validation should reject such rows before training.
    """

    if logits.shape[:-1] != targets.shape or targets.shape != loss_mask.shape:
        raise ValueError(
            "logits [B,T,V], targets [B,T], and loss_mask [B,T] must align"
        )
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(
        log_prob, targets[..., None], axis=-1
    ).squeeze(-1)
    mask = loss_mask.astype(selected.dtype)
    per_example = -jnp.sum(selected * mask, axis=-1) / jnp.maximum(
        jnp.sum(mask, axis=-1), 1
    )
    return jnp.mean(per_example)


def combine_deep_supervision(losses: Sequence[Any], geometric_ratio: float):
    """Combine scalar per-depth losses with normalized geometric weights.

    At the paper depth ``3`` and ratio ``0.8`` the weights are proportional to
    ``[.8**2, .8, 1]``.
    """

    if not losses:
        raise ValueError("deep supervision requires at least one loss")
    weights = jnp.asarray(
        geometric_depth_weights(len(losses), geometric_ratio), jnp.float32
    )
    values = jnp.stack([jnp.asarray(loss, jnp.float32) for loss in losses])
    return jnp.sum(weights * values), weights


__all__ = [
    "causal_cross_entropy_per_example",
    "combine_deep_supervision",
]
