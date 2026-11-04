"""Answer-token cross entropy used by the public Qwen2.5-VL trainers."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ..data import IGNORE_INDEX


def causal_cross_entropy_per_example(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Return one answer-token-mean causal CE value per batch row."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "Causal logits and labels must be [B, T, V] and [B, T]; got "
            f"{tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "Full sequence logits must align with labels; got "
            f"{tuple(logits.shape[:2])} and {tuple(labels.shape)}"
        )

    shifted_labels = labels[:, 1:]
    supervised = shifted_labels.ne(ignore_index)
    counts = supervised.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Every training row must supervise an answer token")

    packed_logits = logits[:, :-1][supervised]
    targets = shifted_labels[supervised].to(packed_logits.device)
    token_losses = F.cross_entropy(
        packed_logits.float(), targets, reduction="none"
    )
    row_ids = (
        torch.arange(supervised.shape[0], device=supervised.device)
        .unsqueeze(1)
        .expand_as(supervised)[supervised]
    )
    sums = torch.zeros(
        supervised.shape[0],
        device=token_losses.device,
        dtype=token_losses.dtype,
    )
    sums.scatter_add_(0, row_ids.to(token_losses.device), token_losses)
    return sums / counts.to(device=sums.device, dtype=sums.dtype)
