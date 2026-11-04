"""Question pooling and AdaLN modules for the Qwen2.5-VL backend.

The module hierarchy created by the public model intentionally uses the
checkpoint names ``question_conditioner`` and
``model.visual.blocks.<layer>.adaln``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class QuestionAttentionPooler(nn.Module):
    """Pool a masked sequence of detached question embeddings into one vector."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if condition_dim % num_heads:
            raise ValueError("condition_dim must be divisible by num_heads")
        self.condition_dim = int(condition_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.condition_dim // self.num_heads

        self.input_norm = nn.RMSNorm(input_dim, eps=1e-6)
        self.input_proj = (
            nn.Identity()
            if input_dim == condition_dim
            else nn.Linear(input_dim, condition_dim, bias=False)
        )
        self.query = nn.Parameter(torch.empty(1, condition_dim))
        self.q_proj = nn.Linear(condition_dim, condition_dim, bias=False)
        self.k_proj = nn.Linear(condition_dim, condition_dim, bias=False)
        self.v_proj = nn.Linear(condition_dim, condition_dim, bias=False)
        self.out_proj = nn.Linear(condition_dim, condition_dim, bias=False)

        hidden_dim = max(1, int(round(condition_dim * mlp_ratio)))
        self.mlp_norm = nn.RMSNorm(condition_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, condition_dim),
        )
        self.reset_query_parameters()

    def reset_query_parameters(self) -> None:
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.ndim != 3:
            raise ValueError(
                "Question embeddings must be [B, T, D], got "
                f"{tuple(token_embeddings.shape)}"
            )
        if token_mask.shape != token_embeddings.shape[:2]:
            raise ValueError(
                f"Question mask {tuple(token_mask.shape)} does not match "
                f"embeddings {tuple(token_embeddings.shape)}"
            )
        token_mask = token_mask.bool()
        if torch.any(~token_mask.any(dim=1)):
            raise ValueError("Every question condition must contain a token")

        # The newly initialized condition path remains FP32 even when the
        # pretrained vision/language paths run in BF16.
        with torch.autocast(device_type=token_embeddings.device.type, enabled=False):
            # Question embeddings are a conditioning input, not a second
            # language-model training path.  Detaching here makes that
            # contract true for every caller instead of relying on each data
            # or model wrapper to remember it.
            x = self.input_proj(
                self.input_norm(token_embeddings.detach().float())
            )
            batch, tokens, _ = x.shape
            query = self.query.unsqueeze(0).expand(batch, -1, -1)
            q = self.q_proj(query).view(batch, 1, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch, tokens, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch, tokens, self.num_heads, self.head_dim)
            logits = torch.einsum("bqhd,bthd->bhqt", q, k) / math.sqrt(
                self.head_dim
            )
            logits = logits.masked_fill(
                ~token_mask[:, None, None, :], torch.finfo(logits.dtype).min
            )
            weights = torch.softmax(logits, dim=-1)
            pooled = torch.einsum("bhqt,bthd->bqhd", weights, v).reshape(
                batch, 1, self.condition_dim
            )
            pooled = self.out_proj(pooled)
            output = query + pooled
            output = output + self.mlp(self.mlp_norm(output))
            return output[:, 0, :]


class AdaLNZero(nn.Module):
    """Independent identity-initialized modulation for one vision block."""

    def __init__(
        self,
        condition_dim: int,
        vision_width: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.to_adaln = nn.Linear(hidden_dim, 6 * vision_width)
        nn.init.zeros_(self.to_adaln.weight)
        nn.init.zeros_(self.to_adaln.bias)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = self.to_adaln(
            F.silu(self.condition_proj(self.condition_norm(condition)))
        )
        return values.chunk(6, dim=-1)
