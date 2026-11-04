# Copyright 2024 Big Vision Authors.
# Copyright 2026 The AdaLoop Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the AdaLoop Authors from Big Vision
# (https://github.com/google-research/big_vision, commit 0127fb6b)
# big_vision/models/vit.py: restricted to the PaliGemma2 448px So400m/14
# tower, added question-conditioned AdaLN to the encoder blocks and a
# weight-tied recurrent block span with per-depth readouts, and removed the
# pooling heads, scanned execution, and checkpoint loading.

"""Checkpoint-compatible recurrent SigLIP So400m/14 vision tower.

This is the subset of the recurrent tower used in the paper.  Blocks ``[3, 15)``
form a weight-tied recurrent core; the prefix runs once and the suffix plus
connector form the readout.  At depth one every pretrained block executes
exactly once in its original order.

The Flax hierarchy deliberately matches Big Vision checkpoints::

  img/embedding
  img/pos_embedding
  img/Transformer/encoderblock_N/{LayerNorm_*,MultiHeadDotProductAttention_0,
                                  MlpBlock_0,adaln}
  img/Transformer/encoder_norm
  img/head

Only independent question-conditioned AdaLN is retained.  Step embeddings,
input injection, boundary normalization, and alternative recurrent modes are
not part of the paper models and are intentionally absent.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ..constants import (
    GEMMA_WIDTH,
    PAPER_ADALN_HIDDEN_DIM,
    PAPER_IMAGE_SIZE,
    PAPER_IMAGE_TOKENS,
    PAPER_LOOP_SPAN,
    PAPER_PATCH_SIZE,
    PAPER_VISION_DEPTH,
    PAPER_VISION_HEADS,
    PAPER_VISION_MLP_DIM,
    PAPER_VISION_WIDTH,
)
from .conditioning import (
    AdaLNParams,
    IndependentAdaLNZero,
    gate_residual,
    modulate,
)


class MlpBlock(nn.Module):
    """Big-Vision-compatible ViT MLP with the same child names."""

    mlp_dim: Optional[int] = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        init = dict(
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        width = x.shape[-1]
        x = nn.Dense(self.mlp_dim or 4 * width, dtype=self.dtype_mm, **init)(x)
        x = nn.with_logical_constraint(
            x, ("act_batch", "act_len", "act_emb")
        )
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(width, dtype=self.dtype_mm, **init)(x)


class AdaptiveEncoder1DBlock(nn.Module):
    """A stock pre-LN SigLIP block with optional identity-init AdaLN."""

    mlp_dim: int = PAPER_VISION_MLP_DIM
    num_heads: int = PAPER_VISION_HEADS
    dropout: float = 0.0
    dtype_mm: str = "float32"
    use_adaln: bool = False
    adaln_hidden_dim: int = PAPER_ADALN_HIDDEN_DIM

    @nn.compact
    def __call__(self, x, condition=None, deterministic: bool = True):
        mods: AdaLNParams | None = None
        if self.use_adaln:
            if condition is None:
                raise ValueError("condition is required by an AdaLN block")
            mods = IndependentAdaLNZero(
                vision_width=x.shape[-1],
                hidden_dim=self.adaln_hidden_dim,
                name="adaln",
            )(condition)

        x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
        y = nn.LayerNorm()(x)
        if mods is not None:
            y = modulate(y, mods.attn.shift, mods.attn.scale)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = nn.with_logical_constraint(
            y, ("act_batch", "act_len", "act_emb")
        )
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        if mods is not None:
            y = gate_residual(y, mods.attn.gate)
        x = x + y

        y = nn.LayerNorm()(x)
        if mods is not None:
            y = modulate(y, mods.mlp.shift, mods.mlp.scale)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = nn.with_logical_constraint(
            y, ("act_batch", "act_len", "act_emb")
        )
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        if mods is not None:
            y = gate_residual(y, mods.mlp.gate)
        x = x + y
        return nn.with_logical_constraint(
            x, ("act_batch", "act_len", "act_emb")
        )


class LoopedEncoder(nn.Module):
    """Prelude/core/coda encoder with a statically unrolled tied core."""

    depth: int = PAPER_VISION_DEPTH
    n_pre: int = PAPER_LOOP_SPAN[0]
    n_core: int = PAPER_LOOP_SPAN[1] - PAPER_LOOP_SPAN[0]
    loop_steps: int = 3
    mlp_dim: int = PAPER_VISION_MLP_DIM
    num_heads: int = PAPER_VISION_HEADS
    dropout: float = 0.0
    dtype_mm: str = "float32"
    remat_policy: str = "nothing_saveable"
    use_adaln: bool = True
    adaln_hidden_dim: int = PAPER_ADALN_HIDDEN_DIM

    @nn.compact
    def __call__(
        self,
        x,
        condition=None,
        *,
        deterministic: bool = True,
        return_all_steps: bool = False,
    ):
        if self.depth != PAPER_VISION_DEPTH:
            raise ValueError("the public backend supports the 27-block So400m tower")
        if (self.n_pre, self.n_pre + self.n_core) != PAPER_LOOP_SPAN:
            raise ValueError("the released PaliGemma loop span is [3,15)")
        if self.loop_steps < 1:
            raise ValueError("loop_steps must be positive")
        if self.use_adaln and condition is None:
            raise ValueError("question-conditioned execution requires condition")
        block_cls = AdaptiveEncoder1DBlock
        if self.remat_policy != "none":
            block_cls = nn.remat(
                AdaptiveEncoder1DBlock,
                prevent_cse=True,
                static_argnums=(3,),
                policy=getattr(jax.checkpoint_policies, self.remat_policy),
            )
        blocks = [
            block_cls(
                name=f"encoderblock_{index}",
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
                use_adaln=(
                    self.use_adaln
                    and self.n_pre <= index < self.n_pre + self.n_core
                ),
                adaln_hidden_dim=self.adaln_hidden_dim,
            )
            for index in range(self.depth)
        ]
        final_norm = nn.LayerNorm(name="encoder_norm")

        for index in range(self.n_pre):
            x = blocks[index](x, condition, deterministic)

        def run_core(value):
            for index in range(self.n_pre, self.n_pre + self.n_core):
                value = blocks[index](value, condition, deterministic)
            return value

        def readout(value):
            for index in range(self.n_pre + self.n_core, self.depth):
                value = blocks[index](value, condition, deterministic)
            return final_norm(value)

        if not return_all_steps:
            for _ in range(self.loop_steps):
                x = run_core(x)
            return readout(x)

        outputs = []
        for _ in range(self.loop_steps):
            x = run_core(x)
            outputs.append(readout(x))
        return tuple(outputs)


class LoopedSigLIPVision(nn.Module):
    """Fixed-448 PaliGemma2 SigLIP tower with AdaLoop execution."""

    num_classes: int = GEMMA_WIDTH
    loop_steps: int = 3
    use_adaln: bool = True
    image_size: int = PAPER_IMAGE_SIZE
    patch_size: Sequence[int] = PAPER_PATCH_SIZE
    width: int = PAPER_VISION_WIDTH
    depth: int = PAPER_VISION_DEPTH
    mlp_dim: int = PAPER_VISION_MLP_DIM
    num_heads: int = PAPER_VISION_HEADS
    n_pre: int = PAPER_LOOP_SPAN[0]
    n_core: int = PAPER_LOOP_SPAN[1] - PAPER_LOOP_SPAN[0]
    dropout: float = 0.0
    dtype_mm: str = "float32"
    remat_policy: str = "nothing_saveable"
    adaln_hidden_dim: int = PAPER_ADALN_HIDDEN_DIM

    @nn.compact
    def __call__(
        self,
        image,
        condition=None,
        *,
        train: bool = False,
        n_steps: int | None = None,
        return_all_steps: bool = False,
    ):
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("image must have shape [B,H,W,3]")
        if image.shape[1:3] != (self.image_size, self.image_size):
            raise ValueError(
                f"the released PaliGemma backend expects {self.image_size}x"
                f"{self.image_size} images, got {image.shape[1:3]}"
            )
        if tuple(self.patch_size) != PAPER_PATCH_SIZE:
            raise ValueError("the released backbone uses 14x14 patches")
        steps = self.loop_steps if n_steps is None else int(n_steps)
        if steps < 1:
            raise ValueError("n_steps must be positive")
        image = jnp.asarray(image, self.dtype_mm)
        x = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=self.dtype_mm,
        )(image)
        batch, height, width, channels = x.shape
        x = jnp.reshape(x, (batch, height * width, channels))
        if x.shape[1] != PAPER_IMAGE_TOKENS:
            raise ValueError("PaliGemma2 Mix-448 must produce 1024 image tokens")
        x = x + self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1 / np.sqrt(channels)),
            (1, height * width, channels),
            x.dtype,
        )
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        encoded = LoopedEncoder(
            depth=self.depth,
            n_pre=self.n_pre,
            n_core=self.n_core,
            loop_steps=steps,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
            remat_policy=self.remat_policy,
            use_adaln=self.use_adaln,
            adaln_hidden_dim=self.adaln_hidden_dim,
            name="Transformer",
        )(
            x,
            condition,
            deterministic=not train,
            return_all_steps=return_all_steps,
        )
        encoded_steps = encoded if return_all_steps else (encoded,)
        head = nn.Dense(self.num_classes, name="head")
        image_steps = tuple(head(value) for value in encoded_steps)
        out = {"logits": image_steps[-1]}
        if return_all_steps:
            out["zimg_steps"] = image_steps
        return image_steps[-1], out


def unstack_scanned_vision_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert Big Vision's scanned block subtree to named block scopes.

    Base PaliGemma checkpoints are sometimes distributed with
    ``img/Transformer/encoderblock`` leaves stacked along axis zero, while the
    tied-loop Python unroll needs ``encoderblock_0`` ... ``encoderblock_26``.
    """

    transformer = params.get("Transformer", params)
    scanned = transformer.get("encoderblock")
    if scanned is None:
        return params

    def take(tree, index):
        return jax.tree.map(lambda value: value[index], tree)

    result = dict(params)
    target = dict(transformer)
    target.pop("encoderblock")
    # Infer depth from the first leaf rather than trusting a config field.
    leaves = jax.tree.leaves(scanned)
    if not leaves:
        raise ValueError("scanned encoderblock subtree contains no leaves")
    depth = int(leaves[0].shape[0])
    for index in range(depth):
        target[f"encoderblock_{index}"] = take(scanned, index)
    if "Transformer" in result:
        result["Transformer"] = target
    else:
        result = target
    return result


__all__ = [
    "AdaptiveEncoder1DBlock",
    "LoopedEncoder",
    "LoopedSigLIPVision",
    "MlpBlock",
    "unstack_scanned_vision_params",
]
