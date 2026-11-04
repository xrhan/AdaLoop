"""Question pooling and AdaLN modules for the PaliGemma AdaLoop backend.

The public PaliGemma recipe conditions vision on a *separately tokenized* bare
question: no BOS, EOS, role marker, or answer cue, and at most 128 tokens.  The
normal PaliGemma language trajectory is intentionally unaffected and keeps its
native BOS/prefix formatting.

Parameter names in this module match the released checkpoints.  In particular
the parent model must instantiate :class:`QuestionConditioner` as
``question_conditioner`` and every adaptive vision block owns an ``adaln``
submodule.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..constants import (
    PAPER_ADALN_HIDDEN_DIM,
    PAPER_CONDITION_DIM,
    PAPER_CONDITION_HEADS,
    PAPER_QUESTION_MAX_TOKENS,
)


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm with a zero-initialized residual scale.

    The pooler's checkpoints were trained with this ``rsqrt`` form; the Gemma
    tower's own norm (``gemma_lora.RMSNorm``) divides by ``sqrt`` instead.
    """

    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param(
            "scale", nn.initializers.zeros_init(), (x.shape[-1],)
        )
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(variance + self.eps) * (1 + scale)


def compact_question_tokens(
    values, mask, *, max_tokens: int = PAPER_QUESTION_MAX_TOKENS
):
    """Left-compact active tokens into a static-width JAX tensor.

    ``values`` may be token IDs ``[B,T]`` or states ``[B,T,...]``.  The first
    ``max_tokens`` active positions are retained.  This helper does not remove
    BOS itself: callers must pass the dedicated bare-question sequence.
    """

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if mask.ndim != 2 or values.shape[:2] != mask.shape:
        raise ValueError(
            "values and mask must share leading [batch,tokens] dimensions; "
            f"got {values.shape} and {mask.shape}"
        )
    positions = jnp.arange(mask.shape[1])[None, :]
    ordered = jnp.sort(jnp.where(mask, positions, mask.shape[1]), axis=1)
    if max_tokens > mask.shape[1]:
        ordered = jnp.pad(
            ordered,
            ((0, 0), (0, max_tokens - mask.shape[1])),
            constant_values=mask.shape[1],
        )
    raw_indices = ordered[:, :max_tokens]
    selected_mask = raw_indices < mask.shape[1]
    indices = jnp.where(selected_mask, raw_indices, 0)
    gather = indices
    for _ in range(values.ndim - 2):
        gather = gather[..., None]
    gather = jnp.broadcast_to(gather, indices.shape + values.shape[2:])
    selected = jnp.take_along_axis(values, gather, axis=1)
    return selected, selected_mask, indices


class AttentionPooler(nn.Module):
    """The paper's shared learned-query question pooler.

    The attention computation is kept explicit to preserve the historical
    parameter leaves ``q_proj``, ``k_proj``, ``v_proj``, and ``out_proj``.
    Newly added conditioning arithmetic is FP32 even when the pretrained
    language and vision towers use BF16 parameters.
    """

    cond_dim: int = PAPER_CONDITION_DIM
    num_heads: int = PAPER_CONDITION_HEADS

    @nn.compact
    def __call__(self, question_states, question_mask):
        if question_states.ndim != 3:
            raise ValueError("question_states must have shape [B,T,D]")
        if question_mask.shape != question_states.shape[:2]:
            raise ValueError("question_mask must have shape [B,T]")
        if self.cond_dim % self.num_heads:
            raise ValueError("cond_dim must be divisible by num_heads")

        # The released checkpoints learned this branch in FP32.
        x = RMSNorm(name="input_norm")(question_states.astype(jnp.float32))
        x = nn.Dense(self.cond_dim, use_bias=False, name="input_proj")(x)

        # One learned summary query.
        query = self.param(
            "query", nn.initializers.normal(0.02), (1, self.cond_dim)
        )
        query = jnp.broadcast_to(query[None], (x.shape[0], 1, self.cond_dim))
        head_dim = self.cond_dim // self.num_heads
        q = nn.DenseGeneral(
            (self.num_heads, head_dim), use_bias=False, name="q_proj"
        )(query)
        k = nn.DenseGeneral(
            (self.num_heads, head_dim), use_bias=False, name="k_proj"
        )(x)
        v = nn.DenseGeneral(
            (self.num_heads, head_dim), use_bias=False, name="v_proj"
        )(x)
        logits = jnp.einsum("bqnh,bknh->bnqk", q, k).astype(jnp.float32)
        logits *= head_dim**-0.5
        active = question_mask.astype(bool)[:, None, None, :]
        logits = jnp.where(active, logits, jnp.asarray(-2.3819763e38, logits.dtype))
        weights = jax.nn.softmax(logits, axis=-1)
        weights = jnp.where(active, weights, 0).astype(v.dtype)
        pooled = jnp.einsum("bnqk,bknh->bqnh", weights, v)
        pooled = nn.DenseGeneral(
            self.cond_dim, axis=(-2, -1), use_bias=False, name="out_proj"
        )(pooled)
        output = query + pooled
        residual = RMSNorm(name="mlp_norm")(output)
        residual = nn.Dense(2 * self.cond_dim, name="mlp_in")(residual)
        residual = nn.gelu(residual)
        residual = nn.Dense(self.cond_dim, name="mlp_out")(residual)
        output = output + residual
        has_tokens = jnp.any(question_mask, axis=1)
        output = jnp.where(has_tokens[:, None, None], output, 0)
        return output, {
            "summary_vector": output[:, 0],
            "attn_weights": weights,
            "has_question": has_tokens,
        }


class QuestionConditioner(nn.Module):
    """Embed and pool a dedicated bare-question token sequence.

    ``question_ids`` must already follow the public data contract: no BOS/EOS
    and no chat/prompt tokens.  Keeping this contract explicit prevents the
    common bug where removing BOS here drops the first real question token a
    second time.
    """

    max_tokens: int = PAPER_QUESTION_MAX_TOKENS
    cond_dim: int = PAPER_CONDITION_DIM
    num_heads: int = PAPER_CONDITION_HEADS
    detach: bool = True

    @nn.compact
    def __call__(
        self,
        llm,
        question_ids,
        question_mask,
        *,
        train: bool = False,
    ):
        if question_ids.ndim != 2 or question_mask.shape != question_ids.shape:
            raise ValueError("question_ids/question_mask must both be [B,T]")
        ids, mask, indices = compact_question_tokens(
            question_ids,
            question_mask.astype(bool),
            max_tokens=self.max_tokens,
        )
        states = llm.embed_tokens(ids, train=train)
        if self.detach:
            states = jax.lax.stop_gradient(states)
        states = jnp.where(mask[..., None], states, 0)
        summary, aux = AttentionPooler(
            cond_dim=self.cond_dim, num_heads=self.num_heads, name="pooler"
        )(states, mask)
        aux.update(
            question_states=states,
            question_mask=mask,
            question_tokens=ids,
            question_token_indices=indices,
        )
        return summary, aux


class AdaLNBranch(NamedTuple):
    shift: Any
    scale: Any
    gate: Any


class AdaLNParams(NamedTuple):
    attn: AdaLNBranch
    mlp: AdaLNBranch


def cond_to_vector(condition):
    if condition.ndim == 2:
        return condition
    if condition.ndim == 3 and condition.shape[1] == 1:
        return condition[:, 0]
    raise ValueError(
        "AdaLN condition must be [B,D] or [B,1,D], got "
        f"{condition.shape}"
    )


def split_adaln_params(values) -> AdaLNParams:
    pieces = jnp.split(values, 6, axis=-1)
    return AdaLNParams(
        attn=AdaLNBranch(*pieces[:3]), mlp=AdaLNBranch(*pieces[3:])
    )


def modulate(value, shift, scale):
    return value * (1 + scale[:, None, :]) + shift[:, None, :]


def gate_residual(value, gate):
    """Scale only the branch update; zero gate preserves pretrained behavior."""

    return value * (1 + gate[:, None, :])


class IndependentAdaLNZero(nn.Module):
    """Checkpoint-compatible per-block independent AdaLN generator."""

    vision_width: int
    hidden_dim: int = PAPER_ADALN_HIDDEN_DIM

    @nn.compact
    def __call__(self, condition) -> AdaLNParams:
        if self.vision_width < 1 or self.hidden_dim < 1:
            raise ValueError("vision_width and hidden_dim must be positive")
        x = cond_to_vector(condition).astype(jnp.float32)
        x = nn.LayerNorm(name="cond_norm")(x)
        x = nn.Dense(self.hidden_dim, name="cond_proj")(x)
        x = nn.silu(x)
        values = nn.Dense(
            6 * self.vision_width,
            use_bias=True,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="to_adaln",
        )(x)
        return split_adaln_params(values)


__all__ = [
    "AdaLNBranch",
    "AdaLNParams",
    "AttentionPooler",
    "IndependentAdaLNZero",
    "QuestionConditioner",
    "RMSNorm",
    "compact_question_tokens",
    "cond_to_vector",
    "gate_residual",
    "modulate",
]
