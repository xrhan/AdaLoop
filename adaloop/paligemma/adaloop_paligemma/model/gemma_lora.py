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
# big_vision/models/ppp/gemma.py and
# big_vision/models/proj/paligemma/gemma_bv.py: restricted to the PaliGemma2
# Gemma-2 2B language tower, added LoRA on the q/kv/o attention projections
# and a pre-logits-only forward, made the decode cache track its valid slots
# explicitly, and removed checkpoint loading.

"""Self-contained Gemma-2 2B language tower with checkpoint-compatible LoRA.

This is a narrowly scoped extraction of the Apache-2.0 Big Vision Gemma
implementation used by PaliGemma2.  Base parameter names and scanned layer
axes are unchanged; LoRA adds only ``q_lora_*``, ``kv_lora_*``, and
``o_lora_*`` under ``llm/layers/attn``.

The implementation intentionally targets the PaliGemma2 3B Mix language
tower (Gemma-2 2B).  It is not a general Gemma model zoo.
"""

from __future__ import annotations

from typing import Any, Iterable

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ..constants import (
    GEMMA_DEPTH,
    GEMMA_HEAD_DIM,
    GEMMA_HEADS,
    GEMMA_KV_HEADS,
    GEMMA_MLP_DIM,
    GEMMA_VOCAB_SIZE,
    GEMMA_WIDTH,
    PAPER_LORA_ALPHA,
    PAPER_LORA_RANK,
    PAPER_LORA_TARGETS,
)


def _targets(values: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    result = tuple(str(value) for value in values)
    unknown = set(result) - {"q", "kv", "o"}
    if unknown:
        raise ValueError(f"unsupported LoRA targets: {sorted(unknown)}")
    return result


def trunc_norm_init(in_axis, out_axis, batch_axis):
    return nn.initializers.variance_scaling(
        1.0,
        "fan_in",
        "truncated_normal",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
    )


def _lora_a_init(std: float):
    def initialize(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * std

    return initialize


def apply_rope(value, *, positions, max_wavelength: float = 10_000.0):
    exponent = (2.0 / value.shape[-1]) * jnp.arange(value.shape[-1] // 2)
    timescale = max_wavelength**exponent
    radians = positions[..., None] / timescale[None, None]
    radians = radians[..., None, :]
    sine, cosine = jnp.sin(radians), jnp.cos(radians)
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate(
        [first * cosine - second * sine, second * cosine + first * sine],
        axis=-1,
    )


class Einsum(nn.Module):
    shape: tuple[int, ...]
    w_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, equation: str, value):
        weight = self.param("w", self.w_init, self.shape)
        return jnp.einsum(equation, value, weight)


class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, value):
        scale = self.param(
            "scale", nn.initializers.zeros_init(), (value.shape[-1],)
        )
        variance = jnp.mean(jnp.square(value), axis=-1, keepdims=True)
        normalized = jnp.asarray(
            value * jnp.reciprocal(jnp.sqrt(variance + 1e-6))
        )
        return normalized * (1 + scale)


class Embedder(nn.Module):
    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_in",
                distribution="normal",
                in_axis=1,
                out_axis=0,
            ),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, token_ids):
        value = self.input_embedding_table[(token_ids,)]
        return value * jnp.sqrt(self.embed_dim).astype(value.dtype)

    def decode(self, value):
        return jnp.dot(value, self.input_embedding_table.T)


class FeedForward(nn.Module):
    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, value):
        gating = self.param(
            "gating_einsum",
            trunc_norm_init(in_axis=(1,), out_axis=(0, 2), batch_axis=()),
            (2, self.features, self.hidden_dim),
        )
        gate = nn.gelu(jnp.dot(value, gating[0]))
        activations = gate * jnp.dot(value, gating[1])
        linear = self.param(
            "linear",
            trunc_norm_init(in_axis=(0,), out_axis=(1,), batch_axis=()),
            (self.hidden_dim, self.features),
        )
        return jnp.dot(activations, linear)


def _update_kv_cache(module, key, value, cache_size: int, cache_dtype):
    initialized = module.has_variable("cache", "idx")
    batch, update_length, heads, head_dim = key.shape
    dtype = cache_dtype or key.dtype
    index = module.variable("cache", "idx", jnp.zeros, (batch,), jnp.int32)
    shape = (batch, cache_size, heads, head_dim)
    key_cache = module.variable("cache", "k_cache", jnp.zeros, shape, dtype)
    value_cache = module.variable("cache", "v_cache", jnp.zeros, shape, dtype)
    if initialized:
        start = (0, index.value[0], 0, 0)
        key_cache.value = jax.lax.dynamic_update_slice(
            key_cache.value, key.astype(dtype), start
        )
        value_cache.value = jax.lax.dynamic_update_slice(
            value_cache.value, value.astype(dtype), start
        )
        index.value = index.value + update_length
    else:
        padding = ((0, 0), (0, cache_size - update_length), (0, 0), (0, 0))
        key_cache.value = jnp.pad(key.astype(dtype), padding)
        value_cache.value = jnp.pad(value.astype(dtype), padding)
        index.value = index.value + update_length
    return key_cache.value.astype(key.dtype), value_cache.value.astype(value.dtype)


class LoraAttention(nn.Module):
    num_heads: int
    num_kv_heads: int
    features: int
    head_dim: int
    attn_logits_softcap: float = 50.0
    cache_dtype: str | None = "bfloat16"
    lora_rank: int = PAPER_LORA_RANK
    lora_alpha: float = PAPER_LORA_ALPHA
    lora_targets: tuple[str, ...] = PAPER_LORA_TARGETS

    def setup(self):
        self.q_einsum = Einsum(
            shape=(self.num_heads, self.features, self.head_dim),
            w_init=trunc_norm_init(in_axis=(1,), out_axis=(0, 2), batch_axis=()),
        )
        self.kv_einsum = Einsum(
            shape=(2, self.num_kv_heads, self.features, self.head_dim),
            w_init=trunc_norm_init(
                in_axis=(2,), out_axis=(0, 1, 3), batch_axis=()
            ),
        )
        self.attn_vec_einsum = Einsum(
            shape=(self.num_heads, self.head_dim, self.features),
            w_init=trunc_norm_init(
                in_axis=(0, 1), out_axis=(2,), batch_axis=()
            ),
        )

    def _enabled(self, target: str) -> bool:
        return self.lora_rank > 0 and target in _targets(self.lora_targets)

    @property
    def _scale(self) -> float:
        return float(self.lora_alpha) / float(max(1, self.lora_rank))

    @nn.compact
    def __call__(
        self, value, positions, attention_mask, decode, deterministic=True
    ):
        del deterministic
        query = self.q_einsum("BTD,NDH->BTNH", value)
        key, values = self.kv_einsum("BSD,2KDH->2BSKH", value)

        if self._enabled("q"):
            left = self.param(
                "q_lora_a",
                _lora_a_init(self.features**-0.5),
                (self.features, self.lora_rank),
            )
            right = self.param(
                "q_lora_b",
                nn.initializers.zeros_init(),
                (self.num_heads, self.lora_rank, self.head_dim),
            )
            query += self._scale * jnp.einsum(
                "BTD,DR,NRH->BTNH", value, left, right
            )
        if self._enabled("kv"):
            left = self.param(
                "kv_lora_a",
                _lora_a_init(self.features**-0.5),
                (self.features, self.lora_rank),
            )
            right = self.param(
                "kv_lora_b",
                nn.initializers.zeros_init(),
                (2, self.num_kv_heads, self.lora_rank, self.head_dim),
            )
            delta = self._scale * jnp.einsum(
                "BSD,DR,ZKRH->ZBSKH", value, left, right
            )
            key, values = key + delta[0], values + delta[1]

        query = apply_rope(query, positions=positions)
        query *= self.head_dim**-0.5
        key = apply_rope(key, positions=positions)
        if decode:
            key, values = _update_kv_cache(
                self,
                key,
                values,
                cache_size=attention_mask.shape[-1],
                cache_dtype=self.cache_dtype,
            )

        query = einops.rearrange(
            query,
            "B T (K G) H -> B T K G H",
            K=self.num_kv_heads,
        )
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", query, key).astype(
            jnp.float32
        )
        if self.attn_logits_softcap:
            logits = (
                jnp.tanh(logits / self.attn_logits_softcap)
                * self.attn_logits_softcap
            )
        expected = (query.shape[0], 1, query.shape[1], key.shape[1])
        if attention_mask.shape != expected:
            raise ValueError(
                f"attention mask {attention_mask.shape} does not match {expected}"
            )
        logits = jnp.where(
            attention_mask[:, :, None], logits, -2.3819763e38
        )
        probabilities = jax.nn.softmax(logits, axis=-1).astype(key.dtype)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probabilities, values)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
        output = self.attn_vec_einsum("BTNH,NHD->BTD", encoded)

        if self._enabled("o"):
            left = self.param(
                "o_lora_a",
                _lora_a_init((self.num_heads * self.head_dim) ** -0.5),
                (self.num_heads, self.head_dim, self.lora_rank),
            )
            right = self.param(
                "o_lora_b",
                nn.initializers.zeros_init(),
                (self.lora_rank, self.features),
            )
            output += self._scale * jnp.einsum(
                "BTNH,NHR,RD->BTD", encoded, left, right
            )
        return output


class LoraBlock(nn.Module):
    num_heads: int = GEMMA_HEADS
    num_kv_heads: int = GEMMA_KV_HEADS
    embed_dim: int = GEMMA_WIDTH
    head_dim: int = GEMMA_HEAD_DIM
    hidden_dim: int = GEMMA_MLP_DIM
    attn_logits_softcap: float = 50.0
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = "bfloat16"
    lora_rank: int = PAPER_LORA_RANK
    lora_alpha: float = PAPER_LORA_ALPHA
    lora_targets: tuple[str, ...] = PAPER_LORA_TARGETS

    def setup(self):
        self.pre_attention_norm = RMSNorm()
        self.attn = LoraAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            features=self.embed_dim,
            head_dim=self.head_dim,
            attn_logits_softcap=self.attn_logits_softcap,
            cache_dtype=self.cache_dtype,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_targets=self.lora_targets,
        )
        self.pre_ffw_norm = RMSNorm()
        self.mlp = FeedForward(
            features=self.embed_dim, hidden_dim=self.hidden_dim
        )
        self.drop = (
            nn.Dropout(self.dropout, self.dropout_bdims)
            if self.dropout
            else lambda x, _: x
        )
        self.post_attention_norm = RMSNorm()
        self.post_ffw_norm = RMSNorm()

    def __call__(
        self,
        value,
        scan_state,
        positions,
        attention_mask,
        decode,
        deterministic=True,
    ):
        value = nn.with_logical_constraint(
            value, ("act_batch", "act_len", "act_emb")
        )
        normalized = self.pre_attention_norm(value)
        attention = self.attn(
            normalized, positions, attention_mask, decode, deterministic
        )
        attention = self.post_attention_norm(attention)
        attention = self.drop(attention, deterministic)
        residual = value + attention
        output = self.mlp(self.pre_ffw_norm(residual))
        output = self.drop(output, deterministic)
        output = self.post_ffw_norm(output)
        return residual + output, scan_state


class GemmaCore(nn.Module):
    """Gemma-2 2B core preserving Big Vision's parameter hierarchy."""

    width: int = GEMMA_WIDTH
    depth: int = GEMMA_DEPTH
    mlp_dim: int = GEMMA_MLP_DIM
    num_heads: int = GEMMA_HEADS
    num_kv_heads: int = GEMMA_KV_HEADS
    head_dim: int = GEMMA_HEAD_DIM
    vocab_size: int = GEMMA_VOCAB_SIZE
    final_logits_softcap: float = 30.0
    attn_logits_softcap: float = 50.0
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = "bfloat16"
    embed_dtype: str = "float32"
    remat_policy: str = "nothing_saveable"
    lora_rank: int = PAPER_LORA_RANK
    lora_alpha: float = PAPER_LORA_ALPHA
    lora_targets: tuple[str, ...] = PAPER_LORA_TARGETS

    @nn.compact
    def __call__(
        self,
        tokens,
        *,
        embedded_prefix=None,
        embed_only=False,
        pre_logits=None,
        positions=None,
        mask=None,
        decode=False,
        deterministic=True,
        return_pre_logits_only=False,
    ):
        if (
            self.width,
            self.depth,
            self.mlp_dim,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        ) != (
            GEMMA_WIDTH,
            GEMMA_DEPTH,
            GEMMA_MLP_DIM,
            GEMMA_HEADS,
            GEMMA_KV_HEADS,
            GEMMA_HEAD_DIM,
        ):
            raise ValueError("the public backend supports Gemma-2 2B only")
        embedder = Embedder(
            vocab_size=self.vocab_size, embed_dim=self.width, name="embedder"
        )
        if pre_logits is not None:
            return embedder.decode(pre_logits)

        pieces = []
        if embedded_prefix is not None:
            pieces.append(embedded_prefix)
        if tokens is not None:
            pieces.append(embedder.encode(tokens))
        if not pieces:
            raise ValueError("tokens or embedded_prefix is required")
        value = jnp.concatenate(pieces, axis=-2).astype(self.embed_dtype)
        if embed_only:
            return value
        batch, length, _ = value.shape
        if positions is None:
            positions = jnp.broadcast_to(
                jnp.arange(length, dtype=jnp.int32)[None], (batch, length)
            )
        if positions.shape != (batch, length):
            raise ValueError("positions must be [B,T]")
        if mask is None:
            mask = nn.attention.make_causal_mask(jnp.ones((batch, length)))
        if mask.ndim == 3:
            mask = mask[:, None]
        cache_size = max(length, mask.shape[-1])
        if mask.shape != (batch, 1, length, cache_size):
            raise ValueError(
                f"mask must be {(batch, 1, length, cache_size)}, got {mask.shape}"
            )

        block_cls = LoraBlock
        if self.remat_policy != "none":
            block_cls = nn.remat(
                LoraBlock,
                prevent_cse=False,
                static_argnums=(5, 6),
                policy=getattr(jax.checkpoint_policies, self.remat_policy),
            )
        kwargs = dict(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            embed_dim=self.width,
            head_dim=self.head_dim,
            hidden_dim=self.mlp_dim,
            attn_logits_softcap=self.attn_logits_softcap,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            cache_dtype=self.cache_dtype,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_targets=_targets(self.lora_targets),
        )
        blocks = nn.scan(
            block_cls,
            variable_axes={"params": 0, "cache": 1},
            split_rngs={"params": True, "dropout": True},
            in_axes=nn.broadcast,
            length=self.depth,
        )(parent=self.scope.push("layers"), **kwargs)
        value, _ = blocks(value, (), positions, mask, decode, deterministic)
        pre_logits = RMSNorm(name="final_norm")(value)
        if return_pre_logits_only:
            return pre_logits
        logits = embedder.decode(pre_logits)
        if self.final_logits_softcap:
            logits = (
                jnp.tanh(logits / self.final_logits_softcap)
                * self.final_logits_softcap
            )
        return logits


@jax.vmap
def _left_to_right_align(value, input_mask, attention_mask):
    length = jnp.sum(input_mask)
    return (
        jnp.roll(value, -length, axis=0),
        jnp.roll(input_mask, -length, axis=0),
        jnp.roll(attention_mask, -length, axis=(0, 1)),
    )


class Gemma2LoRA(nn.Module):
    """PaliGemma-facing wrapper whose params live directly under ``llm``."""

    vocab_size: int = GEMMA_VOCAB_SIZE
    remat_policy: str = "nothing_saveable"
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = "bfloat16"
    lora_rank: int = PAPER_LORA_RANK
    lora_alpha: float = PAPER_LORA_ALPHA
    lora_targets: tuple[str, ...] = PAPER_LORA_TARGETS

    def setup(self):
        # parent=self.scope/name="" is load-bearing: it avoids an extra
        # ``model`` scope absent from Big Vision and the released checkpoints.
        self.model = GemmaCore(
            vocab_size=self.vocab_size,
            remat_policy=self.remat_policy,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            cache_dtype=self.cache_dtype,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_targets=self.lora_targets,
            parent=self.scope,
            name="",
        )

    def embed_tokens(self, token_ids, train: bool = False):
        return self.model(
            token_ids, embed_only=True, deterministic=not train
        )

    def compute_logits(self, pre_logits, train: bool = False):
        return self.model(None, pre_logits=pre_logits, deterministic=not train)

    def compute_pre_logits(self, embeddings, mask=None, train: bool = False):
        batch = embeddings.shape[0]
        return self.model(
            jnp.zeros((batch, 0), jnp.int32),
            embedded_prefix=embeddings,
            mask=mask,
            deterministic=not train,
            return_pre_logits_only=True,
        )

    def prefill_cache(
        self, embeddings, input_mask, attention_mask, *, cache_size: int
    ):
        embeddings, input_mask, attention_mask = _left_to_right_align(
            embeddings, input_mask, attention_mask
        )
        sequence_length = jnp.sum(input_mask, axis=-1)
        positions = jnp.cumsum(input_mask, axis=-1) - 1
        pad = cache_size - embeddings.shape[1]
        if pad < 0:
            raise ValueError("cache_size is smaller than the prompt")
        mask = jnp.pad(attention_mask, ((0, 0), (0, 0), (0, pad)))
        self.put_variable(
            "cache",
            "cache_end",
            jnp.full((embeddings.shape[0],), embeddings.shape[1], jnp.int32),
        )
        self.put_variable(
            "cache",
            "cache_valid",
            jnp.pad(input_mask.astype(bool), ((0, 0), (0, pad))),
        )
        self.put_variable(
            "cache", "valid_count", sequence_length.astype(jnp.int32)
        )
        pre_logits = self.model(
            None,
            embedded_prefix=embeddings,
            positions=positions,
            mask=mask,
            decode=True,
            # Only the last slot is projected: never build [B,T,V] logits for
            # the whole prompt (jit would drop them, but eager calls would not).
            return_pre_logits_only=True,
        )
        return self.compute_logits(pre_logits[:, -1:])

    def extend_cache(self, embeddings):
        if embeddings.shape[1] != 1:
            raise ValueError("extend_cache accepts exactly one token")
        key_cache = self.variables["cache"]["layers"]["attn"]["k_cache"]
        cache_size = key_cache.shape[2]
        cache_end = self.get_variable("cache", "cache_end")
        columns = jnp.arange(cache_size)[None]
        cache_valid = self.get_variable("cache", "cache_valid")
        valid_count = self.get_variable("cache", "valid_count")
        own = columns == cache_end[:, None]
        mask = cache_valid | own
        logits = self.model(
            None,
            embedded_prefix=embeddings,
            positions=valid_count[:, None],
            mask=mask[:, None, :],
            decode=True,
        )
        self.put_variable("cache", "cache_end", cache_end + 1)
        self.put_variable("cache", "cache_valid", cache_valid | own)
        self.put_variable("cache", "valid_count", valid_count + 1)
        return logits


def add_lora_params(
    params: dict[str, Any],
    *,
    rank: int = PAPER_LORA_RANK,
    targets: tuple[str, ...] = PAPER_LORA_TARGETS,
    seed: int = 0,
) -> dict[str, Any]:
    """Add zero-output LoRA leaves to a loaded scanned Gemma parameter tree."""

    if rank < 1:
        return params
    targets = _targets(targets)
    rng = np.random.default_rng(seed)
    attention = params["layers"]["attn"]

    def normal(shape, std):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    if "q" in targets:
        weight = attention["q_einsum"]["w"]
        prefix = weight.shape[:-3]
        heads, features, head_dim = weight.shape[-3:]
        attention.setdefault(
            "q_lora_a", normal(prefix + (features, rank), features**-0.5)
        )
        attention.setdefault(
            "q_lora_b", np.zeros(prefix + (heads, rank, head_dim), np.float32)
        )
    if "kv" in targets:
        weight = attention["kv_einsum"]["w"]
        prefix = weight.shape[:-4]
        _, heads, features, head_dim = weight.shape[-4:]
        attention.setdefault(
            "kv_lora_a", normal(prefix + (features, rank), features**-0.5)
        )
        attention.setdefault(
            "kv_lora_b",
            np.zeros(prefix + (2, heads, rank, head_dim), np.float32),
        )
    if "o" in targets:
        weight = attention["attn_vec_einsum"]["w"]
        prefix = weight.shape[:-3]
        heads, head_dim, features = weight.shape[-3:]
        attention.setdefault(
            "o_lora_a",
            normal(prefix + (heads, head_dim, rank), (heads * head_dim) ** -0.5),
        )
        attention.setdefault(
            "o_lora_b", np.zeros(prefix + (rank, features), np.float32)
        )
    return params


__all__ = [
    "GEMMA_DEPTH",
    "GEMMA_VOCAB_SIZE",
    "GEMMA_WIDTH",
    "Gemma2LoRA",
    "GemmaCore",
    "LoraAttention",
    "LoraBlock",
    "add_lora_params",
    "apply_rope",
]
