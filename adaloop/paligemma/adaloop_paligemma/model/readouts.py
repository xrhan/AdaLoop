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
# big_vision/models/proj/paligemma/paligemma.py: make_attn_mask gained input
# validation; the Variant A/B readout packing is new.

"""Native PaliGemma readouts for AdaLoop Variants A and B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp

from ..constants import PAPER_MAX_SEQUENCE_LENGTH


@dataclass(frozen=True)
class PackedReadout:
    embeddings: Any
    input_mask: Any
    mask_ar: Any
    text_offset: int


def make_attn_mask(input_mask, mask_ar):
    """Big Vision/PaliGemma prefix-LM attention mask.

    A one in ``mask_ar`` begins a new causal block; zero keeps a token in the
    preceding bidirectional block.  Invalid padding is masked on both axes.
    """

    if input_mask.shape != mask_ar.shape or input_mask.ndim != 2:
        raise ValueError("input_mask and mask_ar must have equal [B,T] shapes")
    cumulative = jnp.cumsum(mask_ar, axis=1)
    causal = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = input_mask[:, None, :] & input_mask[:, :, None]
    return causal & valid


def _validate_readouts(readouts: Sequence[Any], depth: int) -> tuple[Any, ...]:
    if depth < 1 or depth > len(readouts):
        raise ValueError(
            f"depth must lie in [1,{len(readouts)}], got {depth}"
        )
    selected = tuple(readouts[:depth])
    first = selected[0]
    if first.ndim != 3:
        raise ValueError("image readouts must be [B,N,D]")
    for value in selected[1:]:
        if value.ndim != 3 or value.shape != first.shape:
            raise ValueError("all image readouts must have the same [B,N,D] shape")
    return selected


def chained_image_mask_ar(
    batch_size: int, image_tokens: int, depth: int
):
    """Block-causal Variant-B mask row.

    The first token of image readouts 2..T starts a new block.  The final image
    readout and the following question remain in the same bidirectional block,
    matching the paper's chained PaliGemma topology.
    """

    if batch_size < 1 or image_tokens < 1 or depth < 1:
        raise ValueError("batch_size, image_tokens, and depth must be positive")
    index = jnp.arange(image_tokens * depth)
    starts = (index % image_tokens == 0) & (index >= image_tokens)
    return jnp.broadcast_to(
        starts.astype(jnp.int32)[None], (batch_size, image_tokens * depth)
    )


def pack_current_image(
    image_readout, text_embeddings, text_input_mask, text_mask_ar
) -> PackedReadout:
    """Pack the ``current_image`` readout ``[I_t | P] A`` (SFT, AdaLN, A)."""

    if image_readout.ndim != 3 or text_embeddings.ndim != 3:
        raise ValueError("image/text embeddings must be rank three")
    if image_readout.shape[0] != text_embeddings.shape[0]:
        raise ValueError("image and text batch dimensions differ")
    if image_readout.shape[-1] != text_embeddings.shape[-1]:
        raise ValueError("image and language embedding widths differ")
    if text_input_mask.shape != text_mask_ar.shape:
        raise ValueError("text_input_mask and text_mask_ar must align")
    batch, image_tokens, _ = image_readout.shape
    image_valid = jnp.ones((batch, image_tokens), dtype=bool)
    image_ar = jnp.zeros((batch, image_tokens), dtype=jnp.int32)
    return PackedReadout(
        embeddings=jnp.concatenate([image_readout, text_embeddings], axis=1),
        input_mask=jnp.concatenate(
            [image_valid, text_input_mask.astype(bool)], axis=1
        ),
        mask_ar=jnp.concatenate(
            [image_ar, text_mask_ar.astype(jnp.int32)], axis=1
        ),
        text_offset=image_tokens,
    )


def pack_accumulated_images(
    image_readouts: Sequence[Any],
    text_embeddings,
    text_input_mask,
    text_mask_ar,
    *,
    depth: int,
    max_length: int = PAPER_MAX_SEQUENCE_LENGTH,
) -> PackedReadout:
    """Pack the Variant-B ``accumulated_images`` readout ``[I1 | ... | It | P] A``.

    Only the causal ``chained`` topology is public.  A fused topology
    is deliberately not exposed because it lets earlier visual snapshots read
    later snapshots.
    """

    selected = _validate_readouts(image_readouts, depth)
    prefix = jnp.concatenate(selected, axis=1)
    if prefix.shape[0] != text_embeddings.shape[0]:
        raise ValueError("image and text batch dimensions differ")
    if prefix.shape[-1] != text_embeddings.shape[-1]:
        raise ValueError("image and language embedding widths differ")
    total = prefix.shape[1] + text_embeddings.shape[1]
    if total > max_length:
        raise ValueError(
            f"Variant-B context {total} exceeds max_length={max_length}"
        )
    batch, image_tokens, _ = selected[0].shape
    image_ar = chained_image_mask_ar(batch, image_tokens, depth)
    image_valid = jnp.ones(image_ar.shape, dtype=bool)
    return PackedReadout(
        embeddings=jnp.concatenate([prefix, text_embeddings], axis=1),
        input_mask=jnp.concatenate(
            [image_valid, text_input_mask.astype(bool)], axis=1
        ),
        mask_ar=jnp.concatenate(
            [image_ar, text_mask_ar.astype(jnp.int32)], axis=1
        ),
        text_offset=prefix.shape[1],
    )


def select_text_slots(states, readout: PackedReadout, text_width: int):
    """Return the original language input slots of per-position outputs.

    The model passes Gemma pre-logits and projects only these text slots.
    """

    start = readout.text_offset
    return states[:, start : start + text_width]


__all__ = [
    "PackedReadout",
    "chained_image_mask_ar",
    "make_attn_mask",
    "pack_accumulated_images",
    "pack_current_image",
    "select_text_slots",
]
