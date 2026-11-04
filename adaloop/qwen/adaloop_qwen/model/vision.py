# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 The AdaLoop Authors.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# Modified by the AdaLoop Authors from Transformers 4.57.1
# (https://github.com/huggingface/transformers)
# src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py: the forwards of
# Qwen2_5_VisionTransformerPretrainedModel and Qwen2_5_VLVisionBlock gained
# question-conditioned AdaLN and recurrent execution of a contiguous block
# span with per-depth coda/merger readouts.

"""Qwen2.5-VL adaptive vision execution.

The implementation in this module deliberately mirrors the Qwen2.5-VL
vision forward shipped by Transformers 4.57.1.  The patch embedding, rotary
positions, window packing, blocks, and merger therefore retain their upstream
module names.  Only execution of a contiguous block span changes: the prefix
is evaluated once, the span is recurrent with latent carry, and the suffix is
used as a readout branch at requested depths.

Transformers is an optional import.  The framework-independent mixins remain
usable by the CPU tests (and by small parity harnesses) without importing a
Qwen checkpoint; the concrete Qwen classes explain how to install the GPU
extra when instantiated without Transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import AdaLNZero


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError(
        "AdaLoop configuration must be a mapping or expose to_dict(); got "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class AdaptiveVisionSpec:
    """Small normalized runtime specification for the public vision paths."""

    variant: str
    loop_start: int | None
    loop_end: int | None
    loop_steps: int
    adaln_layers: tuple[int, ...]
    condition_dim: int
    condition_heads: int
    condition_mlp_ratio: float
    adaln_hidden_dim: int
    condition_detach: bool
    conditioner_mode: str

    @property
    def is_recurrent(self) -> bool:
        return self.variant in {"a", "b", "c"}

    @property
    def uses_adaln(self) -> bool:
        return bool(self.adaln_layers)

    @classmethod
    def from_source(
        cls,
        source: Any,
        *,
        vision_depth: int,
    ) -> "AdaptiveVisionSpec":
        """Normalize a public AdaLoop config.

        Options outside the released topology are rejected here instead of
        silently changing it.
        """

        raw = _as_mapping(source)
        variant = str(raw.get("variant", "")).lower()
        if variant not in {"sft", "adaln", "a", "b", "c"}:
            raise ValueError(
                "Adaptive vision variant must be sft, adaln, a, b, or c; got "
                f"{variant!r}"
            )

        conditioner = raw.get("conditioner", {})
        if conditioner is None:
            conditioner = {}
        if not isinstance(conditioner, Mapping):
            raise TypeError("conditioner must be a mapping")
        conditioner_mode = str(
            conditioner.get(
                "mode",
                raw.get(
                    "adaln_condition_mode",
                    "none"
                    if variant == "sft"
                    else (
                        "question"
                        if raw.get("adaln_enable", True)
                        else "none"
                    ),
                ),
            )
        ).lower()
        if conditioner_mode not in {"none", "question", "constant_question"}:
            raise ValueError(
                "conditioner mode must be none, question, or constant_question; "
                f"got {conditioner_mode!r}"
            )

        if variant == "sft":
            return cls(
                variant="sft",
                loop_start=None,
                loop_end=None,
                loop_steps=1,
                adaln_layers=(),
                condition_dim=int(conditioner.get("hidden_size") or 512),
                condition_heads=int(conditioner.get("attention_heads") or 8),
                condition_mlp_ratio=float(raw.get("condition_mlp_ratio", 2.0)),
                adaln_hidden_dim=int(raw.get("adaln_hidden_dim", 128)),
                condition_detach=bool(raw.get("condition_detach", True)),
                conditioner_mode="none",
            )

        span = raw.get("adaptive_span")
        if span is not None:
            if not isinstance(span, Sequence) or isinstance(span, (str, bytes)):
                raise TypeError("adaptive_span must be [start, end]")
            if len(span) != 2:
                raise ValueError("adaptive_span must contain exactly two entries")
            start, end = int(span[0]), int(span[1])
        elif "loop_start" in raw or "loop_end" in raw:
            start, end = int(raw["loop_start"]), int(raw["loop_end"])
        else:
            explicit_layers = tuple(int(x) for x in raw.get("adaln_layers", ()))
            if not explicit_layers:
                raise ValueError(
                    "Adaptive execution requires adaptive_span, loop_start/loop_end, "
                    "or non-empty adaln_layers"
                )
            start, end = min(explicit_layers), max(explicit_layers) + 1
        if not 0 <= start < end <= int(vision_depth):
            raise ValueError(
                f"Adaptive span [{start}, {end}) must be non-empty and inside "
                f"[0, {vision_depth})"
            )

        round_mode = str(
            raw.get(
                "vision_round_mode",
                raw.get("loop_vision_round_mode", "latent_carry"),
            )
        )
        if round_mode != "latent_carry":
            raise ValueError("The public runtime supports latent_carry only")

        if conditioner_mode == "none":
            adaln_layers: tuple[int, ...] = ()
        elif "adaln_layers" in raw:
            adaln_layers = tuple(int(x) for x in raw["adaln_layers"])
        else:
            adaln_layers = tuple(range(start, end))
        if any(layer < start or layer >= end for layer in adaln_layers):
            raise ValueError(
                "AdaLN layers must lie inside the adaptive span; got "
                f"{adaln_layers!r} for [{start}, {end})"
            )
        if variant == "adaln" and conditioner_mode == "none":
            raise ValueError("The adaln variant requires a question conditioner")

        steps = int(raw.get("train_depth", raw.get("loop_steps", 1)))
        if steps < 1:
            raise ValueError(f"loop depth must be positive, got {steps}")
        if variant == "adaln" and steps != 1:
            raise ValueError("Single-pass AdaLN requires depth 1")

        return cls(
            variant=variant,
            loop_start=start,
            loop_end=end,
            loop_steps=steps,
            adaln_layers=adaln_layers,
            condition_dim=int(
                conditioner.get("hidden_size") or raw.get("condition_dim", 512)
            ),
            condition_heads=int(
                conditioner.get("attention_heads") or raw.get("condition_heads", 8)
            ),
            condition_mlp_ratio=float(
                raw.get("condition_mlp_ratio", 2.0)
            ),
            adaln_hidden_dim=int(raw.get("adaln_hidden_dim", 128)),
            condition_detach=bool(raw.get("condition_detach", True)),
            conditioner_mode=conditioner_mode,
        )


@dataclass
class VisionRuntimeState:
    """Transient controls saved around nested or failed model calls."""

    active_condition: torch.Tensor | None
    active_loop_steps: int | None
    active_loop_depths: torch.Tensor | None
    collect_loop_step_features: bool
    last_loop_step_features: list[torch.Tensor] | None


class AdaLNResidualBlockMixin:
    """Qwen vision-block forward with identity-initialized AdaLN modulation."""

    @staticmethod
    def _modulate(
        value: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return value * (1 + scale) + shift

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        vision_condition: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if vision_condition is None:
            raise ValueError("AdaLN vision block requires a per-patch vision_condition")
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln(vision_condition)
        attn_input = self._modulate(
            self.norm1(hidden_states), shift_attn, scale_attn
        )
        attn_output = self.attn(
            attn_input,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + (1 + gate_attn) * attn_output
        mlp_input = self._modulate(
            self.norm2(hidden_states), shift_mlp, scale_mlp
        )
        return hidden_states + (1 + gate_mlp) * self.mlp(mlp_input)


class AdaptiveVisionRuntimeMixin:
    """Base-independent implementation of Qwen's packed vision forward.

    A concrete host supplies the official Qwen attributes ``patch_embed``,
    ``rot_pos_emb``, ``get_window_index``, ``blocks``, ``merger``,
    ``spatial_merge_unit``, and ``fullatt_block_indexes``.
    """

    _returns_vision_model_output = False

    def _initialize_adaptive_runtime(self, spec: AdaptiveVisionSpec) -> None:
        self.adaloop_spec = spec
        self.variant = spec.variant
        self.adaln_layers = frozenset(spec.adaln_layers)
        self._active_condition: torch.Tensor | None = None
        self._active_loop_steps: int | None = None
        self._active_loop_depths: torch.Tensor | None = None
        self._collect_loop_step_features = False
        self._last_loop_step_features: list[torch.Tensor] | None = None

    @property
    def uses_adaln(self) -> bool:
        return bool(self.adaln_layers)

    def set_active_condition(self, condition: torch.Tensor | None) -> None:
        self._active_condition = condition

    def set_active_loop_steps(self, steps: int | None) -> None:
        if steps is not None and int(steps) < 1:
            raise ValueError(f"Vision loop depth must be >= 1, got {steps}")
        self._active_loop_steps = None if steps is None else int(steps)

    def set_active_loop_depths(self, depths: torch.Tensor | None) -> None:
        self._active_loop_depths = depths

    def set_collect_loop_step_features(self, enabled: bool) -> None:
        self._collect_loop_step_features = bool(enabled)
        self._last_loop_step_features = None

    def take_loop_step_features(self) -> list[torch.Tensor] | None:
        features = self._last_loop_step_features
        self._last_loop_step_features = None
        return features

    def snapshot_runtime_state(self) -> VisionRuntimeState:
        return VisionRuntimeState(
            active_condition=self._active_condition,
            active_loop_steps=self._active_loop_steps,
            active_loop_depths=self._active_loop_depths,
            collect_loop_step_features=self._collect_loop_step_features,
            last_loop_step_features=self._last_loop_step_features,
        )

    def restore_runtime_state(self, state: VisionRuntimeState) -> None:
        self._active_condition = state.active_condition
        self._active_loop_steps = state.active_loop_steps
        self._active_loop_depths = state.active_loop_depths
        self._collect_loop_step_features = state.collect_loop_step_features
        self._last_loop_step_features = state.last_loop_step_features

    def _loop_steps_for_forward(self) -> int:
        steps = (
            self.adaloop_spec.loop_steps
            if self._active_loop_steps is None
            else self._active_loop_steps
        )
        if steps < 1:
            raise ValueError(f"Vision loop depth must be >= 1, got {steps}")
        return int(steps)

    def _patch_condition(
        self,
        grid_thw: torch.Tensor,
        window_index: torch.Tensor,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        condition: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.uses_adaln:
            return None
        if condition is None:
            raise ValueError(
                "AdaLN vision execution requires question conditioning inputs"
            )
        if condition.ndim != 2:
            raise ValueError(
                "Question condition must be [batch, condition_dim], got "
                f"{tuple(condition.shape)}"
            )
        if grid_thw.shape[0] != condition.shape[0]:
            raise ValueError(
                "The public Qwen path requires one image per example: "
                f"got {grid_thw.shape[0]} grids for condition batch "
                f"{condition.shape[0]}"
            )
        patch_counts = grid_thw.prod(dim=-1).to(
            device=condition.device, dtype=torch.long
        )
        patch_condition = torch.repeat_interleave(condition, patch_counts, dim=0)
        if patch_condition.shape[0] != seq_len:
            raise ValueError(
                f"Condition expansion produced {patch_condition.shape[0]} patches, "
                f"expected {seq_len}"
            )
        if seq_len % int(self.spatial_merge_unit):
            raise ValueError(
                f"Packed patch count {seq_len} is not divisible by spatial merge "
                f"unit {self.spatial_merge_unit}"
            )
        patch_condition = patch_condition.reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        patch_condition = patch_condition[window_index].reshape(seq_len, -1)
        return patch_condition.to(device=device, dtype=dtype)

    def _patch_loop_depths(
        self,
        grid_thw: torch.Tensor,
        window_index: torch.Tensor,
        seq_len: int,
        steps: int,
        depths: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand per-example terminal depths into Qwen's packed patch order."""

        if depths.ndim != 1:
            raise ValueError(
                "Per-example loop depths must be rank 1 [batch], got "
                f"{tuple(depths.shape)}"
            )
        if (
            depths.dtype == torch.bool
            or torch.is_floating_point(depths)
            or torch.is_complex(depths)
        ):
            raise TypeError("Per-example loop depths must use an integer dtype")
        if grid_thw.shape[0] != depths.shape[0]:
            raise ValueError(
                "Per-example depths require one image per example: "
                f"got {grid_thw.shape[0]} grids for {depths.shape[0]} depths"
            )
        depths = depths.to(device=device, dtype=torch.long)
        if depths.numel() == 0:
            raise ValueError("Per-example loop depths must not be empty")
        if bool((depths < 1).any()) or bool((depths > steps).any()):
            raise ValueError(
                "Per-example loop depths must lie inside the physical execution "
                f"range [1, {steps}]; got {depths.tolist()}"
            )
        patch_counts = grid_thw.prod(dim=-1).to(device=device, dtype=torch.long)
        patch_depths = torch.repeat_interleave(depths, patch_counts, dim=0)
        if patch_depths.shape[0] != seq_len:
            raise ValueError(
                f"Depth expansion produced {patch_depths.shape[0]} patches, "
                f"expected {seq_len}"
            )
        patch_depths = patch_depths.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit
        )
        return patch_depths[window_index].reshape(seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        question_condition: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        window_index = window_index.to(device=hidden_states.device)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = hidden_states.size()
        if seq_len % int(self.spatial_merge_unit):
            raise ValueError(
                f"Packed patch count {seq_len} is not divisible by spatial merge "
                f"unit {self.spatial_merge_unit}"
            )
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        hidden_states = hidden_states[window_index].reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        rotary_pos_emb = rotary_pos_emb[window_index].reshape(seq_len, -1)
        rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        condition = (
            question_condition
            if question_condition is not None
            else self._active_condition
        )

        def run_block(
            layer: int,
            states: torch.Tensor,
            patch_condition: torch.Tensor | None,
        ) -> torch.Tensor:
            cu_now = (
                cu_seqlens
                if layer in self.fullatt_block_indexes
                else cu_window_seqlens
            )
            if layer in self.adaln_layers:
                # Positional tensor arguments retain conditioner gradients with
                # re-entrant gradient-checkpoint wrappers.
                return self.blocks[layer](
                    states,
                    cu_now,
                    None,
                    position_embeddings,
                    patch_condition,
                    **kwargs,
                )
            return self.blocks[layer](
                states,
                cu_seqlens=cu_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if self.variant == "adaln":
            patch_condition = self._patch_condition(
                grid_thw,
                window_index,
                seq_len,
                hidden_states.dtype,
                hidden_states.device,
                condition,
            )
            for layer in range(len(self.blocks)):
                hidden_states = run_block(layer, hidden_states, patch_condition)
            readout_states = [hidden_states]
            core_states = [hidden_states]
        elif self.variant in {"a", "b", "c"}:
            start = self.adaloop_spec.loop_start
            end = self.adaloop_spec.loop_end
            if start is None or end is None:
                raise RuntimeError(
                    f"Variant {self.variant.upper()} is missing its adaptive span"
                )
            for layer in range(start):
                hidden_states = run_block(layer, hidden_states, None)

            carried_states = hidden_states
            core_states: list[torch.Tensor] = []
            steps = self._loop_steps_for_forward()
            patch_loop_depths = None
            if self._active_loop_depths is not None:
                if self._collect_loop_step_features:
                    raise RuntimeError(
                        "Per-example loop depths produce one mixed terminal "
                        "readout and cannot collect uniform per-depth features"
                    )
                patch_loop_depths = self._patch_loop_depths(
                    grid_thw,
                    window_index,
                    seq_len,
                    steps,
                    self._active_loop_depths,
                    device=hidden_states.device,
                )
            for step in range(steps):
                previous_states = carried_states
                round_states = carried_states
                patch_condition = self._patch_condition(
                    grid_thw,
                    window_index,
                    seq_len,
                    round_states.dtype,
                    round_states.device,
                    condition,
                )
                for layer in range(start, end):
                    round_states = run_block(
                        layer, round_states, patch_condition
                    )
                if patch_loop_depths is not None:
                    active_patches = (patch_loop_depths > step).unsqueeze(-1)
                    round_states = torch.where(
                        active_patches, round_states, previous_states
                    )
                carried_states = round_states
                core_states.append(round_states)

            def run_coda(states: torch.Tensor) -> torch.Tensor:
                for layer in range(end, len(self.blocks)):
                    states = run_block(layer, states, None)
                return states

            if self._collect_loop_step_features:
                readout_states = [run_coda(states) for states in core_states]
            else:
                readout_states = [run_coda(core_states[-1])]
            hidden_states = readout_states[-1]
        else:
            raise RuntimeError(
                f"Adaptive vision received unsupported variant {self.variant!r}"
            )

        reverse_indices = torch.argsort(window_index)
        merged_steps = [
            self.merger(states)[reverse_indices] for states in readout_states
        ]
        merged_hidden_states = merged_steps[-1]
        if self.adaloop_spec.is_recurrent and self._collect_loop_step_features:
            self._last_loop_step_features = merged_steps

        if self._returns_vision_model_output:
            return _base_model_output_with_pooling(
                last_hidden_state=hidden_states,
                pooler_output=merged_hidden_states,
            )
        return merged_hidden_states


_TRANSFORMERS_IMPORT_ERROR: BaseException | None = None
_QWEN_VISION_BLOCK: type[nn.Module] | None = None
_QWEN_VISION_MODEL: type[nn.Module] | None = None
_base_model_output_with_pooling: Any = None
try:  # Imported only when the optional GPU dependency is available.
    import transformers as _transformers
    from transformers.modeling_outputs import (
        BaseModelOutputWithPooling as _BaseModelOutputWithPooling,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLVisionBlock as _Qwen2_5_VLVisionBlock,
        Qwen2_5_VisionTransformerPretrainedModel as _Qwen2_5_VisionTransformer,
    )

    _QWEN_VISION_BLOCK = _Qwen2_5_VLVisionBlock
    _QWEN_VISION_MODEL = _Qwen2_5_VisionTransformer
    _base_model_output_with_pooling = _BaseModelOutputWithPooling
    _version = tuple(
        int(part) for part in _transformers.__version__.split("+")[0].split(".")[:2]
    )
    _RETURNS_VISION_MODEL_OUTPUT = _version >= (4, 58)
except Exception as exc:
    _TRANSFORMERS_IMPORT_ERROR = exc
    _RETURNS_VISION_MODEL_OUTPUT = False


def require_qwen_transformers() -> None:
    """Raise an actionable error when the optional model runtime is absent."""

    if _QWEN_VISION_MODEL is None:
        raise ModuleNotFoundError(
            "Qwen2.5-VL execution requires the qwen[gpu] extra "
            "(transformers==4.57.1 and torch==2.6.0)."
        ) from _TRANSFORMERS_IMPORT_ERROR


if _QWEN_VISION_BLOCK is not None:

    class AdaLNQwen2_5_VLVisionBlock(  # type: ignore[misc,valid-type]
        AdaLNResidualBlockMixin,
        _QWEN_VISION_BLOCK,
    ):
        """Official Qwen2.5 block with AdaLN on both residual branches."""

        def __init__(
            self,
            config: Any,
            *,
            condition_dim: int,
            hidden_dim: int,
        ) -> None:
            _QWEN_VISION_BLOCK.__init__(self, config)
            self.adaln = AdaLNZero(
                condition_dim, config.hidden_size, hidden_dim
            )

else:

    class AdaLNQwen2_5_VLVisionBlock(nn.Module):
        """Unavailable placeholder when Transformers is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__()
            require_qwen_transformers()


if _QWEN_VISION_MODEL is not None:

    class AdaptiveQwen2_5_VisionTransformer(  # type: ignore[misc,valid-type]
        AdaptiveVisionRuntimeMixin,
        _QWEN_VISION_MODEL,
    ):
        """Single-pass AdaLN or latent-carry Qwen2.5 vision transformer."""

        _returns_vision_model_output = _RETURNS_VISION_MODEL_OUTPUT

        def __init__(self, config: Any, spec: AdaptiveVisionSpec) -> None:
            _QWEN_VISION_MODEL.__init__(self, config)
            self.blocks = nn.ModuleList(
                [
                    AdaLNQwen2_5_VLVisionBlock(
                        config,
                        condition_dim=spec.condition_dim,
                        hidden_dim=spec.adaln_hidden_dim,
                    )
                    if layer in spec.adaln_layers
                    else _QWEN_VISION_BLOCK(config)
                    for layer in range(config.depth)
                ]
            )
            self._initialize_adaptive_runtime(spec)

else:

    class AdaptiveQwen2_5_VisionTransformer(
        AdaptiveVisionRuntimeMixin,
        nn.Module,
    ):
        """Unavailable placeholder when Transformers is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            nn.Module.__init__(self)
            require_qwen_transformers()


__all__ = [
    "AdaLNQwen2_5_VLVisionBlock",
    "AdaLNResidualBlockMixin",
    "AdaptiveQwen2_5_VisionTransformer",
    "AdaptiveVisionRuntimeMixin",
    "AdaptiveVisionSpec",
    "VisionRuntimeState",
    "require_qwen_transformers",
]
