"""Qwen2.5-VL conditional-generation wrapper for AdaLoop A/B/C.

The wrapper keeps the official hierarchy (notably ``model.visual`` and
``model.language_model``) so a base Qwen2.5-VL state dict still loads without
key migration.  Adaptive parameters use the checkpoint names
``question_conditioner`` and ``model.visual.blocks.<n>.adaln``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .conditioning import AdaLNZero, QuestionAttentionPooler
from .vision import (
    AdaptiveQwen2_5_VisionTransformer,
    AdaptiveVisionSpec,
    require_qwen_transformers,
)


def _serialized_config(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise TypeError(
        "adaloop_config must be a mapping or expose to_dict(); got "
        f"{type(value).__name__}"
    )


def restore_missing_adaptive_initializers(
    model: nn.Module,
    missing_keys: Sequence[str] | None,
) -> tuple[int, bool]:
    """Restore custom initialization only for absent adaptive parameters.

    Transformers' missing-key initialization can overwrite AdaLN-Zero output
    heads after construction.  Existing adaptive checkpoint values are never
    touched.  A half-present output-head pair is rejected as corruption.
    """

    missing = set(missing_keys or ())
    reset_adaln_heads = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, AdaLNZero):
            continue
        expected = {
            f"{module_name}.to_adaln.weight",
            f"{module_name}.to_adaln.bias",
        }
        absent = expected & missing
        if absent and absent != expected:
            raise RuntimeError(
                "AdaLN checkpoint is incomplete; expected output-head weight "
                f"and bias together, but only these were missing: {sorted(absent)}"
            )
        if absent:
            nn.init.zeros_(module.to_adaln.weight)
            nn.init.zeros_(module.to_adaln.bias)
            reset_adaln_heads += 1

    conditioner = getattr(model, "question_conditioner", None)
    reset_query = False
    if conditioner is not None:
        query_name = next(
            (
                name
                for name, parameter in model.named_parameters()
                if parameter is conditioner.query
            ),
            None,
        )
        if query_name is None:
            raise RuntimeError("Could not resolve question-conditioner query key")
        if query_name in missing:
            devices: list[int] = []
            if conditioner.query.is_cuda:
                devices = [
                    conditioner.query.device.index
                    if conditioner.query.device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=devices):
                isolated_seed = (torch.initial_seed() + 0x51554553) % (2**63 - 1)
                torch.manual_seed(isolated_seed)
                conditioner.reset_query_parameters()
            reset_query = True
    return reset_adaln_heads, reset_query


def _depth_tuple(depths: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(depths, bool):
        raise TypeError("loop depths must be integers, not bool")
    if isinstance(depths, int):
        values = (depths,)
    else:
        values = tuple(depths)
    if not values:
        raise ValueError("At least one loop depth is required")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("loop depths must contain only integers")
    if any(value < 1 for value in values):
        raise ValueError(f"loop depths must be >= 1, got {values!r}")
    return values


@dataclass
class _QwenDecodeCache:
    past_key_values: Any
    attention_mask: torch.Tensor
    next_text_positions: torch.LongTensor


class AdaptiveConditionalGenerationMixin:
    """Base-independent helpers shared by the concrete and mocked model."""

    supports_per_example_loop_depth = True

    def prepare_inputs_for_generation(
        self,
        *args: Any,
        condition_input_ids: torch.LongTensor | None = None,
        condition_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Keep question conditioning only on the multimodal prefill.

        Transformers forwards otherwise-unknown generation kwargs on every
        cached decode step.  The condition is needed exactly once, while the
        physical image is still present, and retaining it afterwards wastes
        memory and makes accidental re-pooling possible in future parent
        implementations.  ``loop_depth`` remains in ``kwargs`` so one fixed
        recurrent topology is used throughout the generation call.
        """

        prepared = super().prepare_inputs_for_generation(
            *args,
            condition_input_ids=condition_input_ids,
            condition_attention_mask=condition_attention_mask,
            **kwargs,
        )
        prefill = prepared.get("pixel_values") is not None
        prepared["condition_input_ids"] = (
            condition_input_ids if prefill else None
        )
        prepared["condition_attention_mask"] = (
            condition_attention_mask if prefill else None
        )
        return prepared

    @property
    def adaptive_visual(self) -> AdaptiveQwen2_5_VisionTransformer:
        visual = self.model.visual
        if not (
            hasattr(visual, "adaloop_spec")
            and hasattr(visual, "snapshot_runtime_state")
        ):
            raise RuntimeError("This model is using the stock SFT vision tower")
        return visual

    def _input_embedding_layer(self) -> nn.Module:
        language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            language_model = getattr(self, "language_model", None)
        if language_model is None or not hasattr(language_model, "embed_tokens"):
            raise RuntimeError("Could not locate Qwen language token embeddings")
        return language_model.embed_tokens

    def _language_model_module(self) -> nn.Module:
        language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            language_model = getattr(self, "language_model", None)
        if language_model is None:
            raise RuntimeError("Could not locate the Qwen language model")
        return language_model

    @staticmethod
    def _last_active_indices(attention_mask: torch.Tensor) -> torch.LongTensor:
        if attention_mask.ndim != 2:
            raise ValueError(
                "readout attention_mask must be [batch, sequence], got "
                f"{tuple(attention_mask.shape)}"
            )
        positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0)
        last = positions.masked_fill(~attention_mask.bool(), -1).amax(dim=1)
        if bool((last < 0).any()):
            raise ValueError("Every readout row must contain an active token")
        return last

    @staticmethod
    def _next_text_positions(
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.LongTensor:
        if position_ids.ndim != 3 or position_ids.shape[:2] != (
            3,
            attention_mask.shape[0],
        ) or position_ids.shape[2] != attention_mask.shape[1]:
            raise ValueError(
                "Qwen M-RoPE positions must be [3, batch, sequence], got "
                f"{tuple(position_ids.shape)}"
            )
        masked = position_ids.masked_fill(
            ~attention_mask.bool().unsqueeze(0), -1
        )
        return masked.amax(dim=(0, 2)).to(torch.long) + 1

    @staticmethod
    def _uncached_autocast(tensor: torch.Tensor) -> Any:
        device_type = tensor.device.type
        if torch.is_autocast_enabled(device_type):
            return torch.autocast(
                device_type=device_type,
                dtype=torch.get_autocast_dtype(device_type),
                enabled=True,
                cache_enabled=False,
            )
        return nullcontext()

    def pool_question_condition(
        self,
        condition_input_ids: torch.LongTensor,
        condition_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed and pool role-free question tokens for vision-only AdaLN."""

        if self.question_conditioner is None:
            raise RuntimeError("This topology has no question conditioner")
        if condition_input_ids.ndim != 2:
            raise ValueError(
                "condition_input_ids must be [batch, tokens], got "
                f"{tuple(condition_input_ids.shape)}"
            )
        if condition_attention_mask.shape != condition_input_ids.shape:
            raise ValueError(
                "condition_attention_mask must match condition_input_ids; got "
                f"{tuple(condition_attention_mask.shape)} and "
                f"{tuple(condition_input_ids.shape)}"
            )
        token_embeddings = self._input_embedding_layer()(condition_input_ids)
        if self.adaloop_spec.condition_detach:
            token_embeddings = token_embeddings.detach()
        return self.question_conditioner(
            token_embeddings, condition_attention_mask
        )

    def collect_vision_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        condition: torch.Tensor | None,
        depths: int | Iterable[int],
    ) -> tuple[torch.Tensor, ...]:
        """Return merged Variant-A image features at selected 1-based depths.

        The prelude and recurrent core are evaluated once through ``max(depths)``;
        the coda/merger is branched from each intermediate latent.  Results are
        returned in the caller's requested order, including duplicate depths.
        """

        requested = _depth_tuple(depths)
        visual = self.adaptive_visual
        if self.adaloop_spec.variant == "adaln":
            if requested != (1,):
                raise ValueError("Single-pass AdaLN exposes only depth 1")
        elif not self.adaloop_spec.is_recurrent:
            raise ValueError(
                "Vision feature collection requires AdaLN or a recurrent variant"
            )
        if visual.uses_adaln and condition is None:
            raise ValueError("AdaLN feature collection requires a pooled condition")

        runtime_state = visual.snapshot_runtime_state()
        try:
            visual.set_active_condition(condition)
            visual.set_active_loop_steps(max(requested))
            visual.set_active_loop_depths(None)
            visual.set_collect_loop_step_features(True)
            visual_dtype = getattr(visual, "dtype", pixel_values.dtype)
            direct = visual(
                pixel_values.to(dtype=visual_dtype), grid_thw=grid_thw
            )
            if self.adaloop_spec.variant == "adaln":
                # The single-pass tower intentionally does not retain recurrent
                # collections; its direct output is the sole merged feature.
                if hasattr(direct, "pooler_output"):
                    direct = direct.pooler_output
                return (direct,)
            all_features = visual.take_loop_step_features()
            if all_features is None or len(all_features) != max(requested):
                observed = None if all_features is None else len(all_features)
                raise RuntimeError(
                    f"Vision loop returned {observed} states; expected "
                    f"{max(requested)}"
                )
            return tuple(all_features[depth - 1] for depth in requested)
        finally:
            visual.restore_runtime_state(runtime_state)

    def language_forward_from_readout(
        self,
        readout: Any,
        image_features: torch.Tensor,
        *,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> Any:
        """Run Qwen language decoding on an assembled A/B/C readout.

        ``readout`` is intentionally structural: the dataclasses in
        :mod:`adaloop_qwen.model.readouts` all expose IDs, mask, grid metadata,
        and optional labels.  Variant-specific orchestration stays outside the
        core model while image-feature scattering and M-RoPE remain here.
        """

        if _QWEN_CAUSAL_LM is None:
            require_qwen_transformers()
        required = ("input_ids", "attention_mask", "image_grid_thw")
        missing = [name for name in required if not hasattr(readout, name)]
        if missing:
            raise TypeError(
                "Readout is missing required field(s): " + ", ".join(missing)
            )
        input_ids = readout.input_ids
        attention_mask = readout.attention_mask
        image_grid_thw = readout.image_grid_thw
        labels = getattr(readout, "labels", None)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_features = image_features.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        if position_ids is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        return _QWEN_CAUSAL_LM.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def prefill_readout(
        self,
        readout: Any,
        image_features: torch.Tensor,
    ) -> Any:
        """Fresh Qwen prefill implementing ``generation.DecodeRuntime`` state."""

        from .generation import DecodeState

        required = ("input_ids", "attention_mask", "image_grid_thw")
        missing = [name for name in required if not hasattr(readout, name)]
        if missing:
            raise TypeError(
                "Readout is missing required field(s): " + ", ".join(missing)
            )
        input_ids = readout.input_ids
        attention_mask = readout.attention_mask
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Readout IDs and mask must be matching [B, T] tensors")
        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_features = image_features.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        position_ids, _rope_deltas = self.model.get_rope_index(
            input_ids,
            image_grid_thw=readout.image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )
        with self._uncached_autocast(inputs_embeds):
            outputs = self._language_model_module()(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=torch.arange(
                    input_ids.shape[1], device=input_ids.device
                ),
            )
            if outputs.past_key_values is None:
                raise RuntimeError("Qwen readout prefill did not return a KV cache")
            last = self._last_active_indices(attention_mask)
            batch = torch.arange(last.shape[0], device=last.device)
            logits = self.lm_head(outputs.last_hidden_state[batch, last])
        cache = _QwenDecodeCache(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            next_text_positions=self._next_text_positions(
                position_ids, attention_mask
            ),
        )
        return DecodeState(
            logits=logits,
            context_length=int(input_ids.shape[1]),
            cache=cache,
        )

    def append_decode(
        self,
        state: Any,
        token_ids: torch.LongTensor,
        token_mask: torch.Tensor,
    ) -> Any:
        """Append one text-token column to a Qwen ``DecodeState`` cache."""

        from .generation import DecodeState

        if not isinstance(state, DecodeState):
            raise TypeError("state must be a generation.DecodeState")
        if not isinstance(state.cache, _QwenDecodeCache):
            raise TypeError("DecodeState does not contain a Qwen readout cache")
        cache = state.cache
        batch_size = int(cache.attention_mask.shape[0])
        if token_ids.shape != (batch_size, 1):
            raise ValueError(
                f"token_ids must be {(batch_size, 1)}, got {tuple(token_ids.shape)}"
            )
        if token_mask.shape != token_ids.shape:
            raise ValueError("token_mask must match token_ids")
        if token_ids.dtype == torch.bool or torch.is_floating_point(token_ids):
            raise TypeError("token_ids must use an integer dtype")
        emit = token_mask[:, 0].bool()
        token_embeds = self.get_input_embeddings()(token_ids)
        token_position_ids = cache.next_text_positions.view(
            1, batch_size, 1
        ).expand(3, -1, -1)
        cache_length_method = getattr(cache.past_key_values, "get_seq_length", None)
        cache_length = (
            int(cache_length_method())
            if callable(cache_length_method)
            else int(state.context_length)
        )
        mask_column = token_mask.to(
            device=cache.attention_mask.device,
            dtype=cache.attention_mask.dtype,
        )
        attention_mask = torch.cat((cache.attention_mask, mask_column), dim=1)
        with self._uncached_autocast(token_embeds):
            outputs = self._language_model_module()(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=token_position_ids,
                past_key_values=cache.past_key_values,
                inputs_embeds=token_embeds,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=torch.tensor(
                    [cache_length], device=token_ids.device, dtype=torch.long
                ),
            )
            if outputs.past_key_values is None:
                raise RuntimeError("Qwen decode append did not return a KV cache")
            logits = self.lm_head(outputs.last_hidden_state[:, -1])
        next_cache = _QwenDecodeCache(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            next_text_positions=(
                cache.next_text_positions + emit.to(torch.long)
            ),
        )
        return DecodeState(
            logits=logits,
            context_length=state.context_length + 1,
            cache=next_cache,
        )

    def make_decode_runtime(self) -> Any:
        """Return a concrete adapter for Variant-C fresh-reprefill generation."""

        from .readouts import interleave_step_features

        owner = self

        class _Runtime:
            def prefill(self, readout: Any, step_features: Sequence[torch.Tensor]):
                depth = int(getattr(readout, "target_depth", len(step_features)))
                if depth < 1 or len(step_features) != depth:
                    raise ValueError(
                        f"Readout depth {depth} requires {depth} step features; "
                        f"got {len(step_features)}"
                    )
                source_grid = getattr(readout, "source_image_grid_thw", None)
                if source_grid is None:
                    raise TypeError("Readout is missing source_image_grid_thw")
                features = interleave_step_features(
                    step_features,
                    source_grid,
                    spatial_merge_size=int(
                        owner.adaptive_visual.spatial_merge_size
                    ),
                )
                return owner.prefill_readout(readout, features)

            def append(
                self,
                state: Any,
                token_ids: torch.Tensor,
                token_mask: torch.Tensor,
            ):
                return owner.append_decode(state, token_ids, token_mask)

        return _Runtime()

    def forward_depths(
        self,
        depths: int | Iterable[int],
        /,
        **model_inputs: Any,
    ) -> tuple[Any, ...]:
        """Run normal Variant-A forwards at multiple depths.

        This correctness-first interface returns ordinary Qwen causal-LM
        outputs and is intentionally simple.  Readout/training code that needs
        one shared vision execution can use :meth:`collect_vision_features`.
        """

        requested = _depth_tuple(depths)
        if "loop_depth" in model_inputs:
            raise ValueError(
                "forward_depths controls loop depth; do not also pass loop_depth"
            )
        return tuple(
            self.forward(loop_depth=depth, **model_inputs) for depth in requested
        )


_TRANSFORMERS_IMPORT_ERROR: BaseException | None = None
_QWEN_CAUSAL_LM: type[nn.Module] | None = None
try:  # Optional dependency; package inspection/config tests stay framework-light.
    from transformers import AutoConfig as _AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration as _Qwen2_5_VLForConditionalGeneration,
    )

    _QWEN_CAUSAL_LM = _Qwen2_5_VLForConditionalGeneration
except Exception as exc:
    _TRANSFORMERS_IMPORT_ERROR = exc
    _AutoConfig = None


if _QWEN_CAUSAL_LM is not None:

    class AdaptiveQwen2_5_VLForConditionalGeneration(  # type: ignore[misc,valid-type]
        AdaptiveConditionalGenerationMixin,
        _QWEN_CAUSAL_LM,
    ):
        """Qwen2.5-VL with single-pass AdaLN or Variant-A recurrence."""

        accepts_loss_kwargs = False

        def __init__(self, config: Any) -> None:
            _QWEN_CAUSAL_LM.__init__(self, config)
            source = getattr(config, "adaloop_config", None)
            if source is None:
                raise ValueError("HF config must contain adaloop_config")
            self.adaloop_spec = AdaptiveVisionSpec.from_source(
                source, vision_depth=config.vision_config.depth
            )
            if self.adaloop_spec.variant != "sft":
                self.model.visual = AdaptiveQwen2_5_VisionTransformer(
                    config.vision_config, self.adaloop_spec
                )
            if self.adaloop_spec.uses_adaln:
                self.question_conditioner: QuestionAttentionPooler | None = (
                    QuestionAttentionPooler(
                        input_dim=config.text_config.hidden_size,
                        condition_dim=self.adaloop_spec.condition_dim,
                        num_heads=self.adaloop_spec.condition_heads,
                        mlp_ratio=self.adaloop_spec.condition_mlp_ratio,
                    )
                )
                self.question_conditioner.float()
            else:
                self.question_conditioner = None
            self.last_loop_depth: int | None = None
            self.last_loop_depths: torch.Tensor | None = None

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            """HF loader with missing-only restoration of custom initializers."""

            requested_loading_info = bool(kwargs.pop("output_loading_info", False))
            kwargs["output_loading_info"] = True
            model, loading_info = super().from_pretrained(*args, **kwargs)
            if model.question_conditioner is not None:
                model.question_conditioner.float()
            restore_missing_adaptive_initializers(
                model, loading_info.get("missing_keys", ())
            )
            if requested_loading_info:
                return model, loading_info
            return model

        @classmethod
        def from_base_pretrained(
            cls,
            base_model: str,
            *,
            adaloop_config: Any,
            revision: str | None = None,
            **hf_kwargs: Any,
        ) -> "AdaptiveQwen2_5_VLForConditionalGeneration":
            """Instantiate adaptive topology while loading official base weights."""

            serialized = _serialized_config(adaloop_config)
            hf_config = hf_kwargs.pop("config", None)
            if hf_config is None:
                config_kwargs = {
                    key: hf_kwargs[key]
                    for key in (
                        "cache_dir",
                        "force_download",
                        "local_files_only",
                        "proxies",
                        "token",
                        "trust_remote_code",
                    )
                    if key in hf_kwargs
                }
                hf_config = _AutoConfig.from_pretrained(
                    base_model, revision=revision, **config_kwargs
                )
            setattr(hf_config, "adaloop_config", serialized)
            return cls.from_pretrained(
                base_model,
                config=hf_config,
                revision=revision,
                **hf_kwargs,
            )

        def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Any = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            pixel_values: torch.Tensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,
            rope_deltas: torch.LongTensor | None = None,
            cache_position: torch.LongTensor | None = None,
            second_per_grid_ts: torch.Tensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            condition_input_ids: torch.LongTensor | None = None,
            condition_attention_mask: torch.Tensor | None = None,
            loop_depth: int | None = None,
            loop_depths: torch.LongTensor | None = None,
            **kwargs: Any,
        ) -> Any:
            selected_depths = None
            if loop_depths is not None:
                if loop_depth is not None:
                    raise ValueError("loop_depths cannot be combined with loop_depth")
                if self.adaloop_spec.variant != "a":
                    raise ValueError("loop_depths is valid only for Variant A")
                if loop_depths.ndim != 1:
                    raise ValueError(
                        "loop_depths must be [batch], got "
                        f"{tuple(loop_depths.shape)}"
                    )
                if (
                    loop_depths.dtype == torch.bool
                    or torch.is_floating_point(loop_depths)
                    or torch.is_complex(loop_depths)
                ):
                    raise TypeError("loop_depths must use an integer dtype")
                if loop_depths.numel() == 0 or bool((loop_depths < 1).any()):
                    raise ValueError("loop_depths must contain positive depths")
                selected_depths = loop_depths
                selected_depth = int(loop_depths.max().item())
            else:
                selected_depth = None
            if selected_depths is None:
                selected_depth = loop_depth
            if selected_depth is not None:
                if isinstance(selected_depth, bool) or not isinstance(
                    selected_depth, int
                ):
                    raise TypeError("loop_depth must be an integer")
                if selected_depth < 1:
                    raise ValueError("loop_depth must be >= 1")
            if self.adaloop_spec.variant in {"sft", "adaln"}:
                if selected_depth not in {None, 1}:
                    raise ValueError(
                        f"{self.adaloop_spec.variant} supports only loop_depth=1"
                    )
                selected_depth = 1
            elif selected_depth is None:
                selected_depth = self.adaloop_spec.loop_steps
            self.last_loop_depth = selected_depth
            self.last_loop_depths = (
                None
                if selected_depths is None
                else selected_depths.detach().clone()
            )

            if pixel_values_videos is not None and self.adaloop_spec.variant != "sft":
                raise ValueError("Adaptive Qwen2.5-VL execution supports images only")
            if pixel_values is not None and image_grid_thw is None:
                raise ValueError("pixel_values requires image_grid_thw")
            if selected_depths is not None:
                if pixel_values is None:
                    raise ValueError("loop_depths requires pixel_values")
                batch_size = None
                if condition_input_ids is not None:
                    batch_size = int(condition_input_ids.shape[0])
                elif input_ids is not None:
                    batch_size = int(input_ids.shape[0])
                elif inputs_embeds is not None:
                    batch_size = int(inputs_embeds.shape[0])
                if batch_size is None or selected_depths.shape[0] != batch_size:
                    raise ValueError(
                        "loop_depths length must match the language batch size"
                    )
                if image_grid_thw is None or image_grid_thw.shape != (batch_size, 3):
                    observed = (
                        None
                        if image_grid_thw is None
                        else tuple(image_grid_thw.shape)
                    )
                    raise ValueError(
                        "Per-example loop depths require one image grid per "
                        f"example; got {observed} for batch {batch_size}"
                    )
                if input_ids is None:
                    raise ValueError(
                        "Per-example loop depths require input_ids so image "
                        "ownership can be validated"
                    )
                image_tokens = input_ids.eq(int(self.config.image_token_id))
                image_span_starts = image_tokens.clone()
                image_span_starts[:, 1:] = (
                    image_tokens[:, 1:] & ~image_tokens[:, :-1]
                )
                image_spans_per_row = image_span_starts.sum(dim=1)
                if not bool(image_spans_per_row.eq(1).all()):
                    raise ValueError(
                        "Per-example loop depths require exactly one image "
                        "span per example; got "
                        f"{image_spans_per_row.tolist()}"
                    )
            if (
                pixel_values is not None
                and self.adaloop_spec.uses_adaln
                and (
                    condition_input_ids is None
                    or condition_attention_mask is None
                )
            ):
                raise ValueError(
                    "AdaLN execution requires condition_input_ids and "
                    "condition_attention_mask"
                )

            if (
                pixel_values is not None
                and self.adaloop_spec.variant in {"b", "c"}
            ):
                raise RuntimeError(
                    f"Variant {self.adaloop_spec.variant.upper()} requires its "
                    "structured language readout. Call collect_vision_features(), "
                    "assemble the readout with model.readouts, and pass it to "
                    "language_forward_from_readout(); Variant C draft rollout "
                    "uses model.generation.generate_variant_c_reprefill()."
                )

            if self.adaloop_spec.variant == "sft":
                return _QWEN_CAUSAL_LM.forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    second_per_grid_ts=second_per_grid_ts,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )

            visual = self.adaptive_visual
            runtime_state = visual.snapshot_runtime_state()
            try:
                visual.set_active_condition(None)
                visual.set_active_loop_steps(selected_depth)
                visual.set_active_loop_depths(selected_depths)
                visual.set_collect_loop_step_features(False)
                if pixel_values is not None and visual.uses_adaln:
                    visual.set_active_condition(
                        self.pool_question_condition(
                            condition_input_ids, condition_attention_mask
                        )
                    )
                return _QWEN_CAUSAL_LM.forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    second_per_grid_ts=second_per_grid_ts,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )
            finally:
                visual.restore_runtime_state(runtime_state)

else:

    class AdaptiveQwen2_5_VLForConditionalGeneration(
        AdaptiveConditionalGenerationMixin,
        nn.Module,
    ):
        """Unavailable placeholder when Transformers is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            nn.Module.__init__(self)
            require_qwen_transformers()

        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> Any:
            require_qwen_transformers()

        @classmethod
        def from_base_pretrained(cls, *_args: Any, **_kwargs: Any) -> Any:
            require_qwen_transformers()


__all__ = [
    "AdaptiveConditionalGenerationMixin",
    "AdaptiveQwen2_5_VLForConditionalGeneration",
    "restore_missing_adaptive_initializers",
]
