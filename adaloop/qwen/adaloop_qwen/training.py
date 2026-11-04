"""Framework-light training utilities for the Qwen2.5-VL AdaLoop backend.

Imports of Torch, PEFT, safetensors, and Accelerate are intentionally lazy so
configuration and checkpoint inspection continue to work with the package's
minimal dependency set.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from .data import _torch
from .manifest import CANONICAL_NON_LORA_WEIGHTS


NON_LORA_MANIFEST = "non_lora_trainables.json"
TRAINING_STATE = "training_state.json"
TRAINING_CONFIG = "training_config.yaml"
TRAINING_RUN_CONTRACT = "training_run_contract.json"


def geometric_depth_weight_tensor(
    steps: int,
    gamma: float,
    *,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Normalized shallow-to-deep weights ``gamma ** (T - depth)`` as a tensor.

    Unlike ``adaloop_common.geometric_depth_weights``, which normalizes in
    float64, the sum is taken in ``dtype`` (float32 by default), as the
    paper's trainer did.
    """

    torch = _torch()
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(gamma, (int, float)) or isinstance(gamma, bool):
        raise ValueError("gamma must be numeric")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    dtype = torch.float32 if dtype is None else dtype
    raw = torch.tensor(
        [gamma ** (steps - 1 - index) for index in range(steps)],
        device=device,
        dtype=dtype,
    )
    return raw / raw.sum()


def sample_geometric_depths(
    batch_size: int,
    max_depth: int,
    gamma: float,
    *,
    device: Any = None,
    generator: Any = None,
) -> Any:
    """Independently sample 1-based terminal depths for Variant A rows.

    Inside an initialized ``torch.distributed`` process group this is a
    collective that follows the paper's sampler: rank zero draws one depth
    for every row of the global microbatch from its own stream (``generator``
    or the default generator of ``device``) and broadcasts them, and each rank
    keeps its contiguous slice.  Identically seeded ranks therefore still give
    every row an independent depth.  A ``generator`` that is not a
    ``torch.Generator`` on the sampling device raises on every rank before
    the draw.  Without a process group the local draw is used.
    """

    torch = _torch()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    probabilities = geometric_depth_weight_tensor(
        max_depth, gamma, device=device, dtype=torch.float32
    )
    generator_error: Exception | None = None
    if generator is not None and not isinstance(generator, torch.Generator):
        generator_error = TypeError("generator must be a torch.Generator or None")
    elif generator is not None and generator.device.type != probabilities.device.type:
        generator_error = ValueError(
            f"generator is on {generator.device}, but depths are sampled on "
            f"{probabilities.device}; pass a generator for the sampling device"
        )
    distributed = torch.distributed
    if not (distributed.is_available() and distributed.is_initialized()):
        if generator_error is not None:
            raise generator_error
        sampled = torch.multinomial(
            probabilities, batch_size, replacement=True, generator=generator
        )
        return sampled.to(dtype=torch.long) + 1

    rank = distributed.get_rank()
    # Row counts and generator checks share one collective, so a generator
    # rejected on any rank raises on every rank instead of leaving the others
    # blocked in the broadcast below.
    local = torch.tensor(
        [batch_size, int(generator_error is not None)],
        device=probabilities.device,
        dtype=torch.long,
    )
    gathered = [torch.zeros_like(local) for _ in range(distributed.get_world_size())]
    distributed.all_gather(gathered, local)
    rejected = [index for index, entry in enumerate(gathered) if int(entry[1].item())]
    if rejected:
        if generator_error is not None:
            raise generator_error
        raise ValueError(
            f"sample_geometric_depths rejected the generator on rank(s) {rejected}"
        )
    sizes = [int(entry[0].item()) for entry in gathered]
    sampled = torch.empty(sum(sizes), device=probabilities.device, dtype=torch.long)
    if rank == 0:
        sampled.copy_(
            torch.multinomial(
                probabilities, sum(sizes), replacement=True, generator=generator
            )
        )
    distributed.broadcast(sampled, src=0)
    offset = sum(sizes[:rank])
    return sampled[offset : offset + batch_size] + 1


def _set_requires_grad(module: Any, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def apply_language_lora(model: Any, lora_config: Any) -> Any:
    """Attach LoRA only to Qwen language self-attention projections.

    A regex, rather than suffix-only PEFT matching, prevents the question
    pooler's q/k/v/out projections from accidentally receiving adapters.
    """

    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA training requires PEFT; install adaloop-qwen[gpu]"
        ) from exc

    _set_requires_grad(model, False)
    targets = tuple(str(name) for name in lora_config.target_modules)
    alternatives = "|".join(targets)
    target_regex = (
        r".*language_model\.layers\.\d+\.self_attn\."
        rf"({alternatives})"
    )
    peft_config = PeftLoraConfig(
        r=int(lora_config.rank),
        lora_alpha=int(lora_config.alpha),
        lora_dropout=float(lora_config.dropout),
        target_modules=target_regex,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_config)


def _parameter_category(name: str) -> str | None:
    lowered = name.lower()
    if "lora_" in lowered:
        return "lora"
    components = lowered.split(".")
    if "merger" in components:
        return "merger"
    if (
        "question_conditioner" in components
        or "question_pooler" in components
        or "adaln" in components
        or any(component.startswith("adaln") for component in components)
    ):
        return "conditioner"
    return None


@dataclass(frozen=True)
class TrainabilityReport:
    trainable_names: tuple[str, ...]
    frozen_parameters: int
    trainable_parameters: int
    by_category: Mapping[str, tuple[str, ...]]


def configure_trainable_parameters(
    model: Any,
    *,
    train_merger: bool = True,
    train_conditioner: bool = True,
    require_lora: bool = True,
) -> TrainabilityReport:
    """Freeze pretrained weights and enable only release trainables."""

    categorized: dict[str, list[str]] = {
        "lora": [],
        "merger": [],
        "conditioner": [],
    }
    frozen_count = 0
    trainable_count = 0
    for name, parameter in model.named_parameters():
        category = _parameter_category(name)
        enabled = (
            category == "lora"
            or (category == "merger" and train_merger)
            or (category == "conditioner" and train_conditioner)
        )
        parameter.requires_grad_(enabled)
        if enabled:
            categorized[category].append(name)
            trainable_count += int(parameter.numel())
        else:
            frozen_count += int(parameter.numel())

    if require_lora and not categorized["lora"]:
        raise RuntimeError("No language LoRA parameters were found after PEFT setup")
    if train_merger and not categorized["merger"]:
        raise RuntimeError("The model exposes no trainable vision-language merger")
    if train_conditioner and not categorized["conditioner"]:
        raise RuntimeError("The model exposes no AdaLN/question-conditioner parameters")
    names = tuple(
        name for category in ("lora", "merger", "conditioner") for name in categorized[category]
    )
    return TrainabilityReport(
        trainable_names=names,
        frozen_parameters=frozen_count,
        trainable_parameters=trainable_count,
        by_category={key: tuple(value) for key, value in categorized.items()},
    )


# The paper trainer's rule: Transformers 4.57.1
# Trainer.get_decay_parameter_names (nothing inside an nn.LayerNorm, no
# lowercased name matching this pattern) minus names containing "bias",
# which the pattern already excludes.
_NO_DECAY_NAME = re.compile(
    r"bias|layernorm|rmsnorm|(?:^|\.)norm(?:$|\.)|_norm(?:$|\.)"
)


def _uses_weight_decay(name: str, *, in_layer_norm: bool) -> bool:
    return not in_layer_norm and _NO_DECAY_NAME.search(name.lower()) is None


def build_optimizer_groups(model: Any, optimizer_config: Any) -> list[dict[str, Any]]:
    """Build exhaustive, non-overlapping AdamW groups for release trainables.

    Weight decay follows the paper's Trainer: parameters of ``nn.LayerNorm``
    modules and parameters whose names contain ``bias`` or a norm pattern are
    not decayed; every other trainable is, 1-D or not (e.g. the merger's
    RMSNorm ``ln_q.weight``).
    """

    torch = _torch()
    layer_norm_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, torch.nn.LayerNorm)
        for parameter in module.parameters()
    }
    learning_rates = {
        "lora": float(optimizer_config.lora_lr),
        "merger": float(optimizer_config.merger_lr),
        "conditioner": float(optimizer_config.conditioner_lr),
    }
    buckets: dict[tuple[str, bool], list[Any]] = {}
    seen: set[int] = set()
    unclassified: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        if identifier in seen:
            raise RuntimeError(f"Trainable parameter is named more than once: {name}")
        seen.add(identifier)
        category = _parameter_category(name)
        if category is None:
            unclassified.append(name)
            continue
        if learning_rates[category] <= 0:
            raise ValueError(
                f"Trainable {category} parameter {name!r} has a non-positive learning rate"
            )
        decay = _uses_weight_decay(
            name, in_layer_norm=identifier in layer_norm_parameters
        )
        buckets.setdefault((category, decay), []).append(parameter)
    if unclassified:
        raise RuntimeError(
            "Trainable parameters are absent from all optimizer groups: "
            + ", ".join(sorted(unclassified))
        )
    if not seen:
        raise RuntimeError("The model has no trainable parameters")

    groups: list[dict[str, Any]] = []
    for category in ("lora", "merger", "conditioner"):
        for decay in (True, False):
            parameters = buckets.get((category, decay), [])
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rates[category],
                        "weight_decay": (
                            float(optimizer_config.weight_decay) if decay else 0.0
                        ),
                        "group_name": f"{category}_{'decay' if decay else 'no_decay'}",
                    }
                )
    if sum(len(group["params"]) for group in groups) != len(seen):
        raise RuntimeError("Optimizer grouping did not cover each trainable parameter once")
    return groups


def build_adamw(model: Any, optimizer_config: Any) -> Any:
    """Construct AdamW with the recipe's category-specific learning rates."""

    torch = _torch()
    return torch.optim.AdamW(
        build_optimizer_groups(model, optimizer_config),
        betas=(float(optimizer_config.beta1), float(optimizer_config.beta2)),
        eps=float(optimizer_config.epsilon),
    )


def build_cosine_scheduler(
    optimizer: Any,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> Any:
    """Build the recipe's update-based warmup + cosine LR schedule.

    ``total_steps`` counts optimizer updates, not dataloader microbatches.  A
    positive warmup begins at zero and reaches the configured group learning
    rates at ``warmup_steps``; cosine decay then reaches ``min_lr_ratio`` at
    the final update.  ``LambdaLR`` preserves the relative rates of the LoRA,
    merger, and conditioner parameter groups.
    """

    torch = _torch()
    for name, value in (
        ("total_steps", total_steps),
        ("warmup_steps", warmup_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if isinstance(min_lr_ratio, bool) or not isinstance(
        min_lr_ratio, (int, float)
    ):
        raise ValueError("min_lr_ratio must be numeric")
    floor = float(min_lr_ratio)
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("min_lr_ratio must be finite and in [0, 1]")

    def lr_scale(current_step: int) -> float:
        if warmup_steps and current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = min(
            1.0,
            max(0.0, float(current_step - warmup_steps) / float(decay_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)


@dataclass(frozen=True)
class TrainingObjective:
    loss: Any
    weights: Any
    depth_losses: tuple[Any, ...]
    per_example_losses: tuple[Any, ...]
    model_outputs: tuple[Any, ...]
    sampled_depths: Any | None = None


DepthForwardHook = Callable[
    [Any, tuple[int, ...], Mapping[str, Any], Any], Any
]


@dataclass(frozen=True)
class TrainingObjectiveHooks:
    """Variant-specific structured-readout entry points.

    A hook receives ``(model, depths, model_inputs, config)`` and returns the
    same ordered outputs accepted from ``model.forward_depths``.  The public
    release hooks below implement B and C; downstream ports can replace one
    without changing loss reduction, depth weighting, or checkpoint logic.
    """

    depth_forwards: Mapping[str, DepthForwardHook]

    def depth_forward(self, variant: str) -> DepthForwardHook | None:
        return self.depth_forwards.get(str(variant).lower())


def _value(output: Any, name: str, default: Any = None) -> Any:
    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def _walk_wrappers(model: Any) -> list[Any]:
    pending = [model]
    result: list[Any] = []
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        result.append(candidate)
        wrapped = getattr(candidate, "module", None)
        if wrapped is not None and wrapped is not candidate:
            pending.append(wrapped)
        getter = getattr(candidate, "get_base_model", None)
        if callable(getter):
            try:
                base = getter()
            except TypeError:
                base = None
            if base is not None and base is not candidate:
                pending.append(base)
        base = getattr(candidate, "base_model", None)
        if base is not None and base is not candidate:
            pending.append(base)
    return result


def _find_method(model: Any, name: str) -> Any | None:
    for candidate in _walk_wrappers(model):
        method = getattr(candidate, name, None)
        if callable(method):
            return method
    return None


def _explicit_forward_keyword(model: Any, name: str) -> bool:
    for candidate in _walk_wrappers(model):
        capability = getattr(candidate, f"supports_{name}", None)
        if name == "loop_depths":
            capability = getattr(
                candidate, "supports_per_example_loop_depth", capability
            )
        if capability is False:
            continue
        if capability is True:
            return True
        forward = getattr(candidate, "forward", None)
        if not callable(forward):
            continue
        try:
            if name in inspect.signature(forward).parameters:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in batch.items()
        if key not in {"sample_id", "sample_ids", "question", "answer"}
    }


def _labels_for_output(output: Any, fallback: Any) -> Any:
    for name in ("labels", "training_labels", "expanded_labels"):
        labels = _value(output, name)
        if labels is not None:
            return labels
    return fallback


def _per_example_loss(output: Any, fallback_labels: Any) -> Any:
    torch = _torch()
    supplied = _value(output, "per_example_loss")
    if supplied is None:
        supplied = _value(output, "per_example_losses")
    if isinstance(supplied, torch.Tensor):
        if supplied.ndim != 1:
            raise ValueError("Model per_example_loss must have shape [B]")
        return supplied
    logits = _value(output, "logits")
    if logits is None:
        raise RuntimeError(
            "Training outputs must expose logits or an explicit per_example_loss"
        )
    from .model.losses import causal_cross_entropy_per_example

    labels = _labels_for_output(output, fallback_labels)
    return causal_cross_entropy_per_example(logits, labels)


def _normalize_depth_outputs(
    outputs: Any, depths: Sequence[int]
) -> tuple[Any, ...]:
    nested = _value(outputs, "depth_outputs")
    if nested is not None:
        outputs = nested
    if isinstance(outputs, Mapping):
        if len(depths) == 1 and not (
            depths[0] in outputs or str(depths[0]) in outputs
        ):
            # Hugging Face ModelOutput objects are mappings too.  A single
            # ordinary output is valid for a one-depth hook invocation and
            # must not be mistaken for a depth-keyed mapping.
            return (outputs,)
        resolved = []
        for depth in depths:
            if depth in outputs:
                resolved.append(outputs[depth])
            elif str(depth) in outputs:
                resolved.append(outputs[str(depth)])
            else:
                raise RuntimeError(f"forward_depths omitted depth {depth}")
        return tuple(resolved)
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != len(depths):
            raise RuntimeError(
                f"forward_depths returned {len(outputs)} outputs for {len(depths)} depths"
            )
        return tuple(outputs)
    if len(depths) == 1:
        return (outputs,)
    raise RuntimeError("forward_depths must return an ordered sequence or depth mapping")


def _structured_runtime(model: Any, *, variant: str) -> Any:
    required = (
        "collect_vision_features",
        "language_forward_from_readout",
        "pool_question_condition",
    )
    eligible = [
        candidate
        for candidate in _walk_wrappers(model)
        if all(callable(getattr(candidate, name, None)) for name in required)
    ]
    if eligible:
        # PEFT/DDP wrappers often proxy arbitrary attributes.  Prefer the
        # concrete owner so config, visual modules, and decode-runtime methods
        # all come from the same object rather than a partially transparent
        # wrapper.
        def direct_capabilities(candidate: Any) -> int:
            return sum(
                any(name in cls.__dict__ for cls in type(candidate).__mro__)
                for name in required
            )

        return max(eligible, key=direct_capabilities)
    raise RuntimeError(
        f"Variant {variant.upper()} training requires model capabilities: "
        + ", ".join(required)
    )


def _configuration_value(runtime: Any, name: str) -> Any:
    config = getattr(runtime, "config", None)
    value = None if config is None else getattr(config, name, None)
    if value is None and config is not None:
        text_config = getattr(config, "text_config", None)
        value = None if text_config is None else getattr(text_config, name, None)
    if value is None:
        raise RuntimeError(f"Qwen runtime config does not define {name}")
    return value


def _runtime_pad_token_id(runtime: Any, override: int | None) -> int:
    """Resolve the ID that fills masked positions of B/C readouts.

    The pinned base ``config.json`` defines no pad ID.  The paper's training ran
    under HF Trainer, whose ``_align_special_tokens`` copied
    ``tokenizer.pad_token_id`` (151643, ``<|endoftext|>``) into the model
    config before the first step; the base generation config carries the same
    ID.  An explicit tokenizer value therefore wins, then the model and text
    configs, then the generation config.
    """

    value = override
    config = getattr(runtime, "config", None)
    for source in (
        config,
        getattr(config, "text_config", None),
        getattr(runtime, "generation_config", None),
    ):
        if value is not None:
            break
        value = getattr(source, "pad_token_id", None)
    if value is None:
        raise RuntimeError(
            "Qwen runtime defines no pad_token_id in its config, text_config, or "
            "generation_config; pass the tokenizer's pad ID to "
            "release_readout_objective_hooks(pad_token_id=...)"
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"pad_token_id must be a non-negative integer; got {value!r}")
    return int(value)


def _spatial_merge_size(runtime: Any) -> int:
    config = getattr(runtime, "config", None)
    vision = None if config is None else getattr(config, "vision_config", None)
    value = None if vision is None else getattr(vision, "spatial_merge_size", None)
    if value is None:
        visual = getattr(getattr(runtime, "model", None), "visual", None)
        value = getattr(visual, "spatial_merge_size", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError("Qwen runtime does not expose a positive spatial_merge_size")
    return int(value)


def _structured_inputs(
    model_inputs: Mapping[str, Any], *, variant: str
) -> tuple[Any, Any, Any, Any, Any]:
    required = (
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "image_grid_thw",
    )
    missing = [name for name in required if model_inputs.get(name) is None]
    if missing:
        raise ValueError(
            f"Variant {variant.upper()} batch is missing field(s): "
            + ", ".join(missing)
        )
    return tuple(model_inputs[name] for name in required)  # type: ignore[return-value]


def _pooled_condition(runtime: Any, model_inputs: Mapping[str, Any]) -> Any:
    condition_ids = model_inputs.get("condition_input_ids")
    condition_mask = model_inputs.get("condition_attention_mask")
    if condition_ids is None or condition_mask is None:
        raise ValueError(
            "Structured recurrent readout requires condition_input_ids and "
            "condition_attention_mask"
        )
    return runtime.pool_question_condition(condition_ids, condition_mask)


def _optional_pooled_condition(
    runtime: Any, model_inputs: Mapping[str, Any]
) -> Any | None:
    condition_ids = model_inputs.get("condition_input_ids")
    condition_mask = model_inputs.get("condition_attention_mask")
    if condition_ids is None and condition_mask is None:
        return None
    if condition_ids is None or condition_mask is None:
        raise ValueError(
            "condition_input_ids and condition_attention_mask must be supplied together"
        )
    return runtime.pool_question_condition(condition_ids, condition_mask)


def _variant_a_depth_forward(
    model: Any,
    depths: tuple[int, ...],
    model_inputs: Mapping[str, Any],
    config: Any,
) -> tuple[Any, ...]:
    """Share one recurrent vision trajectory across Variant-A depth losses."""

    del config
    runtime = _structured_runtime(model, variant="a")
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("Variant-A supervised depths must be positive")
    if tuple(sorted(set(depths))) != depths:
        raise ValueError("Variant-A supervised depths must be unique and increasing")
    input_ids, attention_mask, labels, pixel_values, image_grid_thw = (
        _structured_inputs(model_inputs, variant="a")
    )
    condition = _optional_pooled_condition(runtime, model_inputs)
    step_features = runtime.collect_vision_features(
        pixel_values, image_grid_thw, condition, depths
    )
    if len(step_features) != len(depths):
        raise RuntimeError(
            f"collect_vision_features returned {len(step_features)} values for "
            f"{len(depths)} requested depths"
        )
    readout = SimpleNamespace(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        image_grid_thw=image_grid_thw,
    )
    outputs: list[Any] = []
    for image_features in step_features:
        raw_output = runtime.language_forward_from_readout(readout, image_features)
        logits = _value(raw_output, "logits")
        if logits is None:
            raise RuntimeError("Variant-A language readout did not return logits")
        outputs.append(
            SimpleNamespace(
                logits=logits,
                training_labels=labels,
                raw_output=raw_output,
            )
        )
    return tuple(outputs)


def _variant_b_depth_forward(
    model: Any,
    depths: tuple[int, ...],
    model_inputs: Mapping[str, Any],
    config: Any,
    *,
    max_length: int | None,
    pad_token_id: int | None = None,
) -> tuple[Any, ...]:
    del config
    from .model.readouts import (
        build_accumulated_image_readout,
        interleave_step_features,
    )

    runtime = _structured_runtime(model, variant="b")
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("Variant-B supervised depths must be positive")
    if tuple(sorted(set(depths))) != depths:
        raise ValueError("Variant-B supervised depths must be unique and increasing")
    input_ids, attention_mask, labels, pixel_values, image_grid_thw = (
        _structured_inputs(model_inputs, variant="b")
    )
    merge_size = _spatial_merge_size(runtime)
    pad_id = _runtime_pad_token_id(runtime, pad_token_id)

    def accumulated_readout(depth: int) -> Any:
        return build_accumulated_image_readout(
            input_ids,
            attention_mask,
            image_grid_thw,
            target_depth=depth,
            image_token_id=int(_configuration_value(runtime, "image_token_id")),
            vision_start_token_id=int(
                _configuration_value(runtime, "vision_start_token_id")
            ),
            vision_end_token_id=int(
                _configuration_value(runtime, "vision_end_token_id")
            ),
            spatial_merge_size=merge_size,
            labels=labels,
            pad_token_id=pad_id,
            max_length=max_length,
        )

    # As the paper's collator did, expand every row to the terminal depth
    # before any model compute, so an over-length row is rejected with its
    # full expanded length; shallower readouts are never longer.
    terminal_depth = max(depths)
    terminal_readout = accumulated_readout(terminal_depth)
    condition = _pooled_condition(runtime, model_inputs)
    recurrent_depths = tuple(range(1, terminal_depth + 1))
    step_features = runtime.collect_vision_features(
        pixel_values, image_grid_thw, condition, recurrent_depths
    )
    if len(step_features) != len(recurrent_depths):
        raise RuntimeError(
            f"collect_vision_features returned {len(step_features)} values for "
            f"{len(recurrent_depths)} requested recurrent depths"
        )
    outputs: list[Any] = []
    for depth in depths:
        readout = (
            terminal_readout
            if depth == terminal_depth
            else accumulated_readout(depth)
        )
        language_features = interleave_step_features(
            step_features[:depth],
            image_grid_thw,
            spatial_merge_size=merge_size,
        )
        language_readout = SimpleNamespace(
            input_ids=readout.input_ids,
            attention_mask=readout.attention_mask,
            labels=None,
            image_grid_thw=readout.image_grid_thw,
        )
        raw_output = runtime.language_forward_from_readout(
            language_readout, language_features
        )
        logits = _value(raw_output, "logits")
        if logits is None:
            raise RuntimeError("Variant-B language readout did not return logits")
        outputs.append(
            SimpleNamespace(
                logits=logits,
                training_labels=readout.labels,
                raw_output=raw_output,
            )
        )
    return tuple(outputs)


class _ReprefillDraftRuntime:
    """Correctness-first Variant-C decoder using full prefix reprefills."""

    def __init__(self, runtime: Any, *, pad_token_id: int, merge_size: int) -> None:
        self.runtime = runtime
        self.pad_token_id = int(pad_token_id)
        self.merge_size = int(merge_size)

    @staticmethod
    def _next_logits(output: Any, attention_mask: Any) -> Any:
        torch = _torch()
        logits = _value(output, "logits")
        if logits is None or logits.ndim != 3:
            raise RuntimeError("Variant-C prefill must return logits [B, T, V]")
        active = attention_mask.bool()
        if torch.any(~active.any(dim=1)):
            raise ValueError("Variant-C prefill contains an empty row")
        positions = torch.arange(active.shape[1], device=active.device)[None, :]
        last = positions.masked_fill(~active, -1).max(dim=1).values
        rows = torch.arange(active.shape[0], device=active.device)
        return logits[rows, last]

    def _prefill(self, readout: Any, step_features: Sequence[Any]) -> Any:
        from .model.generation import DecodeState
        from .model.readouts import interleave_step_features

        image_features = interleave_step_features(
            step_features,
            readout.source_image_grid_thw,
            spatial_merge_size=self.merge_size,
        )
        output = self.runtime.language_forward_from_readout(
            readout, image_features, use_cache=False
        )
        return DecodeState(
            logits=self._next_logits(output, readout.attention_mask),
            context_length=int(readout.input_ids.shape[1]),
            cache=(readout, tuple(step_features), image_features),
        )

    def prefill(self, readout: Any, step_features: Sequence[Any]) -> Any:
        return self._prefill(readout, step_features)

    def append(self, state: Any, token_ids: Any, token_mask: Any) -> Any:
        torch = _torch()
        readout, step_features, image_features = state.cache
        stored = torch.where(
            token_mask.bool(),
            token_ids,
            torch.full_like(token_ids, self.pad_token_id),
        )
        input_ids = torch.cat((readout.input_ids, stored), dim=1)
        attention_mask = torch.cat(
            (readout.attention_mask, token_mask.to(readout.attention_mask.dtype)),
            dim=1,
        )
        appended = SimpleNamespace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            image_grid_thw=readout.image_grid_thw,
            source_image_grid_thw=readout.source_image_grid_thw,
        )
        output = self.runtime.language_forward_from_readout(
            appended, image_features, use_cache=False
        )
        from .model.generation import DecodeState

        return DecodeState(
            logits=self._next_logits(output, attention_mask),
            context_length=int(input_ids.shape[1]),
            cache=(appended, step_features, image_features),
        )


def _variant_c_depth_forward(
    model: Any,
    depths: tuple[int, ...],
    model_inputs: Mapping[str, Any],
    config: Any,
    *,
    max_length: int | None,
    pad_token_id: int | None = None,
) -> tuple[Any, ...]:
    from .model.generation import rollout_variant_c_drafts_reprefill
    from .model.prompting import TokenBatch, VariantCPromptFragments
    from .model.readouts import (
        build_interleaved_draft_readout,
        interleave_step_features,
        prepare_interleaved_draft_readout,
    )

    runtime = _structured_runtime(model, variant="c")
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("Variant-C supervised depths must be positive")
    if tuple(sorted(set(depths))) != depths:
        raise ValueError("Variant-C supervised depths must be unique and increasing")
    input_ids, attention_mask, _labels, pixel_values, image_grid_thw = (
        _structured_inputs(model_inputs, variant="c")
    )
    condition = _pooled_condition(runtime, model_inputs)
    recurrent_depths = tuple(range(1, max(depths) + 1))
    step_features = runtime.collect_vision_features(
        pixel_values, image_grid_thw, condition, recurrent_depths
    )
    if len(step_features) != len(recurrent_depths):
        raise RuntimeError(
            f"collect_vision_features returned {len(step_features)} values for "
            f"{len(recurrent_depths)} requested recurrent depths"
        )

    def token_batch(name: str) -> Any:
        ids = model_inputs.get(f"trajectory_{name}_input_ids")
        mask = model_inputs.get(f"trajectory_{name}_attention_mask")
        if ids is None or mask is None:
            raise ValueError(
                f"Variant C batch is missing trajectory_{name}_input_ids/mask"
            )
        return TokenBatch(ids, mask)

    fragments = VariantCPromptFragments(
        question=token_batch("question"),
        answer=token_batch("answer"),
        cue=token_batch("cue"),
        separator=token_batch("separator"),
    )
    merge_size = _spatial_merge_size(runtime)
    pad_id = _runtime_pad_token_id(runtime, pad_token_id)
    prepared = prepare_interleaved_draft_readout(
        input_ids,
        attention_mask,
        image_grid_thw,
        fragments,
        image_token_id=int(_configuration_value(runtime, "image_token_id")),
        vision_start_token_id=int(
            _configuration_value(runtime, "vision_start_token_id")
        ),
        vision_end_token_id=int(
            _configuration_value(runtime, "vision_end_token_id")
        ),
        spatial_merge_size=merge_size,
    )
    if max_length is not None:
        # As the paper's collator did, reject a row whose trajectory would
        # exceed the context if every draft used its whole token budget.
        terminal = max(recurrent_depths)
        budget = int(config.readout.draft_max_tokens)
        worst_case = [
            terminal * (container.numel() + question.numel() + cue.numel())
            + (terminal - 1) * (separator.numel() + budget)
            + answer.numel()
            for container, question, cue, separator, answer in zip(
                prepared.containers,
                prepared.questions,
                prepared.cues,
                prepared.separators,
                prepared.answers,
                strict=True,
            )
        ]
        overflow = [
            f"sample {row}: {length}"
            for row, length in enumerate(worst_case)
            if length > max_length
        ]
        if overflow:
            raise ValueError(
                f"Worst-case Variant-C trajectory exceeds max_length={max_length} "
                f"({', '.join(overflow)})"
            )
    suppressed = model_inputs.get("trajectory_suppressed_token_ids")
    if suppressed is None:
        raise ValueError("Variant C batch is missing trajectory_suppressed_token_ids")
    decode_factory = getattr(runtime, "make_decode_runtime", None)
    if callable(decode_factory):
        decode_runtime = decode_factory()
    else:
        # This path keeps custom structured runtimes usable while retaining
        # exact semantics.  The public Qwen runtime supplies the faster native
        # M-RoPE/KV-cache adapter above.
        decode_runtime = _ReprefillDraftRuntime(
            runtime, pad_token_id=pad_id, merge_size=merge_size
        )
    if not all(
        callable(getattr(decode_runtime, name, None))
        for name in ("prefill", "append")
    ):
        raise RuntimeError(
            "Variant C decode runtime must implement prefill() and append()"
        )
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        rollout = rollout_variant_c_drafts_reprefill(
            decode_runtime,
            prepared,
            step_features,
            target_depth=max(recurrent_depths),
            draft_max_tokens=int(config.readout.draft_max_tokens),
            eos_token_ids=_configuration_value(runtime, "eos_token_id"),
            pad_token_id=pad_id,
            suppressed_token_ids=suppressed,
            max_context_length=max_length,
        )
    finally:
        model.train(was_training)

    outputs: list[Any] = []
    for depth in depths:
        readout = build_interleaved_draft_readout(
            prepared,
            rollout.drafts[: depth - 1],
            rollout.draft_attention_masks[: depth - 1],
            target_depth=depth,
            pad_token_id=pad_id,
            draft_padding_side="right",
            padding_side="right",
            max_length=max_length,
            include_answer=True,
        )
        language_features = interleave_step_features(
            step_features[:depth],
            image_grid_thw,
            spatial_merge_size=merge_size,
        )
        language_readout = SimpleNamespace(
            input_ids=readout.input_ids,
            attention_mask=readout.attention_mask,
            labels=None,
            image_grid_thw=readout.image_grid_thw,
        )
        raw_output = runtime.language_forward_from_readout(
            language_readout, language_features
        )
        logits = _value(raw_output, "logits")
        if logits is None:
            raise RuntimeError("Variant-C language readout did not return logits")
        outputs.append(
            SimpleNamespace(
                logits=logits,
                training_labels=readout.labels,
                raw_output=raw_output,
            )
        )
    return tuple(outputs)


def release_readout_objective_hooks(
    *, max_length: int | None = None, pad_token_id: int | None = None
) -> TrainingObjectiveHooks:
    """Return native shared-vision hooks for recurrent Variants A, B, and C.

    ``pad_token_id`` should be the tokenizer pad ID that the collator uses.
    When omitted, B/C resolve it from the model config and then from the
    generation config, which is where the pinned Qwen base declares it.
    """

    if max_length is not None and (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 1
    ):
        raise ValueError("max_length must be a positive integer or None")
    if pad_token_id is not None and (
        isinstance(pad_token_id, bool)
        or not isinstance(pad_token_id, int)
        or pad_token_id < 0
    ):
        raise ValueError("pad_token_id must be a non-negative integer or None")

    return TrainingObjectiveHooks(
        depth_forwards={
            "a": _variant_a_depth_forward,
            "b": partial(
                _variant_b_depth_forward,
                max_length=max_length,
                pad_token_id=pad_token_id,
            ),
            "c": partial(
                _variant_c_depth_forward,
                max_length=max_length,
                pad_token_id=pad_token_id,
            ),
        }
    )


def _fixed_depth_forward(
    model: Any,
    inputs: Mapping[str, Any],
    *,
    depth: int,
    variant: str,
) -> Any:
    if variant == "sft":
        return model(**inputs)
    # A generic **kwargs signature is never treated as proof of support.
    if _explicit_forward_keyword(model, "loop_depth"):
        return model(**inputs, loop_depth=depth)
    raise RuntimeError(
        f"Variant {variant!r} training requires model.forward(..., loop_depth=int)"
    )


def compute_training_objective(
    model: Any,
    batch: Mapping[str, Any],
    config: Any,
    *,
    generator: Any = None,
    hooks: TrainingObjectiveHooks | None = None,
) -> TrainingObjective:
    """Dispatch terminal, deeply supervised, or sampled-depth training.

    Sampled-depth training is intentionally strict: every row receives an
    independently sampled depth and the model must explicitly implement the
    ``loop_depths: LongTensor[B]`` contract.  It never degrades to one scalar
    depth shared by the batch.  Under ``torch.distributed`` the draw covers the
    global microbatch (see :func:`sample_geometric_depths`), so every rank
    must run this objective for each microbatch.
    """

    torch = _torch()
    if "labels" not in batch:
        raise ValueError("Training batch is missing labels")
    labels = batch["labels"]
    if labels.ndim != 2:
        raise ValueError("Training labels must have shape [B, T]")
    inputs = _model_inputs(batch)
    variant = str(config.variant).lower()
    supervision = str(config.training.supervision)
    steps = int(config.train_depth)

    if supervision == "terminal":
        hook = None if hooks is None else hooks.depth_forward(variant)
        if hook is not None:
            output = _normalize_depth_outputs(
                hook(model, (steps,), inputs, config), (steps,)
            )[0]
        elif variant in {"b", "c"}:
            raise RuntimeError(
                f"Variant {variant.upper()} terminal supervision requires a "
                "structured readout hook; pass release_readout_objective_hooks() "
                "or a custom TrainingObjectiveHooks implementation"
            )
        else:
            output = _fixed_depth_forward(
                model, inputs, depth=steps, variant=variant
            )
        row_losses = _per_example_loss(output, labels)
        loss = row_losses.mean()
        weights = torch.ones(1, device=loss.device, dtype=loss.dtype)
        return TrainingObjective(
            loss=loss,
            weights=weights,
            depth_losses=(loss,),
            per_example_losses=(row_losses,),
            model_outputs=(output,),
        )

    gamma = config.training.geometric_ratio
    if gamma is None:
        raise ValueError(f"{supervision} requires training.geometric_ratio")

    if supervision == "deep_geometric":
        depths = tuple(range(1, steps + 1))
        hook = None if hooks is None else hooks.depth_forward(variant)
        if variant in {"b", "c"}:
            if hook is None:
                raise RuntimeError(
                    f"Variant {variant.upper()} deep supervision requires a "
                    "structured readout hook; pass release_readout_objective_hooks() "
                    "or a custom TrainingObjectiveHooks implementation"
                )
            raw_outputs = hook(model, depths, inputs, config)
        elif hook is not None:
            raw_outputs = hook(model, depths, inputs, config)
        else:
            method = _find_method(model, "forward_depths")
            if method is None:
                raise RuntimeError(
                    "Geometric deep supervision requires "
                    "model.forward_depths(depths, **inputs)"
                )
            raw_outputs = method(depths, **inputs)
        outputs = _normalize_depth_outputs(raw_outputs, depths)
        row_losses = tuple(_per_example_loss(output, labels) for output in outputs)
        depth_losses = tuple(values.mean() for values in row_losses)
        stacked = torch.stack(depth_losses)
        weights = geometric_depth_weight_tensor(
            steps, float(gamma), device=stacked.device, dtype=stacked.dtype
        )
        return TrainingObjective(
            loss=torch.sum(weights * stacked),
            weights=weights,
            depth_losses=depth_losses,
            per_example_losses=row_losses,
            model_outputs=outputs,
        )

    if supervision == "sampled_geometric":
        if variant != "a":
            raise ValueError("Sampled-depth supervision is supported only for Variant A")
        if not _explicit_forward_keyword(model, "loop_depths"):
            raise RuntimeError(
                "Sampled Variant A requires explicit per-example "
                "model.forward(..., loop_depths=LongTensor[B]) support; scalar "
                "batch-depth fallback is disabled"
            )
        sampled = sample_geometric_depths(
            labels.shape[0],
            steps,
            float(gamma),
            device=labels.device,
            generator=generator,
        )
        output = model(**inputs, loop_depths=sampled)
        row_losses = _per_example_loss(output, labels)
        loss = row_losses.mean()
        weights = geometric_depth_weight_tensor(
            steps, float(gamma), device=loss.device, dtype=loss.dtype
        )
        return TrainingObjective(
            loss=loss,
            weights=weights,
            depth_losses=(loss,),
            per_example_losses=(row_losses,),
            model_outputs=(output,),
            sampled_depths=sampled,
        )

    raise ValueError(f"Unsupported supervision mode: {supervision!r}")


def make_training_objective_module(
    model: Any,
    config: Any,
    *,
    hooks: TrainingObjectiveHooks | None = None,
) -> Any:
    """Wrap objective dispatch in ``nn.Module.forward`` for DDP/Accelerate."""

    torch = _torch()

    class _ObjectiveModule(torch.nn.Module):
        def __init__(
            self,
            wrapped: Any,
            recipe: Any,
            objective_hooks: TrainingObjectiveHooks | None,
        ) -> None:
            super().__init__()
            self.adaloop_model = wrapped
            self.recipe = recipe
            # Hooks contain callables rather than parameters and deliberately
            # remain plain Python state outside the module hierarchy.
            self.objective_hooks = objective_hooks

        def forward(self, batch: Mapping[str, Any]) -> Any:
            objective = compute_training_objective(
                self.adaloop_model,
                batch,
                self.recipe,
                hooks=self.objective_hooks,
            )
            return objective.loss

    return _ObjectiveModule(model, config, hooks)


@dataclass(frozen=True)
class TrainingState:
    global_step: int = 0
    epoch: int = 0
    batches_in_epoch: int = 0
    format_version: int = 1

    def __post_init__(self) -> None:
        for name in ("global_step", "epoch", "batches_in_epoch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.format_version != 1:
            raise ValueError("Unsupported training-state format_version")

    def to_dict(self) -> dict[str, int]:
        return {
            "format_version": self.format_version,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batches_in_epoch": self.batches_in_epoch,
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingState":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read training state {source}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("Training state must be a JSON object")
        allowed = {"format_version", "global_step", "epoch", "batches_in_epoch"}
        if set(raw) != allowed:
            raise ValueError(
                "Training state fields must be exactly: " + ", ".join(sorted(allowed))
            )
        return cls(**{name: raw[name] for name in allowed})


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_json_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain only finite JSON values") from exc
    return decoded


def _recipe_payload(config: Any) -> dict[str, Any]:
    serializer = getattr(config, "to_dict", None)
    if not callable(serializer):
        raise TypeError("Training config must implement to_dict()")
    return _canonical_json_mapping(
        serializer(), context="serialized training recipe"
    )


def _mapping_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    """Name every differing leaf by its dotted path, e.g. ``training.seed``."""

    def walk(left: Mapping[str, Any], right: Mapping[str, Any], prefix: str) -> list[str]:
        names: list[str] = []
        for name in sorted(set(left) | set(right)):
            path = f"{prefix}{name}"
            if name not in left or name not in right:
                names.append(path)
            elif isinstance(left[name], Mapping) and isinstance(right[name], Mapping):
                names.extend(walk(left[name], right[name], f"{path}."))
            elif left[name] != right[name]:
                names.append(path)
        return names

    names = walk(left, right, "")
    return ", ".join(names) if names else "unknown nested value"


def _training_model(model: Any) -> Any:
    if hasattr(model, "adaloop_model"):
        return model.adaloop_model
    return model


def save_compact_trainables(model: Any, output_dir: str | Path) -> Path:
    """Save PEFT adapter weights plus every trainable non-LoRA tensor."""

    from safetensors.torch import save_file

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model = _training_model(model)
    if hasattr(model, "peft_config"):
        card = root / "README.md"
        had_card = card.exists()
        model.save_pretrained(root, safe_serialization=True)
        if not had_card:
            # PEFT also writes a model card naming the base model as it was
            # loaded, possibly a local path; checkpoints do not include it.
            card.unlink(missing_ok=True)

    trainable_non_lora = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name.lower()
    }
    state = model.state_dict()
    sidecar = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if name in trainable_non_lora
    }
    missing = sorted(trainable_non_lora - set(sidecar))
    if missing:
        raise RuntimeError(
            "Trainable non-LoRA parameters are missing from state_dict: "
            + ", ".join(missing)
        )

    weights_path = root / CANONICAL_NON_LORA_WEIGHTS
    temporary = weights_path.with_suffix(weights_path.suffix + ".tmp")
    save_file(sidecar, str(temporary))
    temporary.replace(weights_path)

    _atomic_json(
        root / NON_LORA_MANIFEST,
        {
            "format_version": 1,
            "weights": weights_path.name,
            "format": "safetensors",
            "parameters": {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in sorted(sidecar.items())
            },
        },
    )
    return weights_path


_TRAINABLE_ONLY_RESUME_STATE = "_adaloop_trainable_only_resume_state"


def register_trainable_only_resume_state(accelerator: Any) -> None:
    """Keep frozen pretrained weights out of ``accelerator`` resume state.

    Registers Accelerate ``save_state``/``load_state`` pre-hooks once per
    accelerator; repeated calls are no-ops.  ``model.safetensors`` then holds
    exactly the trainable (``requires_grad``) parameters of each prepared
    model under their unwrapped names, shapes, and dtypes, instead of the
    complete frozen Qwen model.  Loading requires exactly that set, copies it
    in place, and skips Accelerate's strict full-model load, so a full-model
    file is rejected; the frozen weights come from rebuilding the model from
    the pinned base revision before loading.  Optimizer, scheduler, and RNG
    states keep their usual Accelerate files.  FSDP and DeepSpeed, which
    serialize their own model state, are left untouched.
    ``save_resume_checkpoint`` and ``load_resume_checkpoint`` call this, so
    only direct ``save_state``/``load_state`` callers need it.
    """

    if getattr(accelerator, _TRAINABLE_ONLY_RESUME_STATE, False):
        return

    def trainable(model: Any) -> dict[str, Any]:
        return {
            name: parameter
            for name, parameter in accelerator.unwrap_model(model).named_parameters()
            if parameter.requires_grad
        }

    def save_trainable(models: list[Any], weights: list[Any], _output_dir: str) -> None:
        # FSDP/DeepSpeed serialize their own model state and pass no weights.
        for index in range(len(weights)):
            weights[index] = {
                name: parameter.detach()
                for name, parameter in trainable(models[index]).items()
            }

    def load_trainable(models: list[Any], input_dir: str) -> None:
        from safetensors.torch import load_file

        torch = _torch()
        for index, model in enumerate(models):
            suffix = f"_{index}" if index else ""
            stored = load_file(str(Path(input_dir) / f"model{suffix}.safetensors"))
            expected = trainable(model)
            missing = sorted(set(expected) - set(stored))
            unexpected = sorted(set(stored) - set(expected))
            if missing or unexpected:
                raise RuntimeError(
                    "Resume weights must contain exactly the trainable "
                    f"parameters: {len(missing)} missing (e.g. {missing[:3]}), "
                    f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})"
                )
            for name, parameter in expected.items():
                value = stored[name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise RuntimeError(
                        f"Resume weight {name} is {tuple(value.shape)} "
                        f"{value.dtype}; expected {tuple(parameter.shape)} "
                        f"{parameter.dtype}"
                    )
            with torch.no_grad():
                for name, parameter in expected.items():
                    parameter.copy_(stored[name])
        # Every model is restored above; skip Accelerate's strict full load.
        models.clear()

    accelerator.register_save_state_pre_hook(save_trainable)
    accelerator.register_load_state_pre_hook(load_trainable)
    setattr(accelerator, _TRAINABLE_ONLY_RESUME_STATE, True)


def save_resume_checkpoint(
    accelerator: Any,
    training_module: Any,
    checkpoint_dir: str | Path,
    state: TrainingState,
    *,
    config: Any | None = None,
    run_contract: Mapping[str, Any] | None = None,
) -> Path:
    """Save a trainable-only Accelerate resume snapshot and its metadata.

    ``model.safetensors`` holds only the trainable tensors (see
    ``register_trainable_only_resume_state``); optimizer, scheduler, and RNG
    states use the usual Accelerate files.  The main process then adds the
    portable compact trainables, the recipe and run contract when given, and
    finally ``training_state.json``, the completion marker.
    """

    import yaml

    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    register_trainable_only_resume_state(accelerator)
    accelerator.save_state(str(root))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(training_module)
        save_compact_trainables(unwrapped, root)
        if config is not None:
            temporary = (root / TRAINING_CONFIG).with_suffix(".yaml.tmp")
            temporary.write_text(
                yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
            )
            temporary.replace(root / TRAINING_CONFIG)
        if run_contract is not None:
            _atomic_json(
                root / TRAINING_RUN_CONTRACT,
                _canonical_json_mapping(
                    run_contract, context="training run contract"
                ),
            )
        # This file is the completion marker used by ``--resume ... latest``;
        # write it only after every other required metadata artifact exists.
        _atomic_json(root / TRAINING_STATE, state.to_dict())
    accelerator.wait_for_everyone()
    return root


def load_resume_checkpoint(
    accelerator: Any,
    checkpoint_dir: str | Path,
    *,
    expected_config: Any | None = None,
    expected_run_contract: Mapping[str, Any] | None = None,
) -> TrainingState:
    """Validate the full recipe/run contract, then restore the resume state.

    The prepared model must already hold the frozen base weights (rebuilt from
    the pinned base revision) and the same trainable parameters as the saved
    run: only those trainable tensors, plus optimizer, scheduler, and RNG
    states, are restored from the trainable-only snapshot written by
    ``save_resume_checkpoint``.
    """

    import yaml

    root = Path(checkpoint_dir)
    state = TrainingState.from_file(root / TRAINING_STATE)
    if expected_config is not None:
        config_path = root / TRAINING_CONFIG
        try:
            stored_raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            stored = _canonical_json_mapping(
                stored_raw, context="stored training recipe"
            )
        except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not validate resume recipe {config_path}: {exc}"
            ) from exc
        expected = _recipe_payload(expected_config)
        if stored != expected:
            raise RuntimeError(
                "Resume checkpoint recipe does not exactly match the requested "
                "config; differing fields: "
                + _mapping_differences(stored, expected)
            )
    if expected_run_contract is not None:
        contract_path = root / TRAINING_RUN_CONTRACT
        try:
            stored_raw = json.loads(contract_path.read_text(encoding="utf-8"))
            stored = _canonical_json_mapping(
                stored_raw, context="stored training run contract"
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not validate resume run contract {contract_path}: {exc}"
            ) from exc
        expected = _canonical_json_mapping(
            expected_run_contract, context="requested training run contract"
        )
        if stored != expected:
            raise RuntimeError(
                "Resume checkpoint run contract does not exactly match the "
                "requested runtime; differing fields: "
                + _mapping_differences(stored, expected)
            )
    register_trainable_only_resume_state(accelerator)
    accelerator.load_state(str(root))
    accelerator.wait_for_everyone()
    return state


__all__ = [
    "NON_LORA_MANIFEST",
    "TRAINING_CONFIG",
    "TRAINING_RUN_CONTRACT",
    "TRAINING_STATE",
    "TrainingObjectiveHooks",
    "TrainabilityReport",
    "TrainingObjective",
    "TrainingState",
    "apply_language_lora",
    "build_adamw",
    "build_cosine_scheduler",
    "build_optimizer_groups",
    "compute_training_objective",
    "configure_trainable_parameters",
    "geometric_depth_weight_tensor",
    "load_resume_checkpoint",
    "make_training_objective_module",
    "register_trainable_only_resume_state",
    "release_readout_objective_hooks",
    "sample_geometric_depths",
    "save_compact_trainables",
    "save_resume_checkpoint",
]
