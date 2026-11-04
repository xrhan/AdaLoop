"""JAX/Flax training primitives for the released PaliGemma AdaLoop recipes.

The code in this module is intentionally independent of cluster launchers and
dataset layouts.  It owns the paper objective, trainable-parameter mask,
optimizer, one data-parallel update, and a complete resumable checkpoint.

Public *inference* checkpoints are compact overlays (see :mod:`checkpoint`).
Training snapshots are different by design: they also preserve Adam state,
PRNG state, epoch, and batch position so a restarted run does not silently
change its data stream.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .checkpoint import _sha256


PARAMS_FILE = "params.msgpack"
OPT_STATE_FILE = "opt_state.msgpack"
TRAINING_STATE_FILE = "training_state.json"
TRAINING_CONFIG_FILE = "training_config.yaml"
RUN_CONTRACT_FILE = "training_run_contract.json"
COMPLETE_FILE = "COMPLETE"


def _jax_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import flax.serialization
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:  # pragma: no cover - optional runtime extra
        raise RuntimeError(
            "PaliGemma training requires JAX, Flax, and Optax; install "
            "adaloop-paligemma[runtime]"
        ) from exc
    return jax, jnp, optax, flax.serialization


@dataclass(frozen=True)
class TrainingState:
    """Host-serializable cursor for deterministic restart."""

    global_step: int
    epoch: int
    batches_consumed_in_epoch: int
    rng_key: tuple[int, int]

    def __post_init__(self) -> None:
        for name, value in (
            ("global_step", self.global_step),
            ("epoch", self.epoch),
            ("batches_consumed_in_epoch", self.batches_consumed_in_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.rng_key, tuple)
            or len(self.rng_key) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in self.rng_key
            )
        ):
            raise ValueError("rng_key must contain two non-negative integers")

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingState":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read training state {path}: {exc}") from exc
        if not isinstance(raw, Mapping) or set(raw) != {
            "global_step",
            "epoch",
            "batches_consumed_in_epoch",
            "rng_key",
        }:
            raise ValueError("Training state has unknown or missing fields")
        return cls(
            global_step=raw["global_step"],
            epoch=raw["epoch"],
            batches_consumed_in_epoch=raw["batches_consumed_in_epoch"],
            rng_key=tuple(raw["rng_key"]),
        )


def _flatten_with_paths(tree: Any) -> list[tuple[tuple[str, ...], Any]]:
    try:
        from flax.traverse_util import flatten_dict
    except ImportError as exc:  # pragma: no cover - optional runtime extra
        raise RuntimeError("Parameter labeling requires Flax") from exc
    return [
        (tuple(str(part) for part in path), value)
        for path, value in flatten_dict(tree).items()
    ]


def parameter_category(path: tuple[str, ...]) -> str:
    """Classify one Flax parameter leaf for optimization.

    Released runs train Gemma LoRA, the SigLIP ``img/head`` connector, and the
    question pooler/AdaLN generators.  The pretrained vision blocks, token
    embeddings, and base Gemma weights remain frozen.
    """

    lowered = tuple(part.lower() for part in path)
    joined = "/".join(lowered)
    if any("lora" in part for part in lowered):
        return "lora"
    if len(lowered) >= 2 and lowered[0] == "img" and lowered[1] == "head":
        return "connector"
    if lowered and lowered[0] == "question_conditioner":
        return "conditioner"
    if "adaln" in lowered or "/adaln/" in f"/{joined}/":
        return "conditioner"
    return "frozen"


def build_parameter_labels(params: Any, *, expect_conditioner: bool) -> Any:
    """Return a pytree of optimizer labels and fail on missing trainables."""

    try:
        from flax.core import FrozenDict, freeze
        from flax.traverse_util import flatten_dict, unflatten_dict
    except ImportError as exc:  # pragma: no cover - optional runtime extra
        raise RuntimeError("Parameter labeling requires Flax") from exc

    flat = flatten_dict(params)
    labels = {path: parameter_category(tuple(str(p) for p in path)) for path in flat}
    counts = {
        name: sum(
            int(np.prod(np.shape(flat[path])))
            for path, category in labels.items()
            if category == name
        )
        for name in ("lora", "connector", "conditioner", "frozen")
    }
    if counts["lora"] == 0:
        raise ValueError("Parameter tree contains no Gemma LoRA leaves")
    if counts["connector"] == 0:
        raise ValueError("Parameter tree contains no img/head connector leaves")
    if expect_conditioner and counts["conditioner"] == 0:
        raise ValueError("Conditioned recipe contains no question/AdaLN leaves")
    if not expect_conditioner and counts["conditioner"]:
        raise ValueError("Unconditioned SFT tree unexpectedly contains AdaLN parameters")
    rebuilt = unflatten_dict(labels)
    return freeze(rebuilt) if isinstance(params, FrozenDict) else rebuilt


def trainable_parameter_counts(params: Any) -> dict[str, int]:
    """Report exact scalar counts by release optimizer category."""

    totals = {name: 0 for name in ("lora", "connector", "conditioner", "frozen")}
    for path, value in _flatten_with_paths(params):
        totals[parameter_category(path)] += int(np.prod(np.shape(value)))
    totals["trainable"] = totals["lora"] + totals["connector"] + totals["conditioner"]
    totals["total"] = totals["trainable"] + totals["frozen"]
    return totals


def cosine_schedule(
    peak_value: float,
    *,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
):
    """Warmup followed by cosine decay to ``peak_value * min_ratio``."""

    _, _, optax, _ = _jax_stack()
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("Require total_steps >= 1 and 0 <= warmup_steps < total_steps")
    if not math.isfinite(min_ratio) or not 0 <= min_ratio <= 1:
        raise ValueError("min_ratio must be finite and in [0, 1]")
    peak = float(peak_value)
    if peak <= 0 or not math.isfinite(peak):
        raise ValueError("peak_value must be finite and positive")
    if warmup_steps:
        warmup = optax.linear_schedule(0.0, peak, warmup_steps)
        decay = optax.cosine_decay_schedule(
            peak,
            decay_steps=total_steps - warmup_steps,
            alpha=float(min_ratio),
        )
        return optax.join_schedules([warmup, decay], [warmup_steps])
    return optax.cosine_decay_schedule(
        peak, decay_steps=total_steps, alpha=float(min_ratio)
    )


def build_optimizer(params: Any, recipe: Any) -> tuple[Any, Any]:
    """Build the paper AdamW optimizer and its exhaustive parameter labels."""

    _, _, optax, _ = _jax_stack()
    config = recipe.training
    optimizer = config.optimizer
    labels = build_parameter_labels(
        params, expect_conditioner=recipe.conditioner.mode != "none"
    )

    def adamw(learning_rate: float):
        schedule = cosine_schedule(
            learning_rate,
            total_steps=int(config.steps),
            warmup_steps=int(config.warmup_steps),
            min_ratio=float(config.min_lr_ratio),
        )
        return optax.adamw(
            schedule,
            b1=float(optimizer.beta1),
            b2=float(optimizer.beta2),
            eps=float(optimizer.epsilon),
            weight_decay=float(optimizer.weight_decay),
        )

    transforms = {
        "lora": adamw(float(optimizer.lora_lr)),
        "connector": adamw(float(optimizer.connector_lr)),
        "conditioner": (
            adamw(float(optimizer.conditioner_lr))
            if recipe.conditioner.mode != "none"
            else optax.set_to_zero()
        ),
        "frozen": optax.set_to_zero(),
    }
    tx = optax.chain(
        optax.clip_by_global_norm(float(optimizer.max_grad_norm)),
        optax.multi_transform(transforms, labels),
    )
    return tx, labels


def _stop_gradient_on_frozen(params: Any, labels: Any) -> Any:
    jax, _, _, _ = _jax_stack()
    return jax.tree.map(
        lambda value, label: (
            jax.lax.stop_gradient(value) if label == "frozen" else value
        ),
        params,
        labels,
    )


def _condition_kwargs(batch: Mapping[str, Any]) -> dict[str, Any]:
    has_ids = "condition_input_ids" in batch
    has_mask = "condition_attention_mask" in batch
    if has_ids != has_mask:
        raise ValueError(
            "condition_input_ids and condition_attention_mask must appear together"
        )
    if not has_ids:
        return {}
    return {
        "condition_input_ids": batch["condition_input_ids"],
        "condition_attention_mask": batch["condition_attention_mask"],
    }


def build_train_step(model: Any, tx: Any, labels: Any, recipe: Any):
    """Create one pmappable AdaLoop update with the paper scalar objective.

    The model owns readout-specific execution (including detached Variant-C
    drafts) and returns the already-normalized scalar objective.  This mirrors
    the trained implementation while keeping optimizer semantics in one place.
    """

    jax, _, optax, _ = _jax_stack()
    train_depth = int(recipe.train_depth)
    variant_c = recipe.variant == "c"
    if variant_c:
        from .model.trajectory import rollout_reprefill_with_apply

    def loss_fn(params, batch, rng):
        apply_params = _stop_gradient_on_frozen(params, labels)
        condition_kwargs = _condition_kwargs(batch)
        rollout_kwargs: dict[str, Any] = {}
        if variant_c:
            # Draft choice is a discrete interface.  As in the paper runs,
            # generate it with a stop-gradient parameter view,
            # one fresh prefill per visual round, and cached token extensions
            # within that round.  The supervised terminal forward below still
            # receives gradients through every selected depth.
            rollout_params = jax.tree.map(jax.lax.stop_gradient, apply_params)
            text_input = batch["text"][:, :-1]
            draft_ids, draft_mask = rollout_reprefill_with_apply(
                model,
                rollout_params,
                image=batch["image"],
                text=text_input,
                mask_ar=batch["mask_ar"][:, :-1],
                input_mask=text_input != int(model.pad_token_id),
                condition_input_ids=condition_kwargs["condition_input_ids"],
                condition_attention_mask=condition_kwargs[
                    "condition_attention_mask"
                ],
                steps=train_depth,
                draft_max_tokens=int(model.draft_max_tokens),
                eos_token_id=int(model.eos_token_id),
                suppressed_token_ids=tuple(model.suppressed_draft_token_ids),
            )
            rollout_kwargs = {"drafts": draft_ids, "draft_mask": draft_mask}
        output = model.apply(
            {"params": apply_params},
            batch["image"],
            batch["text"][:, :-1],
            batch["mask_ar"][:, :-1],
            batch["text"][:, 1:],
            batch["mask_loss"][:, 1:],
            train=True,
            n_steps=train_depth,
            **condition_kwargs,
            **rollout_kwargs,
            rngs={"dropout": rng},
        )
        if output.loss is None or getattr(output.loss, "ndim", None) != 0:
            raise ValueError("Training model must return one scalar AdaLoop loss")
        return output.loss

    def step(params, opt_state, batch, rng):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch, rng)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")
        # optax.global_norm is deprecated in favor of this identical L2 norm.
        grad_norm = optax.tree.norm(grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        update_norm = optax.tree.norm(updates)
        params = optax.apply_updates(params, updates)
        parameter_norm = optax.tree.norm(params)
        return params, opt_state, {
            "loss": loss,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "parameter_norm": parameter_norm,
            "relative_update_norm": update_norm / (parameter_norm + 1e-12),
        }

    return step


def replicate_to_local_devices(tree: Any) -> Any:
    """Give every local device one copy of each leaf, stacked for ``pmap``.

    This replaces ``jax.device_put_replicated``, which JAX 0.10 removed.  The
    stacked axis uses the layout ``pmap`` itself produces: a NamedSharding over
    ``jax.local_devices()`` for the shard_map-based ``pmap`` (JAX >= 0.8),
    which the classic ``pmap`` maps to the same per-device slices.  Neither
    implementation therefore copies the replicated (and donated) state again.
    As before, each leaf crosses from the host once and is then copied between
    devices; no stacked host copy of the frozen base is materialized.
    """

    jax, _, _, _ = _jax_stack()
    devices = jax.local_devices()
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray(devices), ("devices",)),
        jax.sharding.PartitionSpec("devices"),
    )

    def replicate(value):
        first = jax.device_put(np.asarray(value)[None], devices[0])
        return jax.make_array_from_single_device_arrays(
            (len(devices),) + first.shape[1:],
            sharding,
            [first] + [jax.device_put(first, device) for device in devices[1:]],
        )

    return jax.tree.map(replicate, tree)


def unreplicate_from_local_devices(tree: Any) -> Any:
    """Copy the first local replica of a ``pmap``-stacked tree to the host.

    ``value[0]`` would gather every replica of a leaf-sharded array to all
    devices before slicing under the shard_map-based ``pmap``; one addressable
    shard is exactly one replica on every JAX version.
    """

    jax, _, _, _ = _jax_stack()
    shards = jax.device_get(
        jax.tree.map(lambda value: value.addressable_shards[0].data, tree)
    )
    return jax.tree.map(
        lambda value, shard: np.asarray(shard).reshape(value.shape[1:]), tree, shards
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_training_checkpoint(
    directory: str | Path,
    *,
    params: Any,
    opt_state: Any,
    state: TrainingState,
    config_yaml: str,
    run_contract: Mapping[str, Any],
) -> Path:
    """Atomically write a complete local training snapshot."""

    _, _, _, serialization = _jax_stack()
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    complete = root / COMPLETE_FILE
    if complete.exists():
        complete.unlink()
    _atomic_bytes(root / PARAMS_FILE, serialization.to_bytes(params))
    _atomic_bytes(root / OPT_STATE_FILE, serialization.to_bytes(opt_state))
    _atomic_bytes(root / TRAINING_CONFIG_FILE, config_yaml.encode("utf-8"))
    _atomic_bytes(
        root / TRAINING_STATE_FILE,
        (json.dumps(asdict(state), sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    contract = dict(run_contract)
    contract["checkpoint_sha256"] = {
        PARAMS_FILE: _sha256(root / PARAMS_FILE),
        OPT_STATE_FILE: _sha256(root / OPT_STATE_FILE),
        TRAINING_CONFIG_FILE: _sha256(root / TRAINING_CONFIG_FILE),
        TRAINING_STATE_FILE: _sha256(root / TRAINING_STATE_FILE),
    }
    _atomic_bytes(
        root / RUN_CONTRACT_FILE,
        (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    _atomic_bytes(complete, b"complete\n")
    return root


def load_training_checkpoint(
    directory: str | Path,
    *,
    params_template: Any,
    opt_state_template: Any,
    expected_config_yaml: str,
    expected_run_contract: Mapping[str, Any],
) -> tuple[Any, Any, TrainingState]:
    """Restore a complete snapshot after strict recipe/data verification."""

    _, _, _, serialization = _jax_stack()
    root = Path(directory).expanduser().resolve()
    required = {
        PARAMS_FILE,
        OPT_STATE_FILE,
        TRAINING_STATE_FILE,
        TRAINING_CONFIG_FILE,
        RUN_CONTRACT_FILE,
        COMPLETE_FILE,
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise ValueError(f"Incomplete training checkpoint {root}: missing {missing}")
    actual_config = (root / TRAINING_CONFIG_FILE).read_text(encoding="utf-8")
    if actual_config != expected_config_yaml:
        raise ValueError("Resume config differs from the serialized training recipe")
    try:
        contract = json.loads((root / RUN_CONTRACT_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid training run contract: {exc}") from exc
    hashes = contract.pop("checkpoint_sha256", None)
    if contract != dict(expected_run_contract):
        raise ValueError("Resume data/runtime contract differs from the original run")
    if not isinstance(hashes, Mapping):
        raise ValueError("Training run contract contains no checkpoint hashes")
    for name in (
        PARAMS_FILE,
        OPT_STATE_FILE,
        TRAINING_CONFIG_FILE,
        TRAINING_STATE_FILE,
    ):
        if hashes.get(name) != _sha256(root / name):
            raise ValueError(f"Training checkpoint hash mismatch for {name}")
    params = serialization.from_bytes(
        params_template, (root / PARAMS_FILE).read_bytes()
    )
    opt_state = serialization.from_bytes(
        opt_state_template, (root / OPT_STATE_FILE).read_bytes()
    )
    return params, opt_state, TrainingState.from_file(root / TRAINING_STATE_FILE)


__all__ = [
    "COMPLETE_FILE",
    "OPT_STATE_FILE",
    "PARAMS_FILE",
    "RUN_CONTRACT_FILE",
    "TRAINING_CONFIG_FILE",
    "TRAINING_STATE_FILE",
    "TrainingState",
    "build_optimizer",
    "build_parameter_labels",
    "build_train_step",
    "cosine_schedule",
    "load_training_checkpoint",
    "parameter_category",
    "replicate_to_local_devices",
    "save_training_checkpoint",
    "trainable_parameter_counts",
    "unreplicate_from_local_devices",
]
