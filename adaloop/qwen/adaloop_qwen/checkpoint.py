"""Strict, portable Qwen2.5-VL AdaLoop checkpoint loading and export.

The canonical release format keeps PEFT language adapters separate from the
trainable vision-language merger and AdaLN/question-pooler sidecar. A hashed
recipe is the single source of truth for topology; the manifest
(:mod:`adaloop_qwen.manifest`) repeats only the fields needed to reject
incompatible files before importing PyTorch.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from .config import IMMUTABLE_REVISION, QwenAdaLoopConfig
from .errors import CheckpointValidationError
from .manifest import (
    CANONICAL_ADAPTER_CONFIG,
    CANONICAL_ADAPTER_WEIGHTS,
    CANONICAL_CONFIG,
    CANONICAL_NON_LORA_WEIGHTS,
    CANONICAL_PROCESSOR_DIR,
    CANONICAL_RUNTIME_CONTRACT,
    MANIFEST_FILE,
    CheckpointManifest,
    QwenRuntimeContract,
    write_manifest,
)

# PEFT saves each adapter tensor under its wrapped module name without the
# adapter name. Its loader skips some other spellings (e.g. with the adapter
# name, or without the "model." segment) with only a warning, leaving those
# LoRA weights at their initialization, so no other spelling is accepted.
_LORA_WEIGHT_KEY = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\.(?P<layer>\d+)\."
    r"self_attn\.(?P<target>[A-Za-z0-9_]+)\.lora_(?P<side>[AB])\.weight$"
)
_PEFT_ADAPTER_NAME = "default"
_SIDECAR_MARKERS = (".merger.", "question_conditioner.", ".adaln.")
# PEFT LoRA options that change which modules are adapted, which tensors are
# loaded, or how loaded adapters act (some init_lora_weights values rewrite
# the base weights). Training keeps PEFT's defaults, which absent keys
# (older PEFT versions) also take.
_DEFAULT_LORA_OPTIONS: dict[str, Any] = {
    "alpha_pattern": {},
    "exclude_modules": None,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "lora_bias": False,
    "modules_to_save": None,
    "rank_pattern": {},
    "target_parameters": None,
    "trainable_token_indices": None,
    "use_dora": False,
    "use_rslora": False,
}


def _expected_lora_target_regex(recipe: QwenAdaLoopConfig) -> str:
    alternatives = "|".join(recipe.lora.target_modules)
    return (
        r".*language_model\.layers\.\d+\.self_attn\."
        rf"({alternatives})"
    )


def _equivalent_base_model_identity(value: Any, recipe: QwenAdaLoopConfig) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    expected_name = recipe.base_model.rstrip("/").split("/")[-1]
    normalized = value.strip().replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] == expected_name


def _validate_adapter_config(
    root: Path,
    recipe: QwenAdaLoopConfig,
    *,
    allow_equivalent_base: bool = False,
) -> None:
    """Reject PEFT metadata that contradicts the signed AdaLoop recipe."""

    path = root / CANONICAL_ADAPTER_CONFIG
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            f"could not read PEFT adapter config {path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise CheckpointValidationError("adapter_config.json must contain an object")

    expected_scalars = {
        "r": recipe.lora.rank,
        "lora_alpha": recipe.lora.alpha,
        "lora_dropout": recipe.lora.dropout,
        "bias": "none",
    }
    mismatches = [
        f"{name}: adapter={raw.get(name)!r}, recipe={expected!r}"
        for name, expected in expected_scalars.items()
        if raw.get(name) != expected
    ]
    adapter_base = raw.get("base_model_name_or_path")
    if adapter_base != recipe.base_model and not (
        allow_equivalent_base
        and _equivalent_base_model_identity(adapter_base, recipe)
    ):
        mismatches.append(
            "base_model_name_or_path: "
            f"adapter={adapter_base!r}, recipe={recipe.base_model!r}"
        )
    task_type = raw.get("task_type")
    if task_type != "CAUSAL_LM":
        mismatches.append(
            f"task_type: adapter={task_type!r}, recipe='CAUSAL_LM'"
        )

    targets = raw.get("target_modules")
    expected_targets = set(recipe.lora.target_modules)
    if isinstance(targets, str):
        if targets != _expected_lora_target_regex(recipe):
            mismatches.append(
                "target_modules regex does not match the released language-only "
                "q/k/v/o target"
            )
    elif isinstance(targets, (list, tuple, set)):
        if {str(item) for item in targets} != expected_targets:
            mismatches.append(
                f"target_modules: adapter={targets!r}, "
                f"recipe={sorted(expected_targets)!r}"
            )
    else:
        mismatches.append("target_modules is missing or has an invalid type")
    mismatches.extend(
        f"{name}: adapter={raw[name]!r}, PEFT default={default!r}"
        for name, default in _DEFAULT_LORA_OPTIONS.items()
        if name in raw and raw[name] != default
    )

    if mismatches:
        raise CheckpointValidationError(
            "PEFT adapter config contradicts adaloop_config.yaml: "
            + "; ".join(mismatches)
        )


def _canonicalize_adapter_config(
    root: Path, recipe: QwenAdaLoopConfig
) -> None:
    """Validate a local/cache base identity, then remove its path from export."""

    _validate_adapter_config(root, recipe, allow_equivalent_base=True)
    path = root / CANONICAL_ADAPTER_CONFIG
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            f"could not canonicalize PEFT adapter config {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise CheckpointValidationError("adapter_config.json must contain an object")
    raw["base_model_name_or_path"] = recipe.base_model
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    _validate_adapter_config(root, recipe)


def _validate_adapter_weight_shapes(
    shapes: Mapping[str, tuple[int, ...]],
    recipe: QwenAdaLoopConfig,
    *,
    expected_num_layers: int | None = None,
    expected: Mapping[str, tuple[int, ...]] | None = None,
) -> None:
    """Validate exact language-attention LoRA coverage from tensor metadata.

    ``expected`` is the exact adapter inventory of the model that loads the
    file (see ``_model_tensor_inventory``); every name and shape must match.
    """

    if not shapes:
        raise CheckpointValidationError("PEFT adapter weights are empty")
    if expected_num_layers is not None and (
        isinstance(expected_num_layers, bool)
        or not isinstance(expected_num_layers, int)
        or expected_num_layers < 1
    ):
        raise ValueError("expected_num_layers must be a positive integer or None")

    targets = tuple(recipe.lora.target_modules)
    target_set = set(targets)
    observed: dict[tuple[int, str, str], str] = {}
    invalid: list[str] = []
    duplicates: list[str] = []
    for name, shape in shapes.items():
        match = _LORA_WEIGHT_KEY.fullmatch(name)
        if match is None:
            invalid.append(name)
            continue
        layer_text = match.group("layer")
        layer = int(layer_text)
        target = match.group("target")
        side = match.group("side")
        if str(layer) != layer_text or target not in target_set:
            invalid.append(name)
            continue
        if len(shape) != 2 or any(int(size) < 1 for size in shape):
            invalid.append(name)
            continue
        rank_axis = 0 if side == "A" else 1
        if int(shape[rank_axis]) != recipe.lora.rank:
            invalid.append(name)
            continue
        slot = (layer, target, side)
        previous = observed.get(slot)
        if previous is not None:
            duplicates.extend((previous, name))
            continue
        observed[slot] = name

    observed_layers = {layer for layer, _, _ in observed}
    if expected_num_layers is None:
        expected_layers = (
            set(range(max(observed_layers) + 1)) if observed_layers else set()
        )
    else:
        expected_layers = set(range(expected_num_layers))
    missing_layers = expected_layers - observed_layers
    unexpected_layers = observed_layers - expected_layers
    expected_slots = {
        (layer, target, side)
        for layer in expected_layers
        for target in targets
        for side in ("A", "B")
    }
    missing_slots = expected_slots - set(observed)
    unexpected_slots = set(observed) - expected_slots
    if (
        invalid
        or duplicates
        or missing_layers
        or unexpected_layers
        or missing_slots
        or unexpected_slots
    ):
        describe = lambda slot: f"layer={slot[0]}/{slot[1]}/lora_{slot[2]}"  # noqa: E731
        raise CheckpointValidationError(
            "PEFT adapter tensors contradict the released LoRA topology: "
            f"invalid={sorted(set(invalid))}, "
            f"duplicates={sorted(set(duplicates))}, "
            f"missing_layers={sorted(missing_layers)}, "
            f"unexpected_layers={sorted(unexpected_layers)}, "
            f"missing={sorted(describe(slot) for slot in missing_slots)}, "
            f"unexpected={sorted(describe(slot) for slot in unexpected_slots)}"
        )
    if expected is not None:
        _validate_tensor_inventory(shapes, expected, context="PEFT adapter weights")


def _validate_adapter_weights(
    root: Path,
    recipe: QwenAdaLoopConfig,
    *,
    expected_num_layers: int | None = None,
    expected: Mapping[str, tuple[int, ...]] | None = None,
) -> None:
    """Inspect safetensors and validate exact language-attention LoRA coverage."""

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "Validating adapter weights requires safetensors"
        ) from exc
    path = root / CANONICAL_ADAPTER_WEIGHTS
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            shapes = {
                name: tuple(int(size) for size in handle.get_slice(name).get_shape())
                for name in handle.keys()
            }
    except Exception as exc:
        raise CheckpointValidationError(
            f"could not inspect PEFT adapter weights {path}: {exc}"
        ) from exc
    _validate_adapter_weight_shapes(
        shapes, recipe, expected_num_layers=expected_num_layers, expected=expected
    )


def _sidecar_parameter_roles(name: str) -> frozenset[str]:
    roles: set[str] = set()
    if ".merger." in name:
        roles.add("merger")
    if "question_conditioner." in name:
        roles.add("question_conditioner")
    if ".adaln." in name:
        roles.add("adaln")
    return frozenset(roles)


def _validate_sidecar_tensors(
    sidecar: Mapping[str, Any], recipe: QwenAdaLoopConfig
) -> None:
    """Validate sidecar tensor roles without constructing a 3B model."""

    if not sidecar:
        raise CheckpointValidationError("non-LoRA sidecar is empty")
    observed_roles: set[str] = set()
    invalid: list[str] = []
    for name, tensor in sidecar.items():
        roles = _sidecar_parameter_roles(str(name))
        if len(roles) != 1 or "lora_" in str(name).lower():
            invalid.append(str(name))
            continue
        shape = getattr(tensor, "shape", None)
        if shape is None or any(int(size) < 1 for size in shape):
            invalid.append(str(name))
            continue
        observed_roles.update(roles)
    if invalid:
        raise CheckpointValidationError(
            "non-LoRA sidecar contains parameters outside the released "
            f"merger/AdaLN/question-pooler topology: {sorted(invalid)}"
        )
    required = {"merger"}
    forbidden: set[str] = set()
    if recipe.conditioner.mode == "none":
        forbidden.update({"question_conditioner", "adaln"})
    else:
        required.update({"question_conditioner", "adaln"})
    missing = required - observed_roles
    unexpected = forbidden & observed_roles
    if missing or unexpected:
        raise CheckpointValidationError(
            "non-LoRA sidecar roles contradict adaloop_config.yaml: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


@dataclass(frozen=True)
class _TensorInventory:
    """Exact tensor names and shapes of the two released weight files.

    ``text_layers`` is the base model's text layer count: LoRA must cover
    every text layer, whatever ``adapter_config.json`` selects.
    """

    adapter: Mapping[str, tuple[int, ...]]
    sidecar: Mapping[str, tuple[int, ...]]
    text_layers: int


def _tensor_shapes(tensors: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        str(name): tuple(int(size) for size in tensor.shape)
        for name, tensor in tensors.items()
    }


def _model_tensor_inventory(model: Any) -> _TensorInventory:
    """Return the exact adapter and sidecar tensors a PEFT-wrapped model loads.

    Export, conversion, and loading all validate against this one inventory,
    so a checkpoint is certified with exactly the tensors the loader accepts.
    Adapter names are the ones PEFT saves (the adapter name removed); sidecar
    names are the wrapped merger, question-pooler, and AdaLN parameters.
    """

    adapter: dict[str, tuple[int, ...]] = {}
    sidecar: dict[str, tuple[int, ...]] = {}
    for name, parameter in model.named_parameters():
        shape = tuple(int(size) for size in parameter.shape)
        if "lora_" in name.lower():
            adapter[name.replace(f".{_PEFT_ADAPTER_NAME}", "")] = shape
        elif any(marker in name for marker in _SIDECAR_MARKERS):
            sidecar[name] = shape
    return _TensorInventory(
        adapter=adapter, sidecar=sidecar, text_layers=_model_text_layer_count(model)
    )


def _name_preview(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(repr(name) for name in names[:limit])
    if len(names) > limit:
        shown += f", ... ({len(names)} in total)"
    return f"[{shown}]"


def _validate_tensor_inventory(
    shapes: Mapping[str, tuple[int, ...]],
    expected: Mapping[str, tuple[int, ...]],
    *,
    context: str,
) -> None:
    """Refuse missing, unexpected, and mis-shaped tensors."""

    missing = sorted(set(expected) - set(shapes))
    unexpected = sorted(set(shapes) - set(expected))
    mismatched = sorted(
        f"{name}: {tuple(shapes[name])} instead of {tuple(expected[name])}"
        for name in set(shapes) & set(expected)
        if tuple(shapes[name]) != tuple(expected[name])
    )
    if missing or unexpected or mismatched:
        raise CheckpointValidationError(
            f"{context} do not exactly match the released trainable topology: "
            f"missing={_name_preview(missing)}, "
            f"unexpected={_name_preview(unexpected)}, "
            f"mis-shaped={_name_preview(mismatched)}"
        )


def _text_layer_count(config: Any) -> int:
    text_config = getattr(config, "text_config", config)
    value = getattr(text_config, "num_hidden_layers", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CheckpointValidationError(
            "pinned base config does not expose a positive text num_hidden_layers"
        )
    return int(value)


def _model_text_layer_count(model: Any) -> int:
    pending = [model]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        config = getattr(candidate, "config", None)
        if config is not None:
            try:
                return _text_layer_count(config)
            except CheckpointValidationError:
                pass
        getter = getattr(candidate, "get_base_model", None)
        if callable(getter):
            try:
                pending.append(getter())
            except TypeError:
                pass
        for name in ("base_model", "model", "module"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
    raise CheckpointValidationError(
        "live model does not expose the base text num_hidden_layers needed "
        "to validate exact adapter coverage"
    )


def _directory_is_nonempty(path: Path) -> bool:
    return next(path.iterdir(), None) is not None


def _validate_checkpoint_destination(root: Path, *, overwrite: bool) -> bool:
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    if root.is_symlink():
        raise CheckpointValidationError(
            f"checkpoint destination must not be a symlink: {root}"
        )
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"checkpoint destination is not a directory: {root}")
    nonempty = root.is_dir() and _directory_is_nonempty(root)
    if nonempty and not overwrite:
        raise FileExistsError(
            f"checkpoint destination is not empty: {root}; pass overwrite=True "
            "only when replacing its canonical checkpoint files is intended"
        )
    return nonempty


def _backup_path(root: Path, label: str) -> Path:
    path = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.{label}-", dir=str(root.parent))
    )
    path.rmdir()
    return path


def _publish_into_nonempty_directory(stage: Path, root: Path) -> None:
    """Publish staged files transactionally, preserving unrelated run state."""

    staged_files = sorted(
        (path for path in stage.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(stage).as_posix(),
    )
    if any(path.is_symlink() for path in stage.rglob("*")):
        raise CheckpointValidationError("staged checkpoint must not contain symlinks")
    manifest = stage / MANIFEST_FILE
    if manifest not in staged_files:
        raise CheckpointValidationError(f"staged checkpoint is missing {MANIFEST_FILE}")
    staged_files.remove(manifest)
    staged_files.append(manifest)

    relative_files = [path.relative_to(stage) for path in staged_files]
    stale_files: list[Path] = []
    processor_root = root / CANONICAL_PROCESSOR_DIR
    if processor_root.is_dir():
        stale_files.extend(
            path.relative_to(root)
            for path in processor_root.rglob("*")
            if (path.is_file() or path.is_symlink())
            and path.relative_to(root) not in relative_files
        )
    stale_files = sorted(set(stale_files), key=lambda path: path.as_posix())
    owned_paths = list(dict.fromkeys((*relative_files, *stale_files)))
    backup_order = sorted(
        owned_paths,
        key=lambda path: (path != Path(MANIFEST_FILE), path.as_posix()),
    )
    for relative in owned_paths:
        destination = root / relative
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise CheckpointValidationError(
                f"cannot replace non-file checkpoint destination path: {destination}"
            )
        parent = destination.parent
        while parent != root:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise CheckpointValidationError(
                    f"checkpoint destination parent is unsafe: {parent}"
                )
            parent = parent.parent

    backup = _backup_path(root, "backup")
    backup.mkdir()
    backed_up: list[Path] = []
    published: list[Path] = []
    try:
        # Remove any old completion marker before touching its referenced
        # files. A concurrent reader therefore sees either the old complete
        # checkpoint or no checkpoint, never a mixed manifest/file set.
        for relative in backup_order:
            destination = root / relative
            if destination.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(saved)
                backed_up.append(relative)
        for source, relative in zip(staged_files, relative_files, strict=True):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            published.append(relative)
    except Exception:
        try:
            for relative in reversed(published):
                destination = root / relative
                if destination.is_file():
                    destination.unlink()
            for relative in reversed(backed_up):
                saved = backup / relative
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                saved.replace(destination)
        except Exception as rollback_error:
            raise RuntimeError(
                "checkpoint publication and rollback both failed; preserved "
                f"backup at {backup}"
            ) from rollback_error
        shutil.rmtree(backup, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


@contextmanager
def _staged_checkpoint_directory(
    output_dir: str | Path, *, overwrite: bool
) -> Iterator[tuple[Path, Path]]:
    """Yield ``(destination, staging_dir)`` and publish only on success."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise CheckpointValidationError(
            f"checkpoint destination must not be a symlink: {requested}"
        )
    root = requested.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    _validate_checkpoint_destination(root, overwrite=overwrite)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    try:
        yield root, stage
        nonempty = _validate_checkpoint_destination(root, overwrite=overwrite)
        if nonempty:
            _publish_into_nonempty_directory(stage, root)
        else:
            backup: Path | None = None
            if root.exists():
                backup = _backup_path(root, "empty")
                root.replace(backup)
            try:
                stage.replace(root)
            except Exception:
                if backup is not None and backup.exists() and not root.exists():
                    backup.replace(root)
                raise
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


@dataclass
class LoadedQwenCheckpoint:
    model: Any
    processor: Any
    config: QwenAdaLoopConfig
    runtime_contract: QwenRuntimeContract
    manifest: CheckpointManifest
    checkpoint_dir: Path


def _apply_processor_contract(
    processor: Any, runtime: QwenRuntimeContract
) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    image_processor = getattr(processor, "image_processor", None)
    if tokenizer is None or image_processor is None:
        raise CheckpointValidationError(
            "Qwen processor must expose tokenizer and image_processor"
        )
    tokenizer.model_max_length = runtime.max_context_length
    image_processor.min_pixels = runtime.min_pixels
    image_processor.max_pixels = runtime.max_pixels


def _write_processor_and_runtime(
    root: Path,
    processor: Any,
    recipe: QwenAdaLoopConfig,
) -> QwenRuntimeContract:
    runtime = QwenRuntimeContract.from_recipe(recipe)
    _apply_processor_contract(processor, runtime)
    processor_root = root / CANONICAL_PROCESSOR_DIR
    processor_root.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(processor_root)
    processor_files = [path for path in processor_root.rglob("*") if path.is_file()]
    if not processor_files:
        raise RuntimeError("Qwen processor export produced no files")
    (root / CANONICAL_RUNTIME_CONTRACT).write_text(
        json.dumps(runtime.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runtime


def _resolve_checkpoint(source: str | Path, *, revision: str | None = None) -> Path:
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise CheckpointValidationError(
            f"{source!r} is not a local directory; install huggingface_hub to load Hub repositories"
        ) from exc
    return Path(snapshot_download(repo_id=str(source), revision=revision)).resolve()


def _load_non_lora_sidecar(
    model: Any,
    path: Path,
    recipe: QwenAdaLoopConfig,
    expected: Mapping[str, tuple[int, ...]],
    *,
    is_trainable: bool = False,
) -> frozenset[str]:
    from safetensors.torch import load_file

    sidecar = load_file(str(path), device="cpu")
    _validate_sidecar_tensors(sidecar, recipe)
    _validate_tensor_inventory(
        _tensor_shapes(sidecar), expected, context="non-LoRA sidecar tensors"
    )
    incompatible = model.load_state_dict(sidecar, strict=False)
    if incompatible.unexpected_keys:
        raise CheckpointValidationError(
            f"unexpected sidecar keys: {sorted(incompatible.unexpected_keys)}"
        )
    if is_trainable:
        parameters = dict(model.named_parameters())
        for name in sidecar:
            parameters[name].requires_grad_(True)
    return frozenset(sidecar)


class QwenCheckpointLoader:
    """Load a canonical adapter-format AdaLoop checkpoint."""

    @staticmethod
    def inspect(checkpoint_dir: str | Path) -> CheckpointManifest:
        path = Path(checkpoint_dir) / MANIFEST_FILE
        if not path.is_file():
            raise CheckpointValidationError(f"no {MANIFEST_FILE} found in {checkpoint_dir}")
        return CheckpointManifest.from_file(path)

    @classmethod
    def from_pretrained(
        cls, checkpoint: str | Path, *, checkpoint_revision: str | None = None,
        device: str | Any | None = None, dtype: Any | None = None,
        attn_implementation: str | None = "sdpa",
        merge_lora: bool = True, is_trainable: bool = False,
        verify_hashes: bool = True,
        processor_source: str | Path | None = None,
        **model_kwargs: Any,
    ) -> LoadedQwenCheckpoint:
        root = _resolve_checkpoint(checkpoint, revision=checkpoint_revision)
        manifest = cls.inspect(root)
        manifest.validate_files(root, verify_hashes=verify_hashes)
        recipe = manifest.load_and_validate_config(root)
        runtime_contract = manifest.load_and_validate_runtime(root, recipe)
        _validate_adapter_config(root, recipe)
        _validate_adapter_weights(root, recipe)
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Loading Qwen checkpoints requires the adaloop-qwen[gpu] dependencies"
            ) from exc
        from .model.modeling import AdaptiveQwen2_5_VLForConditionalGeneration

        common = dict(
            revision=recipe.base_revision,
            torch_dtype=torch.bfloat16 if dtype is None else dtype,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
        if attn_implementation is not None:
            common["attn_implementation"] = attn_implementation
        if recipe.variant == "sft":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                recipe.base_model, **common
            )
        else:
            model = AdaptiveQwen2_5_VLForConditionalGeneration.from_base_pretrained(
                recipe.base_model, adaloop_config=recipe, **common
            )
        _validate_adapter_weights(
            root,
            recipe,
            expected_num_layers=_model_text_layer_count(model),
        )
        model = PeftModel.from_pretrained(model, str(root), is_trainable=is_trainable)
        # PEFT skips adapter keys it does not expect and adapts whatever
        # adapter_config.json selects; require the exact tensors it loaded.
        inventory = _model_tensor_inventory(model)
        _validate_adapter_weights(root, recipe, expected=inventory.adapter)
        sidecar_names = _load_non_lora_sidecar(
            model,
            root / CANONICAL_NON_LORA_WEIGHTS,
            recipe,
            inventory.sidecar,
            is_trainable=is_trainable,
        )
        if is_trainable:
            trainable_non_lora = {
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and "lora_" not in name.lower()
            }
            if trainable_non_lora != set(sidecar_names):
                raise CheckpointValidationError(
                    "trainable checkpoint load did not restore exactly the "
                    "non-LoRA sidecar parameters: "
                    f"missing={sorted(set(sidecar_names) - trainable_non_lora)}, "
                    f"unexpected={sorted(trainable_non_lora - set(sidecar_names))}"
                )
        if merge_lora:
            if is_trainable:
                raise ValueError("merge_lora=true is incompatible with is_trainable=true")
            model = model.merge_and_unload()
        model.eval() if not is_trainable else model.train()
        if device is not None:
            model.to(device)
        source = processor_source or (root / CANONICAL_PROCESSOR_DIR)
        processor = AutoProcessor.from_pretrained(
            source,
            revision=None if Path(str(source)).is_dir() else recipe.base_revision,
            use_fast=True,
        )
        _apply_processor_contract(processor, runtime_contract)
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"
        return LoadedQwenCheckpoint(
            model=model, processor=processor, config=recipe,
            runtime_contract=runtime_contract,
            manifest=manifest, checkpoint_dir=root,
        )


def export_release_checkpoint(
    model: Any, recipe: QwenAdaLoopConfig, output_dir: str | Path, *,
    processor: Any,
    overwrite: bool = False,
) -> Path:
    """Stage and publish PEFT plus trainable non-LoRA tensors safely.

    Non-empty destinations require ``overwrite=True``. In that explicit mode,
    canonical checkpoint files are transactionally replaced while unrelated
    files, such as Accelerate resume state, are retained. The manifest is
    published last and therefore acts as the completion marker.
    """
    if not IMMUTABLE_REVISION.fullmatch(recipe.base_revision):
        raise CheckpointValidationError(
            "export requires an immutable hexadecimal base_revision"
        )
    if not hasattr(model, "peft_config") or not hasattr(model, "save_pretrained"):
        raise TypeError("export_release_checkpoint expects a PEFT model")
    with _staged_checkpoint_directory(
        output_dir, overwrite=overwrite
    ) as (root, stage):
        model.save_pretrained(stage, safe_serialization=True)
        # PEFT also writes a README.md model card that names the base model as
        # it was loaded, possibly a local path; it is not a checkpoint file.
        (stage / "README.md").unlink(missing_ok=True)
        if not (stage / CANONICAL_ADAPTER_CONFIG).is_file() or not (
            stage / CANONICAL_ADAPTER_WEIGHTS
        ).is_file():
            raise RuntimeError("PEFT export did not create canonical adapter files")
        _canonicalize_adapter_config(stage, recipe)
        inventory = _model_tensor_inventory(model)
        _validate_adapter_weights(
            stage,
            recipe,
            expected_num_layers=inventory.text_layers,
            expected=inventory.adapter,
        )
        sidecar = {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" not in name.lower()
        }
        if not sidecar:
            raise RuntimeError("no trainable non-LoRA parameters were found")
        _validate_sidecar_tensors(sidecar, recipe)
        # The loader restores every merger, pooler, and AdaLN tensor; a
        # partly frozen module would otherwise publish an unloadable sidecar.
        _validate_tensor_inventory(
            _tensor_shapes(sidecar), inventory.sidecar,
            context="non-LoRA sidecar tensors",
        )
        from safetensors.torch import save_file

        save_file(sidecar, str(stage / CANONICAL_NON_LORA_WEIGHTS))
        (stage / CANONICAL_CONFIG).write_text(
            yaml.safe_dump(recipe.to_dict(), sort_keys=False), encoding="utf-8"
        )
        _write_processor_and_runtime(stage, processor, recipe)
        manifest_path = write_manifest(recipe, stage)
        manifest = CheckpointManifest.from_file(manifest_path)
        manifest.validate_files(stage, verify_hashes=True)
        manifest.load_and_validate_config(stage)
        manifest.load_and_validate_runtime(stage, recipe)
    return root / MANIFEST_FILE
