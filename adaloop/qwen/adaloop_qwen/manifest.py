"""The Qwen checkpoint manifest and runtime contract.

``adaloop_manifest.json`` names every canonical file of a checkpoint with its
SHA-256, and ``adaloop_runtime.json`` pins the input limits inference needs.
Both are parsed strictly here, without importing PyTorch, so incompatible
files are rejected before any weights are read.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import (
    BACKEND,
    IMMUTABLE_REVISION,
    READOUT_FOR_VARIANT,
    VARIANTS,
    ConditionerConfig,
    LoRAConfig,
    QwenAdaLoopConfig,
    _mapping,
    _strict_keys,
)
from .errors import CheckpointValidationError

MANIFEST_FILE = "adaloop_manifest.json"
MANIFEST_FORMAT_VERSION = 1
CANONICAL_CONFIG = "adaloop_config.yaml"
CANONICAL_ADAPTER_CONFIG = "adapter_config.json"
CANONICAL_ADAPTER_WEIGHTS = "adapter_model.safetensors"
CANONICAL_NON_LORA_WEIGHTS = "non_lora_trainables.safetensors"
CANONICAL_RUNTIME_CONTRACT = "adaloop_runtime.json"
CANONICAL_PROCESSOR_DIR = "processor"
REQUIRED_WEIGHT_ROLES = frozenset(
    {"adapter_config", "adapter_weights", "non_lora_weights"}
)
ROLE_PATHS = {
    "adapter_config": CANONICAL_ADAPTER_CONFIG,
    "adapter_weights": CANONICAL_ADAPTER_WEIGHTS,
    "non_lora_weights": CANONICAL_NON_LORA_WEIGHTS,
}
ROLE_FORMATS = {
    "adapter_config": "json",
    "adapter_weights": "safetensors",
    "non_lora_weights": "safetensors",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointValidationError(f"{context} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise CheckpointValidationError(
            f"{context} must be a portable path relative to the checkpoint root"
        )
    return value


@dataclass(frozen=True)
class WeightFile:
    role: str
    path: str
    format: str
    sha256: str

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> "WeightFile":
        context = f"weight_files[{index}]"
        raw = _mapping(raw, context, error=CheckpointValidationError)
        fields = {"role", "path", "format", "sha256"}
        _strict_keys(
            raw,
            context=context,
            allowed=fields,
            required=fields,
            error=CheckpointValidationError,
        )
        role = raw["role"]
        if role not in REQUIRED_WEIGHT_ROLES:
            raise CheckpointValidationError(
                f"{context}.role must be one of {sorted(REQUIRED_WEIGHT_ROLES)}"
            )
        path = _portable_relative_path(raw["path"], f"{context}.path")
        if path != ROLE_PATHS[role]:
            raise CheckpointValidationError(
                f"{context}.path must be {ROLE_PATHS[role]!r} for role {role!r}"
            )
        file_format = raw["format"]
        if file_format != ROLE_FORMATS[role]:
            raise CheckpointValidationError(
                f"{context}.format must be {ROLE_FORMATS[role]!r} for role {role!r}"
            )
        digest = raw["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CheckpointValidationError(
                f"{context}.sha256 must be 64 lowercase hexadecimal characters"
            )
        return cls(role=role, path=path, format=file_format, sha256=digest)


@dataclass(frozen=True)
class ProcessorFile:
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> "ProcessorFile":
        context = f"processor_files[{index}]"
        raw = _mapping(raw, context, error=CheckpointValidationError)
        fields = {"path", "sha256"}
        _strict_keys(
            raw,
            context=context,
            allowed=fields,
            required=fields,
            error=CheckpointValidationError,
        )
        path = _portable_relative_path(raw["path"], f"{context}.path")
        parts = PurePosixPath(path).parts
        if len(parts) < 2 or parts[0] != CANONICAL_PROCESSOR_DIR:
            raise CheckpointValidationError(
                f"{context}.path must lie under {CANONICAL_PROCESSOR_DIR}/"
            )
        digest = raw["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CheckpointValidationError(
                f"{context}.sha256 must be 64 lowercase hexadecimal characters"
            )
        return cls(path=path, sha256=digest)


@dataclass(frozen=True)
class QwenRuntimeContract:
    """Canonical input limits needed to reproduce released inference."""

    max_context_length: int
    min_pixels: int
    max_pixels: int
    format_version: int = 1

    @classmethod
    def from_dict(cls, raw: Any) -> "QwenRuntimeContract":
        raw = _mapping(raw, "Qwen runtime contract", error=CheckpointValidationError)
        fields = {
            "format_version",
            "max_context_length",
            "min_pixels",
            "max_pixels",
        }
        _strict_keys(
            raw,
            context="Qwen runtime contract",
            allowed=fields,
            required=fields,
            error=CheckpointValidationError,
        )
        if raw["format_version"] != 1:
            raise CheckpointValidationError(
                "Qwen runtime contract format_version must be 1"
            )

        def positive(name: str) -> int:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CheckpointValidationError(
                    f"Qwen runtime contract {name} must be a positive integer"
                )
            return value

        maximum_length = positive("max_context_length")
        minimum_pixels = positive("min_pixels")
        maximum_pixels = positive("max_pixels")
        if minimum_pixels > maximum_pixels:
            raise CheckpointValidationError(
                "Qwen runtime contract min_pixels must not exceed max_pixels"
            )
        return cls(
            max_context_length=maximum_length,
            min_pixels=minimum_pixels,
            max_pixels=maximum_pixels,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "QwenRuntimeContract":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError(
                f"could not read Qwen runtime contract {source}: {exc}"
            ) from exc
        return cls.from_dict(raw)

    @classmethod
    def from_recipe(cls, recipe: QwenAdaLoopConfig) -> "QwenRuntimeContract":
        return cls(
            max_context_length=recipe.preprocessing.max_length,
            min_pixels=recipe.preprocessing.min_pixels,
            max_pixels=recipe.preprocessing.max_pixels,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "format_version": self.format_version,
            "max_context_length": self.max_context_length,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

    def validate_recipe(self, recipe: QwenAdaLoopConfig) -> None:
        expected = type(self).from_recipe(recipe)
        if self != expected:
            raise CheckpointValidationError(
                "Qwen runtime contract contradicts adaloop_config.yaml: "
                f"runtime={self.to_dict()}, recipe={expected.to_dict()}"
            )


@dataclass(frozen=True)
class CheckpointManifest:
    format_version: int
    backend: str
    base_model: str
    base_revision: str
    variant: str
    loop_span: tuple[int, int] | None
    train_depth: int
    readout_mode: str
    lora: LoRAConfig
    conditioner: ConditionerConfig
    config_file: str
    config_sha256: str
    weight_files: tuple[WeightFile, ...]
    runtime_contract_file: str
    runtime_contract_sha256: str
    processor_files: tuple[ProcessorFile, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "CheckpointManifest":
        raw = _mapping(raw, "checkpoint manifest", error=CheckpointValidationError)
        fields = {
            "format_version", "backend", "base_model", "base_revision",
            "variant", "loop_span", "train_depth", "readout_mode", "lora",
            "conditioner", "config_file", "config_sha256", "weight_files",
            "runtime_contract_file", "runtime_contract_sha256", "processor_files",
        }
        _strict_keys(
            raw,
            context="checkpoint manifest",
            allowed=fields,
            required=fields,
            error=CheckpointValidationError,
        )
        if raw["format_version"] != MANIFEST_FORMAT_VERSION:
            raise CheckpointValidationError(
                f"format_version must be {MANIFEST_FORMAT_VERSION}"
            )
        if raw["backend"] != BACKEND:
            raise CheckpointValidationError(
                f"backend must be {BACKEND!r}; Qwen3 checkpoints are unsupported"
            )
        base_model = raw["base_model"]
        if not isinstance(base_model, str) or "Qwen2.5-VL" not in base_model:
            raise CheckpointValidationError(
                "base_model must identify a Qwen2.5-VL checkpoint"
            )
        revision = raw["base_revision"]
        if not isinstance(revision, str) or not IMMUTABLE_REVISION.fullmatch(revision):
            raise CheckpointValidationError(
                "released checkpoints must pin an immutable 40-64 character "
                "lowercase hexadecimal base_revision"
            )
        variant = str(raw["variant"]).lower()
        if variant not in VARIANTS:
            raise CheckpointValidationError(f"unsupported variant: {variant!r}")
        span_raw = raw["loop_span"]
        if span_raw is None:
            span = None
        elif (
            isinstance(span_raw, (list, tuple))
            and len(span_raw) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) for item in span_raw)
            and 0 <= span_raw[0] < span_raw[1]
        ):
            span = (span_raw[0], span_raw[1])
        else:
            raise CheckpointValidationError("loop_span must be null or [start, end]")
        if variant == "sft" and span is not None:
            raise CheckpointValidationError("sft checkpoints require loop_span=null")
        if variant != "sft" and span is None:
            raise CheckpointValidationError(f"variant {variant} requires loop_span")
        train_depth = raw["train_depth"]
        if isinstance(train_depth, bool) or not isinstance(train_depth, int) or train_depth < 1:
            raise CheckpointValidationError("train_depth must be a positive integer")
        if variant in {"sft", "adaln"} and train_depth != 1:
            raise CheckpointValidationError(f"variant {variant} requires train_depth=1")
        if variant in {"a", "b", "c"} and train_depth < 2:
            raise CheckpointValidationError(f"variant {variant} requires train_depth >= 2")
        expected_readout = READOUT_FOR_VARIANT[variant]
        if raw["readout_mode"] != expected_readout:
            raise CheckpointValidationError(
                f"variant {variant} requires readout_mode={expected_readout}"
            )
        try:
            lora = LoRAConfig.from_dict(raw["lora"])
            conditioner = ConditionerConfig.from_dict(raw["conditioner"])
        except ValueError as exc:
            raise CheckpointValidationError(str(exc)) from exc
        if variant == "sft" and conditioner.mode != "none":
            raise CheckpointValidationError("sft requires conditioner.mode=none")
        if variant in {"adaln", "b", "c"} and conditioner.mode == "none":
            raise CheckpointValidationError(f"variant {variant} requires a conditioner")
        config_digest = raw["config_sha256"]
        if not isinstance(config_digest, str) or not _SHA256.fullmatch(config_digest):
            raise CheckpointValidationError(
                "config_sha256 must be 64 lowercase hexadecimal characters"
            )
        weights_raw = raw["weight_files"]
        if not isinstance(weights_raw, list):
            raise CheckpointValidationError("weight_files must be a list")
        weights = tuple(WeightFile.from_dict(item, i) for i, item in enumerate(weights_raw))
        roles = {item.role for item in weights}
        if roles != REQUIRED_WEIGHT_ROLES or len(weights) != len(REQUIRED_WEIGHT_ROLES):
            raise CheckpointValidationError(
                "weight_files must contain exactly one of each role: "
                + ", ".join(sorted(REQUIRED_WEIGHT_ROLES))
            )
        runtime_file = _portable_relative_path(
            raw["runtime_contract_file"], "runtime_contract_file"
        )
        if runtime_file != CANONICAL_RUNTIME_CONTRACT:
            raise CheckpointValidationError(
                f"runtime_contract_file must be {CANONICAL_RUNTIME_CONTRACT!r}"
            )
        runtime_digest = raw["runtime_contract_sha256"]
        if not isinstance(runtime_digest, str) or not _SHA256.fullmatch(runtime_digest):
            raise CheckpointValidationError(
                "runtime_contract_sha256 must be 64 lowercase hexadecimal characters"
            )
        processor_raw = raw["processor_files"]
        if not isinstance(processor_raw, list) or not processor_raw:
            raise CheckpointValidationError("processor_files must be a non-empty list")
        processor_files = tuple(
            ProcessorFile.from_dict(item, index)
            for index, item in enumerate(processor_raw)
        )
        processor_paths = [item.path for item in processor_files]
        if len(set(processor_paths)) != len(processor_paths):
            raise CheckpointValidationError("processor_files contains duplicate paths")
        return cls(
            format_version=MANIFEST_FORMAT_VERSION, backend=BACKEND,
            base_model=base_model, base_revision=revision, variant=variant,
            loop_span=span, train_depth=train_depth, readout_mode=expected_readout,
            lora=lora, conditioner=conditioner,
            config_file=_portable_relative_path(raw["config_file"], "config_file"),
            config_sha256=config_digest, weight_files=weights,
            runtime_contract_file=runtime_file,
            runtime_contract_sha256=runtime_digest,
            processor_files=processor_files,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "CheckpointManifest":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError(f"could not read manifest {path}: {exc}") from exc
        return cls.from_dict(raw)

    def weight(self, role: str) -> WeightFile:
        for item in self.weight_files:
            if item.role == role:
                return item
        raise CheckpointValidationError(f"manifest is missing weight role {role!r}")

    def validate_files(self, root: str | Path, *, verify_hashes: bool = True) -> None:
        root = Path(root).resolve()
        required = (
            self.config_file,
            self.runtime_contract_file,
            *(item.path for item in self.weight_files),
            *(item.path for item in self.processor_files),
        )
        for relative in required:
            path = (root / relative).resolve()
            if path != root and root not in path.parents:
                raise CheckpointValidationError(f"checkpoint path escapes root: {relative}")
            if not path.is_file():
                raise CheckpointValidationError(f"checkpoint file is missing: {relative}")
        if verify_hashes:
            config_digest = _sha256(root / self.config_file)
            if config_digest != self.config_sha256:
                raise CheckpointValidationError(
                    f"SHA-256 mismatch for {self.config_file}: expected "
                    f"{self.config_sha256}, got {config_digest}"
                )
            for item in self.weight_files:
                digest = _sha256(root / item.path)
                if digest != item.sha256:
                    raise CheckpointValidationError(
                        f"SHA-256 mismatch for {item.path}: expected {item.sha256}, got {digest}"
                    )
            runtime_digest = _sha256(root / self.runtime_contract_file)
            if runtime_digest != self.runtime_contract_sha256:
                raise CheckpointValidationError(
                    f"SHA-256 mismatch for {self.runtime_contract_file}: expected "
                    f"{self.runtime_contract_sha256}, got {runtime_digest}"
                )
            for item in self.processor_files:
                digest = _sha256(root / item.path)
                if digest != item.sha256:
                    raise CheckpointValidationError(
                        f"SHA-256 mismatch for {item.path}: expected "
                        f"{item.sha256}, got {digest}"
                    )
        processor_root = root / CANONICAL_PROCESSOR_DIR
        observed_processor_files = {
            path.relative_to(root).as_posix()
            for path in processor_root.rglob("*")
            if path.is_file()
        }
        declared_processor_files = {item.path for item in self.processor_files}
        if observed_processor_files != declared_processor_files:
            raise CheckpointValidationError(
                "processor directory contents do not match the manifest: "
                f"missing={sorted(declared_processor_files - observed_processor_files)}, "
                f"unexpected={sorted(observed_processor_files - declared_processor_files)}"
            )

    def load_and_validate_config(self, root: str | Path) -> QwenAdaLoopConfig:
        config = QwenAdaLoopConfig.from_file(Path(root) / self.config_file)
        comparisons = {
            "backend": (self.backend, config.backend),
            "base_model": (self.base_model, config.base_model),
            "base_revision": (self.base_revision, config.base_revision),
            "variant": (self.variant, config.variant),
            "loop_span": (self.loop_span, config.adaptive_span),
            "train_depth": (self.train_depth, config.train_depth),
            "readout_mode": (self.readout_mode, config.readout.mode),
            "lora": (self.lora, config.lora),
            "conditioner": (self.conditioner, config.conditioner),
        }
        mismatches = [
            f"{name}: manifest={left!r}, config={right!r}"
            for name, (left, right) in comparisons.items() if left != right
        ]
        if mismatches:
            raise CheckpointValidationError(
                "checkpoint manifest/config mismatch: " + "; ".join(mismatches)
            )
        return config

    def load_and_validate_runtime(
        self, root: str | Path, recipe: QwenAdaLoopConfig
    ) -> QwenRuntimeContract:
        runtime = QwenRuntimeContract.from_file(
            Path(root) / self.runtime_contract_file
        )
        runtime.validate_recipe(recipe)
        return runtime


def _manifest_dict(recipe: QwenAdaLoopConfig, root: Path) -> dict[str, Any]:
    files = []
    for role in sorted(REQUIRED_WEIGHT_ROLES):
        path = root / ROLE_PATHS[role]
        files.append({
            "role": role, "path": ROLE_PATHS[role],
            "format": ROLE_FORMATS[role], "sha256": _sha256(path),
        })
    processor_root = root / CANONICAL_PROCESSOR_DIR
    processor_files = sorted(
        (
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in processor_root.rglob("*")
            if path.is_file()
        ),
        key=lambda item: item["path"],
    )
    if not processor_files:
        raise CheckpointValidationError("canonical checkpoint has no processor files")
    runtime_path = root / CANONICAL_RUNTIME_CONTRACT
    if not runtime_path.is_file():
        raise CheckpointValidationError(
            f"canonical checkpoint is missing {CANONICAL_RUNTIME_CONTRACT}"
        )
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "backend": recipe.backend,
        "base_model": recipe.base_model,
        "base_revision": recipe.base_revision,
        "variant": recipe.variant,
        "loop_span": list(recipe.adaptive_span) if recipe.adaptive_span is not None else None,
        "train_depth": recipe.train_depth,
        "readout_mode": recipe.readout.mode,
        "lora": {
            "rank": recipe.lora.rank, "alpha": recipe.lora.alpha,
            "dropout": recipe.lora.dropout,
            "target_modules": list(recipe.lora.target_modules),
        },
        "conditioner": {
            key: value for key, value in vars(recipe.conditioner).items()
            if value is not None
        },
        "config_file": CANONICAL_CONFIG,
        "config_sha256": _sha256(root / CANONICAL_CONFIG),
        "weight_files": files,
        "runtime_contract_file": CANONICAL_RUNTIME_CONTRACT,
        "runtime_contract_sha256": _sha256(runtime_path),
        "processor_files": processor_files,
    }


def write_manifest(recipe: QwenAdaLoopConfig, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    raw = _manifest_dict(recipe, root)
    CheckpointManifest.from_dict(raw)
    destination = root / MANIFEST_FILE
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination
