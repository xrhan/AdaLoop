"""Strict portable checkpoints for the PaliGemma2/JAX AdaLoop backend.

The public artifact is intentionally an overlay, not a redistribution of the
licensed PaliGemma base.  It contains the exact trainable parameter leaves plus
a signed recipe and runtime contract.  Loading derives that leaf set (paths,
shapes, and dtypes) from the recipe, never from the overlay, so a repacked
overlay that omits a trained leaf cannot silently fall back to base or
initial values.  Loading also requires the exact base NPZ and tokenizer named
by the recipe; both are checked by byte size and SHA-256 before a
reconstruction hook is allowed to use them.

Export is deliberately stricter than ordinary Flax restore.  A caller
supplies a hook which reconstructs the complete parameter template from the
pinned base.  Every leaf omitted from the compact overlay is then compared
exactly with the full trained parameter tree.  This makes it impossible to silently
discard a trained or otherwise changed parameter while exporting.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from .config import BACKEND, PaliGemmaAdaLoopConfig
from .constants import (
    GEMMA_DEPTH,
    GEMMA_HEAD_DIM,
    GEMMA_HEADS,
    GEMMA_KV_HEADS,
    GEMMA_WIDTH,
    PAPER_VISION_WIDTH,
)
from .errors import CheckpointValidationError, ReconstructionError


FORMAT_VERSION = 1
CANONICAL_MANIFEST = "adaloop_manifest.json"
CANONICAL_CONFIG = "adaloop_config.yaml"
CANONICAL_RUNTIME = "adaloop_runtime.json"
CANONICAL_WEIGHTS = "trainable.params.msgpack"
CANONICAL_FILES = frozenset(
    {CANONICAL_MANIFEST, CANONICAL_CONFIG, CANONICAL_RUNTIME, CANONICAL_WEIGHTS}
)
FILE_ROLES = {
    "config": (CANONICAL_CONFIG, "yaml"),
    "runtime_contract": (CANONICAL_RUNTIME, "json"),
    "trainable_weights": (CANONICAL_WEIGHTS, "flax_msgpack"),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_RE = re.compile(r"^encoderblock_(\d+)$")
_LORA_NAMES = frozenset(
    {
        "q_lora_a",
        "q_lora_b",
        "kv_lora_a",
        "kv_lora_b",
        "o_lora_a",
        "o_lora_b",
    }
)
_DTYPE_BYTES = {"bfloat16": 2, "float32": 4}

RestoreHook = Callable[[bytes], Mapping[str, Any]]
SerializeHook = Callable[[Mapping[str, Any]], bytes]
BaseReconstructionHook = Callable[
    [PaliGemmaAdaLoopConfig, Path, Path], Mapping[str, Any]
]
PublicReconstructionHook = Callable[
    [PaliGemmaAdaLoopConfig, Path, Path, Mapping[str, Any]], Any
]


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointValidationError(f"{context} must be a mapping")
    return value


def _strict_keys(
    raw: Mapping[str, Any], *, context: str, allowed: set[str], required: set[str]
) -> None:
    unknown = set(raw) - allowed
    missing = required - set(raw)
    if unknown:
        raise CheckpointValidationError(
            f"{context} contains unknown field(s): {', '.join(sorted(unknown))}"
        )
    if missing:
        raise CheckpointValidationError(
            f"{context} is missing field(s): {', '.join(sorted(missing))}"
        )


def _positive_int(value: Any, context: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise CheckpointValidationError(f"{context} must be a {qualifier} integer")
    return value


def _sha256_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CheckpointValidationError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    return value


def _portable_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointValidationError(f"{context} must be a non-empty string")
    if "\\" in value:
        raise CheckpointValidationError(f"{context} must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("~")
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CheckpointValidationError(
            f"{context} must be a normalized path relative to the checkpoint root"
        )
    return value


def _safe_child(root: Path, relative: str, context: str) -> Path:
    normalized = _portable_path(relative, context)
    child = root.joinpath(*PurePosixPath(normalized).parts)
    current = root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise CheckpointValidationError(f"{context} must not traverse a symlink")
    return child


def _regular_file(path: Path, context: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise CheckpointValidationError(f"{context} must be a regular file: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointValidationError(
                    f"{context} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CheckpointValidationError(
            f"{context} contains non-finite JSON number {value}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except CheckpointValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"could not read {context} {path}: {exc}") from exc
    return _mapping(value, context)


def _default_restore(payload: bytes) -> Mapping[str, Any]:
    try:
        from flax import serialization
    except ImportError as exc:  # pragma: no cover - exercised in JAX environment
        raise RuntimeError(
            "restoring Flax msgpack requires the `jax` extra or an explicit restore_hook"
        ) from exc
    value = serialization.msgpack_restore(payload)
    if not isinstance(value, Mapping):
        raise CheckpointValidationError("Flax msgpack root must be a mapping")
    return value


def _default_serialize(value: Mapping[str, Any]) -> bytes:
    try:
        from flax import serialization
    except ImportError as exc:  # pragma: no cover - exercised in JAX environment
        raise RuntimeError(
            "serializing Flax msgpack requires the `jax` extra or an explicit serialize_hook"
        ) from exc
    return serialization.to_bytes(value)


@dataclass(frozen=True)
class FileRecord:
    role: str
    path: str
    format: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> "FileRecord":
        context = f"manifest.files[{index}]"
        raw = _mapping(raw, context)
        fields = {"role", "path", "format", "sha256", "size_bytes"}
        _strict_keys(raw, context=context, allowed=fields, required=fields)
        role = raw["role"]
        if role not in FILE_ROLES:
            raise CheckpointValidationError(
                f"{context}.role must be one of {sorted(FILE_ROLES)}"
            )
        path = _portable_path(raw["path"], f"{context}.path")
        expected_path, expected_format = FILE_ROLES[role]
        if (path, raw["format"]) != (expected_path, expected_format):
            raise CheckpointValidationError(
                f"{context} must use path={expected_path!r}, format={expected_format!r}"
            )
        return cls(
            role=role,
            path=path,
            format=expected_format,
            sha256=_sha256_digest(raw["sha256"], f"{context}.sha256"),
            size_bytes=_positive_int(
                raw["size_bytes"], f"{context}.size_bytes", allow_zero=True
            ),
        )

    def validate(self, root: Path, *, verify_hashes: bool) -> Path:
        path = _safe_child(root, self.path, f"manifest file {self.role}")
        _regular_file(path, f"manifest file {self.role}")
        actual_size = path.stat().st_size
        if actual_size != self.size_bytes:
            raise CheckpointValidationError(
                f"size mismatch for {self.path}: manifest={self.size_bytes}, "
                f"actual={actual_size}"
            )
        if verify_hashes:
            actual_hash = _sha256(path)
            if actual_hash != self.sha256:
                raise CheckpointValidationError(
                    f"SHA-256 mismatch for {self.path}: manifest={self.sha256}, "
                    f"actual={actual_hash}"
                )
        return path


@dataclass(frozen=True)
class TensorRecord:
    path: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> "TensorRecord":
        context = f"manifest.tensors[{index}]"
        raw = _mapping(raw, context)
        fields = {"path", "shape", "dtype", "size_bytes"}
        _strict_keys(raw, context=context, allowed=fields, required=fields)
        path = _portable_path(raw["path"], f"{context}.path")
        shape_raw = raw["shape"]
        if not isinstance(shape_raw, list):
            raise CheckpointValidationError(f"{context}.shape must be a list")
        shape = tuple(
            _positive_int(item, f"{context}.shape", allow_zero=True)
            for item in shape_raw
        )
        dtype = raw["dtype"]
        if not isinstance(dtype, str) or not dtype:
            raise CheckpointValidationError(f"{context}.dtype must be non-empty")
        return cls(
            path=path,
            shape=shape,
            dtype=dtype,
            size_bytes=_positive_int(
                raw["size_bytes"], f"{context}.size_bytes", allow_zero=True
            ),
        )


@dataclass(frozen=True)
class TopologyRecord:
    variant: str
    adaptive_span: tuple[int, int] | None
    train_depth: int
    vision_round_mode: str
    readout_mode: str
    base_model_sha256: str
    tokenizer_sha256: str

    @classmethod
    def from_dict(cls, raw: Any) -> "TopologyRecord":
        raw = _mapping(raw, "manifest.topology")
        fields = {
            "variant",
            "adaptive_span",
            "train_depth",
            "vision_round_mode",
            "readout_mode",
            "base_model_sha256",
            "tokenizer_sha256",
        }
        _strict_keys(raw, context="manifest.topology", allowed=fields, required=fields)
        span_raw = raw["adaptive_span"]
        if span_raw is None:
            span = None
        elif (
            isinstance(span_raw, list)
            and len(span_raw) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) for item in span_raw)
            and 0 <= span_raw[0] < span_raw[1]
        ):
            span = (span_raw[0], span_raw[1])
        else:
            raise CheckpointValidationError(
                "manifest.topology.adaptive_span must be null or [start, end]"
            )
        for name in ("variant", "vision_round_mode", "readout_mode"):
            if not isinstance(raw[name], str) or not raw[name]:
                raise CheckpointValidationError(
                    f"manifest.topology.{name} must be a non-empty string"
                )
        return cls(
            variant=raw["variant"],
            adaptive_span=span,
            train_depth=_positive_int(raw["train_depth"], "manifest.topology.train_depth"),
            vision_round_mode=raw["vision_round_mode"],
            readout_mode=raw["readout_mode"],
            base_model_sha256=_sha256_digest(
                raw["base_model_sha256"], "manifest.topology.base_model_sha256"
            ),
            tokenizer_sha256=_sha256_digest(
                raw["tokenizer_sha256"], "manifest.topology.tokenizer_sha256"
            ),
        )

    @classmethod
    def from_recipe(cls, recipe: PaliGemmaAdaLoopConfig) -> "TopologyRecord":
        return cls(
            variant=recipe.variant,
            adaptive_span=recipe.adaptive_span,
            train_depth=recipe.train_depth,
            vision_round_mode=recipe.vision_round_mode,
            readout_mode=recipe.readout.mode,
            base_model_sha256=recipe.base_model.sha256,
            tokenizer_sha256=recipe.tokenizer.sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["adaptive_span"] = (
            list(self.adaptive_span) if self.adaptive_span is not None else None
        )
        return result


@dataclass(frozen=True)
class PaliGemmaRuntimeContract:
    format_version: int
    backend: str
    base_model: Mapping[str, Any]
    tokenizer: Mapping[str, Any]
    backbone: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    generation: Mapping[str, Any]
    parameter_overlay: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Any) -> "PaliGemmaRuntimeContract":
        raw = _mapping(raw, "runtime contract")
        fields = {
            "format_version",
            "backend",
            "base_model",
            "tokenizer",
            "backbone",
            "preprocessing",
            "generation",
            "parameter_overlay",
        }
        _strict_keys(raw, context="runtime contract", allowed=fields, required=fields)
        if raw["format_version"] != FORMAT_VERSION or raw["backend"] != BACKEND:
            raise CheckpointValidationError("unsupported runtime contract format/backend")
        overlay = _mapping(raw["parameter_overlay"], "parameter_overlay")
        overlay_fields = {
            "format",
            "path",
            "path_separator",
            "legacy_parameter_names",
            "root_scopes",
        }
        _strict_keys(
            overlay,
            context="parameter_overlay",
            allowed=overlay_fields,
            required=overlay_fields,
        )
        if (
            overlay["format"] != "flax_msgpack"
            or overlay["path"] != CANONICAL_WEIGHTS
            or overlay["path_separator"] != "/"
            or overlay["legacy_parameter_names"] is not True
        ):
            raise CheckpointValidationError("invalid parameter_overlay contract")
        scopes = overlay["root_scopes"]
        if not isinstance(scopes, list) or not scopes or any(
            not isinstance(item, str) or not item for item in scopes
        ):
            raise CheckpointValidationError("parameter_overlay.root_scopes must be strings")
        if len(set(scopes)) != len(scopes):
            raise CheckpointValidationError("parameter_overlay.root_scopes contains duplicates")
        for block in ("base_model", "tokenizer", "backbone", "preprocessing", "generation"):
            _mapping(raw[block], f"runtime contract {block}")
        return cls(
            format_version=FORMAT_VERSION,
            backend=BACKEND,
            base_model=dict(raw["base_model"]),
            tokenizer=dict(raw["tokenizer"]),
            backbone=dict(raw["backbone"]),
            preprocessing=dict(raw["preprocessing"]),
            generation=dict(raw["generation"]),
            parameter_overlay=dict(overlay),
        )

    @classmethod
    def from_recipe(
        cls, recipe: PaliGemmaAdaLoopConfig, root_scopes: Sequence[str]
    ) -> "PaliGemmaRuntimeContract":
        data = recipe.to_dict()
        return cls(
            format_version=FORMAT_VERSION,
            backend=BACKEND,
            base_model=data["base_model"],
            tokenizer=data["tokenizer"],
            backbone=data["backbone"],
            preprocessing=data["preprocessing"],
            generation=data["generation"],
            parameter_overlay={
                "format": "flax_msgpack",
                "path": CANONICAL_WEIGHTS,
                "path_separator": "/",
                "legacy_parameter_names": True,
                "root_scopes": sorted(root_scopes),
            },
        )

    def validate_recipe(self, recipe: PaliGemmaAdaLoopConfig) -> None:
        expected = self.from_recipe(
            recipe, self.parameter_overlay["root_scopes"]
        )
        if asdict(self) != asdict(expected):
            raise CheckpointValidationError(
                "adaloop_runtime.json contradicts adaloop_config.yaml"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaliGemmaCheckpointManifest:
    format_version: int
    backend: str
    files: tuple[FileRecord, ...]
    topology: TopologyRecord
    tensors: tuple[TensorRecord, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "PaliGemmaCheckpointManifest":
        raw = _mapping(raw, "manifest")
        fields = {"format_version", "backend", "files", "topology", "tensors"}
        _strict_keys(raw, context="manifest", allowed=fields, required=fields)
        if raw["format_version"] != FORMAT_VERSION or raw["backend"] != BACKEND:
            raise CheckpointValidationError("unsupported checkpoint format/backend")
        files_raw = raw["files"]
        tensors_raw = raw["tensors"]
        if not isinstance(files_raw, list) or not isinstance(tensors_raw, list):
            raise CheckpointValidationError("manifest files/tensors must be lists")
        files = tuple(FileRecord.from_dict(item, i) for i, item in enumerate(files_raw))
        roles = [item.role for item in files]
        if len(set(roles)) != len(roles) or set(roles) != set(FILE_ROLES):
            raise CheckpointValidationError(
                f"manifest must contain each file role exactly once: {sorted(FILE_ROLES)}"
            )
        tensors = tuple(
            TensorRecord.from_dict(item, i) for i, item in enumerate(tensors_raw)
        )
        tensor_paths = [item.path for item in tensors]
        if not tensors or len(set(tensor_paths)) != len(tensor_paths):
            raise CheckpointValidationError(
                "manifest tensors must be non-empty with unique paths"
            )
        if tensor_paths != sorted(tensor_paths):
            raise CheckpointValidationError("manifest tensors must be sorted by path")
        return cls(
            format_version=FORMAT_VERSION,
            backend=BACKEND,
            files=files,
            topology=TopologyRecord.from_dict(raw["topology"]),
            tensors=tensors,
        )

    @classmethod
    def from_file(cls, path: Path) -> "PaliGemmaCheckpointManifest":
        return cls.from_dict(_read_json(path, "manifest"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "backend": self.backend,
            "files": [asdict(item) for item in self.files],
            "topology": self.topology.to_dict(),
            "tensors": [
                {
                    "path": item.path,
                    "shape": list(item.shape),
                    "dtype": item.dtype,
                    "size_bytes": item.size_bytes,
                }
                for item in self.tensors
            ],
        }


def _flatten_tree(
    tree: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    result: dict[tuple[str, ...], Any] = {}
    for raw_key, value in tree.items():
        key = str(raw_key)
        if not key or "/" in key:
            raise CheckpointValidationError(
                "parameter-tree keys must be non-empty and must not contain '/'"
            )
        path = prefix + (key,)
        if isinstance(value, Mapping):
            if not value:
                raise CheckpointValidationError(
                    f"parameter subtree {'/'.join(path)} is empty"
                )
            result.update(_flatten_tree(value, path))
        else:
            if path in result:  # pragma: no cover - mappings cannot repeat a key
                raise CheckpointValidationError(
                    f"duplicate parameter path: {'/'.join(path)}"
                )
            result[path] = value
    return result


def _unflatten_tree(flat: Mapping[tuple[str, ...], Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path in sorted(flat):
        if not path:
            raise CheckpointValidationError("empty parameter path")
        cursor: MutableMapping[str, Any] = root
        for part in path[:-1]:
            existing = cursor.get(part)
            if existing is None:
                child: dict[str, Any] = {}
                cursor[part] = child
                cursor = child
            elif isinstance(existing, MutableMapping):
                cursor = existing
            else:
                raise CheckpointValidationError(
                    f"parameter leaf/subtree collision at {'/'.join(path)}"
                )
        if path[-1] in cursor:
            raise CheckpointValidationError(
                f"duplicate parameter path: {'/'.join(path)}"
            )
        cursor[path[-1]] = flat[path]
    return root


def _array(value: Any, context: str) -> np.ndarray:
    try:
        result = np.asarray(value)
    except Exception as exc:
        raise CheckpointValidationError(
            f"parameter leaf {context} is not array-like"
        ) from exc
    if result.dtype.hasobject:
        raise CheckpointValidationError(
            f"parameter leaf {context} has unsupported object dtype"
        )
    return result


def _leaf_bytes_digest(value: Any, context: str) -> str:
    array = _array(value, context)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    # Hash the C-order bytes through a flat uint8 view.  ``memoryview`` cannot
    # export extension dtypes such as ml_dtypes' bfloat16 (every frozen leaf
    # of the PaliGemma base) or zero-size arrays; the view accepts any
    # fixed-size dtype and yields identical bytes for builtin dtypes.
    digest.update(array.reshape(-1).view(np.uint8))
    return digest.hexdigest()


def _tensor_inventory(flat: Mapping[tuple[str, ...], Any]) -> tuple[TensorRecord, ...]:
    records: list[TensorRecord] = []
    for path in sorted(flat):
        name = "/".join(path)
        header = flat[path]
        if isinstance(header, TensorRecord):
            # Header-only restore (JAX-free inspection) of an undecoded leaf.
            records.append(
                TensorRecord(name, header.shape, header.dtype, header.size_bytes)
            )
            continue
        array = _array(flat[path], name)
        records.append(
            TensorRecord(
                path=name,
                shape=tuple(int(size) for size in array.shape),
                dtype=str(array.dtype),
                size_bytes=int(array.nbytes),
            )
        )
    return tuple(records)


# flax.serialization extension codes for an ndarray and a NumPy scalar leaf.
_FLAX_NDARRAY = 1
_FLAX_NPSCALAR = 3


def _header_record(shape: Any, dtype: str, size_bytes: int) -> TensorRecord:
    # NumPy itself has no bfloat16 (that needs ml_dtypes); its width is fixed.
    itemsize = 2 if dtype == "bfloat16" else np.dtype(dtype).itemsize
    shape = tuple(int(size) for size in shape)
    if size_bytes != itemsize * math.prod(shape):
        raise CheckpointValidationError(
            f"Flax msgpack leaf of shape {shape} and dtype {dtype} has "
            f"{size_bytes} bytes"
        )
    return TensorRecord(path="", shape=shape, dtype=dtype, size_bytes=size_bytes)


def read_overlay_headers(payload: bytes) -> Mapping[str, Any]:
    """Read a Flax msgpack overlay's tree and leaf headers without JAX.

    A restore hook for inspection: each array leaf becomes a
    :class:`TensorRecord` carrying its shape, dtype, and byte size, which is
    everything inventory validation compares.  Leaves are never decoded, so
    bfloat16 overlays need neither Flax nor ml_dtypes.
    """

    import msgpack

    def ext_hook(code: int, data: bytes) -> Any:
        if code not in (_FLAX_NDARRAY, _FLAX_NPSCALAR):
            raise CheckpointValidationError(
                f"unsupported Flax msgpack extension type {code}"
            )
        shape, dtype, buffer = msgpack.unpackb(data, raw=True)
        return _header_record(shape, dtype.decode("ascii"), len(buffer))

    def object_hook(value: dict[str, Any]) -> Any:
        # Flax stores leaves above 1 GiB as a mapping of flat chunks.
        if "__msgpack_chunked_array__" not in value:
            return value
        shape = [value["shape"][str(index)] for index in range(len(value["shape"]))]
        chunks = [value["chunks"][str(index)] for index in range(len(value["chunks"]))]
        if len({chunk.dtype for chunk in chunks}) != 1:
            raise CheckpointValidationError("chunked Flax msgpack leaf mixes dtypes")
        return _header_record(
            shape, chunks[0].dtype, sum(chunk.size_bytes for chunk in chunks)
        )

    try:
        tree = msgpack.unpackb(
            payload, ext_hook=ext_hook, object_hook=object_hook, raw=False
        )
    except CheckpointValidationError:
        raise
    except Exception as exc:
        raise CheckpointValidationError(f"invalid Flax msgpack overlay: {exc}") from exc
    if not isinstance(tree, Mapping):
        raise CheckpointValidationError("Flax msgpack root must be a mapping")
    return tree


def _is_lora(path: tuple[str, ...]) -> bool:
    return len(path) == 4 and path[:3] == ("llm", "layers", "attn") and path[3] in _LORA_NAMES


def _adaln_block(path: tuple[str, ...]) -> int | None:
    if len(path) < 5 or path[0:2] != ("img", "Transformer") or path[3] != "adaln":
        return None
    match = _BLOCK_RE.fullmatch(path[2])
    return None if match is None else int(match.group(1))


def _expected_trainable_paths(
    flat: Mapping[tuple[str, ...], Any], recipe: PaliGemmaAdaLoopConfig
) -> set[tuple[str, ...]]:
    """Select the trainable leaves of a full parameter tree by category.

    Compaction uses this to split a full tree; it checks topology coverage
    only.  Overlays are compared with :func:`_recipe_trainable_inventory`.
    """

    selected: set[tuple[str, ...]] = set()
    lora_observed: set[str] = set()
    connector_count = 0
    conditioner_count = 0
    adaln_blocks: set[int] = set()
    malformed_adaln: list[str] = []

    conditioned = recipe.conditioner.mode != "none"
    span = recipe.adaptive_span
    for path in flat:
        if _is_lora(path):
            selected.add(path)
            lora_observed.add(path[-1])
            continue
        if len(path) >= 3 and path[:2] == ("img", "head"):
            selected.add(path)
            connector_count += 1
            continue
        if path and path[0] == "question_conditioner":
            if not conditioned:
                raise CheckpointValidationError(
                    "unconditioned recipe contains question_conditioner parameters"
                )
            selected.add(path)
            conditioner_count += 1
            continue
        block = _adaln_block(path)
        if block is not None:
            if not conditioned or span is None or not span[0] <= block < span[1]:
                malformed_adaln.append("/".join(path))
                continue
            selected.add(path)
            adaln_blocks.add(block)
            continue
        if "adaln" in path:
            malformed_adaln.append("/".join(path))

    missing_lora = _LORA_NAMES - lora_observed
    if missing_lora or lora_observed - _LORA_NAMES:
        raise CheckpointValidationError(
            "parameter tree has incomplete LoRA coverage: "
            f"missing={sorted(missing_lora)}, "
            f"unexpected={sorted(lora_observed - _LORA_NAMES)}"
        )
    if connector_count == 0:
        raise CheckpointValidationError("parameter tree is missing img/head connector leaves")
    if conditioned:
        assert span is not None
        expected_blocks = set(range(*span))
        if conditioner_count == 0 or adaln_blocks != expected_blocks or malformed_adaln:
            raise CheckpointValidationError(
                "parameter tree contradicts conditioner/AdaLN topology: "
                f"question_conditioner_leaves={conditioner_count}, "
                f"missing_adaln_blocks={sorted(expected_blocks - adaln_blocks)}, "
                f"unexpected_adaln_blocks={sorted(adaln_blocks - expected_blocks)}, "
                f"malformed_or_out_of_span={sorted(malformed_adaln)}"
            )
    elif conditioner_count or adaln_blocks or malformed_adaln:
        raise CheckpointValidationError(
            "SFT recipe must not contain question-conditioner or AdaLN leaves"
        )
    return selected


def _recipe_trainable_inventory(
    recipe: PaliGemmaAdaLoopConfig,
) -> tuple[TensorRecord, ...]:
    """Return every trainable leaf of the recipe's model, sorted by path.

    These are the q/fused-kv/o LoRA leaves of all scanned Gemma layers and the
    ``img/head`` connector, plus, for conditioned recipes, the shared question
    pooler and each in-span AdaLN generator.  The pinned backbone's dimensions
    and the recipe fix every shape, so validation needs no JAX.  The connector
    keeps the pinned base's bfloat16; the runtime initializes the other leaves
    in float32.
    """

    layers, width, rank = GEMMA_DEPTH, GEMMA_WIDTH, recipe.lora.rank
    heads, kv_heads, head_dim = GEMMA_HEADS, GEMMA_KV_HEADS, GEMMA_HEAD_DIM
    lora = {
        "q": {
            "q_lora_a": (layers, width, rank),
            "q_lora_b": (layers, heads, rank, head_dim),
        },
        "kv": {
            "kv_lora_a": (layers, width, rank),
            "kv_lora_b": (layers, 2, kv_heads, rank, head_dim),
        },
        "o": {
            "o_lora_a": (layers, heads, head_dim, rank),
            "o_lora_b": (layers, rank, width),
        },
    }
    leaves: dict[str, tuple[tuple[int, ...], str]] = {
        "img/head/kernel": ((PAPER_VISION_WIDTH, width), "bfloat16"),
        "img/head/bias": ((width,), "bfloat16"),
    }
    for target in recipe.lora.target_modules:
        for name, shape in lora[target].items():
            leaves[f"llm/layers/attn/{name}"] = (shape, "float32")
    if recipe.conditioner.mode != "none":
        span = recipe.adaptive_span
        cond = recipe.conditioner.hidden_size
        cond_heads = recipe.conditioner.attention_heads
        hidden = recipe.adaln.hidden_size
        assert span is not None and cond and cond_heads and hidden
        # The pooler's MLP ratio is 2; the AdaLN generator's is 1 (no mlp_in).
        pooler = {
            "input_norm/scale": (width,),
            "input_proj/kernel": (width, cond),
            "query": (1, cond),
            "q_proj/kernel": (cond, cond_heads, cond // cond_heads),
            "k_proj/kernel": (cond, cond_heads, cond // cond_heads),
            "v_proj/kernel": (cond, cond_heads, cond // cond_heads),
            "out_proj/kernel": (cond_heads, cond // cond_heads, cond),
            "mlp_norm/scale": (cond,),
            "mlp_in/kernel": (cond, 2 * cond),
            "mlp_in/bias": (2 * cond,),
            "mlp_out/kernel": (2 * cond, cond),
            "mlp_out/bias": (cond,),
        }
        for name, shape in pooler.items():
            leaves[f"question_conditioner/pooler/{name}"] = (shape, "float32")
        adaln = {
            "cond_norm/scale": (cond,),
            "cond_norm/bias": (cond,),
            "cond_proj/kernel": (cond, hidden),
            "cond_proj/bias": (hidden,),
            "to_adaln/kernel": (hidden, 6 * PAPER_VISION_WIDTH),
            "to_adaln/bias": (6 * PAPER_VISION_WIDTH,),
        }
        for block in range(*span):
            for name, shape in adaln.items():
                path = f"img/Transformer/encoderblock_{block}/adaln/{name}"
                leaves[path] = (shape, "float32")
    return tuple(
        TensorRecord(path, shape, dtype, _DTYPE_BYTES[dtype] * math.prod(shape))
        for path, (shape, dtype) in sorted(
            leaves.items(), key=lambda item: item[0].split("/")
        )
    )


def _validate_recipe_inventory(
    inventory: Sequence[TensorRecord], recipe: PaliGemmaAdaLoopConfig
) -> None:
    """Require exactly the recipe's trainable leaves, shapes, and dtypes."""

    expected = {record.path: record for record in _recipe_trainable_inventory(recipe)}
    observed = {record.path: record for record in inventory}
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    mismatched = [
        f"{path}: expected={expected[path].shape}/{expected[path].dtype}, "
        f"observed={observed[path].shape}/{observed[path].dtype}"
        for path in sorted(set(expected) & set(observed))
        if observed[path] != expected[path]
    ]
    if missing or unexpected or mismatched:
        raise CheckpointValidationError(
            "compact overlay contradicts the recipe's trainable leaves: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )


def _validate_overlay_tree(
    overlay: Mapping[str, Any],
    recipe: PaliGemmaAdaLoopConfig,
    manifest_tensors: Sequence[TensorRecord] | None = None,
) -> tuple[dict[tuple[str, ...], Any], tuple[TensorRecord, ...]]:
    flat = _flatten_tree(overlay)
    inventory = _tensor_inventory(flat)
    _validate_recipe_inventory(inventory, recipe)
    if manifest_tensors is not None and tuple(manifest_tensors) != inventory:
        raise CheckpointValidationError(
            "trainable.params.msgpack tensor inventory contradicts manifest"
        )
    return flat, inventory


def _restore_overlay(
    path: Path,
    recipe: PaliGemmaAdaLoopConfig,
    manifest_tensors: Sequence[TensorRecord],
    restore_hook: RestoreHook | None,
) -> Mapping[str, Any]:
    hook = restore_hook or _default_restore
    try:
        overlay = hook(path.read_bytes())
    except CheckpointValidationError:
        raise
    except Exception as exc:
        raise CheckpointValidationError(f"could not restore {path.name}: {exc}") from exc
    overlay = _mapping(overlay, "trainable overlay")
    _validate_overlay_tree(overlay, recipe, manifest_tensors)
    return overlay


def _verify_frozen_leaves(
    trained_flat: Mapping[tuple[str, ...], Any],
    template_flat: Mapping[tuple[str, ...], Any],
    trainable: set[tuple[str, ...]],
) -> None:
    trained_paths = set(trained_flat)
    template_paths = set(template_flat)
    if trained_paths != template_paths:
        raise CheckpointValidationError(
            "trained parameter tree differs from the exact reconstructed "
            "template: "
            f"missing={sorted('/'.join(path) for path in template_paths - trained_paths)}, "
            f"unexpected={sorted('/'.join(path) for path in trained_paths - template_paths)}"
        )
    for path in sorted(trained_paths - trainable):
        name = "/".join(path)
        trained = _array(trained_flat[path], name)
        template = _array(template_flat[path], name)
        if trained.shape != template.shape or str(trained.dtype) != str(template.dtype):
            raise CheckpointValidationError(
                f"omitted frozen leaf {name} differs in shape or dtype"
            )
        if _leaf_bytes_digest(trained, name) != _leaf_bytes_digest(template, name):
            raise CheckpointValidationError(
                f"omitted leaf {name} differs from the exact reconstructed base; "
                "refusing a lossy compact export"
            )


def validate_external_asset(path: str | Path, asset: Any, context: str) -> Path:
    """Verify one licensed external asset against its signed recipe record."""
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise CheckpointValidationError(f"{context} must not be a symlink")
    resolved = requested.resolve()
    _regular_file(resolved, context)
    size = resolved.stat().st_size
    if size != asset.size_bytes:
        raise CheckpointValidationError(
            f"{context} size mismatch: recipe={asset.size_bytes}, actual={size}"
        )
    digest = _sha256(resolved)
    if digest != asset.sha256:
        raise CheckpointValidationError(
            f"{context} SHA-256 mismatch: recipe={asset.sha256}, actual={digest}"
        )
    return resolved


def _runtime_root_scopes(recipe: PaliGemmaAdaLoopConfig) -> tuple[str, ...]:
    scopes = ["img", "llm"]
    if recipe.conditioner.mode != "none":
        scopes.append("question_conditioner")
    return tuple(scopes)


def _validate_runtime_root_scopes(
    runtime: PaliGemmaRuntimeContract, recipe: PaliGemmaAdaLoopConfig
) -> None:
    observed = tuple(sorted(runtime.parameter_overlay["root_scopes"]))
    expected = tuple(sorted(_runtime_root_scopes(recipe)))
    if observed != expected:
        raise CheckpointValidationError(
            f"runtime root scopes contradict recipe: expected={expected}, observed={observed}"
        )


@dataclass(frozen=True)
class VerifiedPaliGemmaCheckpoint:
    """A completely verified portable artifact and its external base assets.

    ``reconstruct`` is the integration seam for the model runtime.  Its hook is
    called as ``hook(recipe, base_npz, tokenizer_model, overlay_tree)`` only
    after all internal and external hashes have passed validation and the
    overlay's tensor inventory has matched both the manifest and the recipe's
    complete trainable leaves.
    """

    root: Path
    recipe: PaliGemmaAdaLoopConfig
    manifest: PaliGemmaCheckpointManifest
    runtime_contract: PaliGemmaRuntimeContract
    overlay_path: Path
    base_checkpoint: Path
    tokenizer: Path
    reconstructed: Any = field(default=None, compare=False, repr=False)
    _restore_hook: RestoreHook | None = field(default=None, compare=False, repr=False)

    def load_overlay(self, restore_hook: RestoreHook | None = None) -> Mapping[str, Any]:
        return _restore_overlay(
            self.overlay_path,
            self.recipe,
            self.manifest.tensors,
            restore_hook or self._restore_hook,
        )

    def reconstruct(
        self,
        hook: PublicReconstructionHook | None = None,
        *,
        restore_hook: RestoreHook | None = None,
    ) -> Any:
        return self._reconstruct(hook, self.load_overlay(restore_hook))

    def _reconstruct(
        self, hook: PublicReconstructionHook | None, overlay: Mapping[str, Any]
    ) -> Any:
        if hook is None:
            # Keep JAX/Flax lazy: inspection and metadata imports remain
            # accelerator-neutral, while normal loading needs no user hook.
            from .runtime import reconstruct_runtime

            hook = reconstruct_runtime
        if not callable(hook):
            raise TypeError("reconstruction hook must be callable")
        try:
            return hook(self.recipe, self.base_checkpoint, self.tokenizer, overlay)
        except Exception as exc:
            raise ReconstructionError(f"PaliGemma reconstruction hook failed: {exc}") from exc


class PaliGemmaCheckpointLoader:
    """Validate portable artifacts without importing JAX until weights restore."""

    @staticmethod
    def inspect(root: str | Path) -> PaliGemmaCheckpointManifest:
        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise CheckpointValidationError("checkpoint root must not be a symlink")
        resolved = requested.resolve()
        if not resolved.is_dir():
            raise CheckpointValidationError(f"checkpoint root is not a directory: {resolved}")
        path = resolved / CANONICAL_MANIFEST
        _regular_file(path, "checkpoint manifest")
        return PaliGemmaCheckpointManifest.from_file(path)

    @classmethod
    def verify_artifact(
        cls,
        root: str | Path,
        *,
        verify_hashes: bool = True,
        restore_hook: RestoreHook | None = None,
    ) -> tuple[
        Path,
        PaliGemmaCheckpointManifest,
        PaliGemmaAdaLoopConfig,
        PaliGemmaRuntimeContract,
        Path,
    ]:
        """Verify all files and metadata internal to a public artifact.

        This is useful for offline inspection.  It does *not* claim that the
        external base/tokenizer are present; :meth:`from_pretrained` performs
        those mandatory identity checks before reconstruction.
        """

        root_path, manifest, recipe, runtime, overlay_path = cls._verify_metadata(
            root, verify_hashes=verify_hashes
        )
        _restore_overlay(overlay_path, recipe, manifest.tensors, restore_hook)
        return root_path, manifest, recipe, runtime, overlay_path

    @classmethod
    def _verify_metadata(
        cls, root: str | Path, *, verify_hashes: bool
    ) -> tuple[
        Path,
        PaliGemmaCheckpointManifest,
        PaliGemmaAdaLoopConfig,
        PaliGemmaRuntimeContract,
        Path,
    ]:
        if not isinstance(verify_hashes, bool):
            raise TypeError("verify_hashes must be a bool")
        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise CheckpointValidationError("checkpoint root must not be a symlink")
        root_path = requested.resolve()
        manifest = cls.inspect(root_path)
        paths = {
            record.role: record.validate(root_path, verify_hashes=verify_hashes)
            for record in manifest.files
        }
        recipe = PaliGemmaAdaLoopConfig.from_file(paths["config"])
        if manifest.topology != TopologyRecord.from_recipe(recipe):
            raise CheckpointValidationError(
                "manifest topology contradicts adaloop_config.yaml"
            )
        runtime = PaliGemmaRuntimeContract.from_dict(
            _read_json(paths["runtime_contract"], "runtime contract")
        )
        runtime.validate_recipe(recipe)
        _validate_runtime_root_scopes(runtime, recipe)
        return root_path, manifest, recipe, runtime, paths["trainable_weights"]

    @classmethod
    def from_pretrained(
        cls,
        root: str | Path,
        *,
        base_checkpoint: str | Path,
        tokenizer: str | Path,
        verify_hashes: bool = True,
        reconstruct: bool = True,
        reconstruction_hook: PublicReconstructionHook | None = None,
        restore_hook: RestoreHook | None = None,
    ) -> VerifiedPaliGemmaCheckpoint:
        if not isinstance(reconstruct, bool):
            raise TypeError("reconstruct must be a bool")
        root_path, manifest, recipe, runtime, overlay_path = cls._verify_metadata(
            root, verify_hashes=verify_hashes
        )
        overlay = _restore_overlay(overlay_path, recipe, manifest.tensors, restore_hook)
        base_path = validate_external_asset(
            base_checkpoint, recipe.base_model, "base checkpoint"
        )
        tokenizer_path = validate_external_asset(
            tokenizer, recipe.tokenizer, "tokenizer"
        )
        verified = VerifiedPaliGemmaCheckpoint(
            root=root_path,
            recipe=recipe,
            manifest=manifest,
            runtime_contract=runtime,
            overlay_path=overlay_path,
            base_checkpoint=base_path,
            tokenizer=tokenizer_path,
            _restore_hook=restore_hook,
        )
        if reconstruct:
            reconstructed = verified._reconstruct(reconstruction_hook, overlay)
            object.__setattr__(verified, "reconstructed", reconstructed)
        return verified


def _file_record(role: str, path: Path) -> FileRecord:
    expected_path, expected_format = FILE_ROLES[role]
    if path.name != expected_path:
        raise AssertionError(f"internal canonical-path mismatch: {path}")
    return FileRecord(
        role=role,
        path=expected_path,
        format=expected_format,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _validate_destination(root: Path, *, overwrite: bool) -> None:
    if root.is_symlink():
        raise CheckpointValidationError("output directory must not be a symlink")
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"output destination is not a directory: {root}")
    if not root.exists() or next(root.iterdir(), None) is None:
        return
    if not overwrite:
        raise FileExistsError(
            f"output destination must be absent or an empty directory: {root}"
        )
    names = {path.name for path in root.iterdir()}
    if names - CANONICAL_FILES or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise CheckpointValidationError(
            "overwrite refuses a destination containing non-checkpoint files"
        )


def _export_root(output_dir: str | Path, *, overwrite: bool) -> Path:
    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise CheckpointValidationError("output directory must not be a symlink")
    root = requested.resolve()
    _validate_destination(root, overwrite=overwrite)
    return root


def check_export_destination(output_dir: str | Path) -> Path:
    """Check, creating nothing, that a fresh export to ``output_dir`` can start.

    The destination must not be a symlink and must be absent or an empty
    directory, and its nearest existing ancestor must accept the sibling
    staging directory that export writes first.  Returns the resolved path.
    """

    root = _export_root(output_dir, overwrite=False)
    # Export creates the missing parents; probe the nearest existing ancestor.
    ancestor = root.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise NotADirectoryError(f"{ancestor} is not a directory")
    Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=ancestor)).rmdir()
    return root


def _publish_stage(stage: Path, root: Path, *, overwrite: bool) -> None:
    _validate_destination(root, overwrite=overwrite)
    if not root.exists():
        stage.replace(root)
        return
    if next(root.iterdir(), None) is None:
        root.rmdir()
        stage.replace(root)
        return
    backup = root.with_name(root.name + ".backup")
    if backup.exists():
        raise CheckpointValidationError(
            f"refusing overwrite because backup path already exists: {backup}"
        )
    root.replace(backup)
    try:
        stage.replace(root)
    except Exception:
        if not root.exists():
            backup.replace(root)
        raise
    else:
        shutil.rmtree(backup)


def _coerce_recipe(
    config: PaliGemmaAdaLoopConfig | str | Path,
) -> PaliGemmaAdaLoopConfig:
    if isinstance(config, PaliGemmaAdaLoopConfig):
        # Round-trip through the strict parser so hand-constructed dataclasses
        # receive the same cross-field validation as YAML recipes.
        return PaliGemmaAdaLoopConfig.from_dict(config.to_dict())
    return PaliGemmaAdaLoopConfig.from_file(config)


def _compact_full_parameter_tree(
    params: Mapping[str, Any],
    recipe: PaliGemmaAdaLoopConfig,
    base_path: Path,
    tokenizer_path: Path,
    reconstruction_hook: BaseReconstructionHook,
) -> tuple[Mapping[str, Any], tuple[TensorRecord, ...]]:
    params = _mapping(params, "full parameter tree")
    try:
        template_tree = reconstruction_hook(recipe, base_path, tokenizer_path)
    except Exception as exc:
        raise ReconstructionError(f"base reconstruction hook failed: {exc}") from exc
    template_tree = _mapping(template_tree, "reconstructed parameter template")
    trained_flat = _flatten_tree(params)
    template_flat = _flatten_tree(template_tree)
    trainable = _expected_trainable_paths(trained_flat, recipe)
    _expected_trainable_paths(template_flat, recipe)
    _verify_frozen_leaves(trained_flat, template_flat, trainable)
    overlay = _unflatten_tree({path: trained_flat[path] for path in trainable})
    _, inventory = _validate_overlay_tree(overlay, recipe)
    return overlay, inventory


def _write_portable_artifact(
    *,
    recipe: PaliGemmaAdaLoopConfig,
    overlay: Mapping[str, Any],
    inventory: tuple[TensorRecord, ...],
    base_path: Path,
    tokenizer_path: Path,
    output_dir: str | Path,
    restore_hook: RestoreHook | None,
    serialize_hook: SerializeHook | None,
    overwrite: bool,
) -> PaliGemmaCheckpointManifest:
    serializer = serialize_hook or _default_serialize
    try:
        overlay_bytes = serializer(overlay)
    except Exception as exc:
        raise CheckpointValidationError(f"could not serialize compact overlay: {exc}") from exc
    if not isinstance(overlay_bytes, bytes) or not overlay_bytes:
        raise CheckpointValidationError("serialize_hook must return non-empty bytes")

    output_root = _export_root(output_dir, overwrite=overwrite)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        config_output = stage / CANONICAL_CONFIG
        runtime_output = stage / CANONICAL_RUNTIME
        weights_output = stage / CANONICAL_WEIGHTS
        config_output.write_text(recipe.to_yaml(), encoding="utf-8")
        runtime = PaliGemmaRuntimeContract.from_recipe(
            recipe, _runtime_root_scopes(recipe)
        )
        runtime_output.write_bytes(_json_bytes(runtime.to_dict()))
        weights_output.write_bytes(overlay_bytes)
        files = tuple(
            _file_record(role, stage / expected[0])
            for role, expected in sorted(FILE_ROLES.items())
        )
        manifest = PaliGemmaCheckpointManifest(
            format_version=FORMAT_VERSION,
            backend=BACKEND,
            files=files,
            topology=TopologyRecord.from_recipe(recipe),
            tensors=inventory,
        )
        # Completion marker is written last.
        (stage / CANONICAL_MANIFEST).write_bytes(_json_bytes(manifest.to_dict()))
        PaliGemmaCheckpointLoader.from_pretrained(
            stage,
            base_checkpoint=base_path,
            tokenizer=tokenizer_path,
            verify_hashes=True,
            reconstruct=False,
            restore_hook=restore_hook,
        )
        _publish_stage(stage, output_root, overwrite=overwrite)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return PaliGemmaCheckpointLoader.inspect(output_root)


def export_parameter_tree(
    params: Mapping[str, Any],
    config: PaliGemmaAdaLoopConfig | str | Path,
    base_checkpoint: str | Path,
    tokenizer: str | Path,
    output_dir: str | Path,
    *,
    reconstruction_hook: BaseReconstructionHook,
    restore_hook: RestoreHook | None = None,
    serialize_hook: SerializeHook | None = None,
    overwrite: bool = False,
) -> PaliGemmaCheckpointManifest:
    """Export a live full training parameter tree as a compact checkpoint.

    It intentionally omits optimizer/cursor state: callers may continue to save
    their private training-resume state separately, while this artifact is the
    strict portable inference and new-training initialization checkpoint.
    """

    if not callable(reconstruction_hook):
        raise TypeError("reconstruction_hook must be callable")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    recipe = _coerce_recipe(config)
    base_path = validate_external_asset(
        base_checkpoint, recipe.base_model, "base checkpoint"
    )
    tokenizer_path = validate_external_asset(tokenizer, recipe.tokenizer, "tokenizer")
    overlay, inventory = _compact_full_parameter_tree(
        params, recipe, base_path, tokenizer_path, reconstruction_hook
    )
    return _write_portable_artifact(
        recipe=recipe,
        overlay=overlay,
        inventory=inventory,
        base_path=base_path,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        restore_hook=restore_hook,
        serialize_hook=serialize_hook,
        overwrite=overwrite,
    )


__all__ = [
    "CANONICAL_CONFIG",
    "CANONICAL_MANIFEST",
    "CANONICAL_RUNTIME",
    "CANONICAL_WEIGHTS",
    "PaliGemmaCheckpointLoader",
    "PaliGemmaCheckpointManifest",
    "PaliGemmaRuntimeContract",
    "VerifiedPaliGemmaCheckpoint",
    "check_export_destination",
    "export_parameter_tree",
    "read_overlay_headers",
    "validate_external_asset",
]
