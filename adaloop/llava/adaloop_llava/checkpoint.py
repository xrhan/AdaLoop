"""Portable LLaVA-1.5 AdaLoop checkpoints: export, conversion, validation and loading.

A portable checkpoint is a compact overlay on the pinned stage-2 base and CLIP tower. Every path
in it is relative to the checkpoint root:

.. code-block:: text

    checkpoint/
    ├── adaloop_manifest.json             written last; the completion marker
    ├── adaloop_config.yaml               the resolved recipe; local paths are recorded as null
    ├── adaloop_runtime.json              inference contract (conversation, image, condition, decode)
    ├── config.json                       LlavaConfig with the loop_* keys (role model_config)
    ├── adapter_config.json               PEFT LoRA config
    ├── adapter_model.safetensors         LoRA tensors of every language-model layer
    └── non_lora_trainables.safetensors   projector (4 tensors) + loop core (0 / 48 / 52 / 66 tensors)

The manifest records the size and SHA-256 of these six files, the identity of the base model and
of the CLIP tower with the size and SHA-256 of each of their files, the topology (variant, loop
mode, half-open span, training depth, step embedding, readout, deep supervision, question
condition, vision geometry), the LoRA settings and the exact sidecar inventory (name, shape and
dtype of every tensor).

* :func:`convert_adapter_checkpoint` converts an adapter directory written by the training code
  (``adapter_config.json``, ``adapter_model.safetensors``, ``non_lora_trainables.bin`` and
  ``config.json``), after comparing its ``config.json`` and ``adapter_config.json`` field by
  field with a recipe. :func:`export_release_checkpoint` is the same conversion applied to a
  fresh training output; ``adaloop-llava-train`` calls it at the end of a run.
* :class:`LlavaCheckpointLoader` validates a portable checkpoint (manifest, hashes, cross-checks
  of the manifest, recipe, runtime contract and ``config.json``, exact tensor inventories,
  identity of the supplied base and CLIP directories) and loads it with
  :func:`adaloop_llava.model.loading.load_for_eval`.

Validation does not import torch: the tensor inventories are read from the safetensors headers.
Loading and conversion import torch, transformers and peft when they run.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from .config import (
    BACKEND,
    DATA_LOCATION_KEYS,
    VARIANT_FOR_LOOP,
    VARIANTS,
    Config,
    build_dataclass,
    flatten_keys,
    lm_lora_target_regex,
    load_config,
    resolve_loop,
    to_dict,
    validate,
)
from .errors import CheckpointValidationError, ConfigValidationError
from .model.loop_utils import LOOP_CONFIG_DEFAULTS, TRAJ_DRAFT_TOKENS, TRAJ_SEPARATOR_TOKEN_ID, infer_loop_mode

#: Version of the portable checkpoint format (manifest ``format_version``).
FORMAT_VERSION = 1

MANIFEST_FILE = "adaloop_manifest.json"
RECIPE_FILE = "adaloop_config.yaml"
RUNTIME_FILE = "adaloop_runtime.json"
MODEL_CONFIG_FILE = "config.json"
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"
SIDECAR_FILE = "non_lora_trainables.safetensors"
#: The sidecar as the training code writes it (a ``torch.save`` file).
SIDECAR_BIN_FILE = "non_lora_trainables.bin"
#: Training writes this next to the adapter; :func:`export_release_checkpoint` reads the recipe
#: from it when none is given.
RESOLVED_RECIPE_FILE = "resolved_config.yaml"
#: Where ``adaloop-llava-train`` exports the portable checkpoint inside the run directory.
RELEASE_SUBDIR = "release"

#: Manifest role of each checkpoint file.
FILE_ROLES: Dict[str, str] = {
    RECIPE_FILE: "recipe",
    RUNTIME_FILE: "runtime",
    MODEL_CONFIG_FILE: "model_config",
    ADAPTER_CONFIG_FILE: "adapter_config",
    ADAPTER_WEIGHTS_FILE: "adapter_weights",
    SIDECAR_FILE: "non_lora_weights",
}
#: Files that the loaders would read instead of the canonical ones; a portable checkpoint must not
#: contain them (``load_for_eval`` prefers a ``non_lora_trainables.bin`` when one exists).
SHADOWING_FILES = (SIDECAR_BIN_FILE, "adapter_model.bin")

#: Decoding of the portable inference API, the same as the paper evaluation harness: greedy, at
#: most 32 new tokens, no repetition penalty, left padding, fp16 language model and CLIP tower
#: with the loop core kept in fp32.
DECODE_POLICY: Dict[str, Any] = {
    "strategy": "greedy",
    "max_new_tokens": 32,
    "repetition_penalty": None,
    "padding_side": "left",
    "language_model_dtype": "float16",
    "loop_core_dtype": "float32",
}

#: Files of the stage-2 base fingerprinted in a manifest (besides the weights): required ones, and
#: ones recorded when present. The slow tokenizer reads the three tokenizer files.
BASE_REQUIRED_FILES = ("config.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")
BASE_OPTIONAL_FILES = ("generation_config.json", "added_tokens.json")
#: Files of the CLIP tower fingerprinted in a manifest (besides the weights).
CLIP_REQUIRED_FILES = ("config.json", "preprocessor_config.json")

#: PEFT options that change which modules are adapted or how the adapters act. The training code
#: keeps PEFT's defaults; absent keys take them.
_DEFAULT_LORA_OPTIONS: Dict[str, Any] = {
    "alpha_pattern": {},
    "exclude_modules": None,
    "fan_in_fan_out": False,
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
    "use_qalora": False,
    "use_rslora": False,
}

#: Language-model architecture keys that a checkpoint's ``config.json`` must share with the base.
_LM_ARCH_KEYS = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                 "num_attention_heads", "num_key_value_heads", "head_dim", "mm_hidden_size",
                 "mm_projector_type", "max_position_embeddings", "rms_norm_eps", "rope_theta",
                 "tie_word_embeddings")

_SAFETENSORS_DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
                       "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
                       "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}
_FLOAT_DTYPES = ("float32", "float16", "bfloat16", "float64")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_CHUNK = 1 << 20

PathLike = Union[str, "os.PathLike[str]"]


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def _fail(message: str) -> None:
    raise CheckpointValidationError(message)


def sha256_file(path: PathLike) -> str:
    """SHA-256 of a file's bytes (lowercase hex)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(path: Path) -> Dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _read_json(path: Path, what: str) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"could not read {what} {path.name}: {exc}") from exc


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{context} must be a mapping")
    return value


def _strict_keys(raw: Mapping[str, Any], *, context: str, required: Sequence[str],
                 optional: Sequence[str] = ()) -> None:
    unknown = sorted(set(raw) - set(required) - set(optional))
    missing = sorted(set(required) - set(raw))
    if unknown:
        _fail(f"{context} has unknown field(s): {', '.join(unknown)}")
    if missing:
        _fail(f"{context} is missing field(s): {', '.join(missing)}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def portable_relative_path(value: Any, context: str) -> str:
    """``value`` as a portable POSIX path relative to a root (no absolute path, ``~``, ``..``,
    backslash or empty component); raises :class:`CheckpointValidationError` otherwise."""
    if not isinstance(value, str) or not value:
        _fail(f"{context} must be a non-empty relative path")
    path = PurePosixPath(value)
    if (path.is_absolute() or value.startswith("~") or "\\" in value or re.match(r"^[A-Za-z]:", value)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        _fail(f"{context} must be a portable path relative to the checkpoint root, got {value!r}")
    return value


_LOCAL_PATH = re.compile(r"^(?:/|~|\.{1,2}/|[A-Za-z]:[\\/]|\\\\|[A-Za-z][A-Za-z0-9+.-]*://)")


def local_path_values(data: Any, prefix: str = "") -> List[str]:
    """Dotted keys of the string values of ``data`` that look like local or remote file paths
    (absolute, home-relative, ``./`` or ``../`` relative, or a URI such as ``gs://...``)."""
    found: List[str] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            found += local_path_values(value, f"{prefix}{key}.")
    elif isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            found += local_path_values(value, f"{prefix}{index}.")
    elif isinstance(data, str) and _LOCAL_PATH.match(data.strip()):
        found.append(prefix[:-1] or "<value>")
    return found


def _require_no_local_paths(data: Any, what: str) -> None:
    keys = local_path_values(data)
    if keys:
        _fail(f"{what} records local paths ({', '.join(keys[:5])}); a portable checkpoint must not")


# ----------------------------------------------------------------------------------------------
# Safetensors headers (no torch)
# ----------------------------------------------------------------------------------------------

def read_safetensors_inventory(path: PathLike) -> Dict[str, Dict[str, Any]]:
    """``{name: {"shape": [...], "dtype": "float32"}}`` from a safetensors file's header."""
    path = Path(path)
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            length = struct.unpack("<Q", prefix)[0] if len(prefix) == 8 else 0
            raw = handle.read(length) if 0 < length <= size - 8 else None
    except OSError as exc:
        raise CheckpointValidationError(f"could not read {path.name}: {exc}") from exc
    if raw is None:
        _fail(f"{path.name} is not a safetensors file (header length {length}, file size {size})")
    try:
        header = json.loads(raw)
    except ValueError as exc:
        raise CheckpointValidationError(f"{path.name}: malformed safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        _fail(f"{path.name}: malformed safetensors header")
    header.pop("__metadata__", None)
    out: Dict[str, Dict[str, Any]] = {}
    end = 0
    for name, entry in header.items():
        if not isinstance(entry, dict) or entry.get("dtype") not in _SAFETENSORS_DTYPES:
            _fail(f"{path.name}: tensor {name!r} has an unsupported header entry {entry!r}")
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if (not isinstance(shape, list) or not all(_is_int(x) and x >= 0 for x in shape)
                or not isinstance(offsets, list) or len(offsets) != 2 or not all(_is_int(x) for x in offsets)):
            _fail(f"{path.name}: tensor {name!r} has a malformed shape or offsets")
        end = max(end, offsets[1])
        out[name] = {"shape": list(shape), "dtype": _SAFETENSORS_DTYPES[entry["dtype"]]}
    if 8 + length + end != size:
        _fail(f"{path.name}: the tensor data does not match the file size")
    return out


# ----------------------------------------------------------------------------------------------
# External assets: the stage-2 base and the CLIP tower
# ----------------------------------------------------------------------------------------------

def _is_weight_file(name: str) -> bool:
    return name.endswith((".safetensors", ".bin"))


def _sharded(directory: Path, index_name: str) -> List[str]:
    index = _read_json(directory / index_name, "weight index")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        _fail(f"{directory.name}/{index_name} has no weight_map")
    shards = sorted({str(v) for v in weight_map.values()})
    for shard in shards:
        portable_relative_path(shard, f"{index_name} shard")
    return [index_name] + shards


def asset_files(directory: PathLike, kind: str) -> List[str]:
    """The files of a base (``kind="base"``) or CLIP (``kind="vision_tower"``) directory that a
    manifest fingerprints: configuration, tokenizer or preprocessor files, and the weights that
    Transformers loads, in its order of preference (safetensors, single file before a sharded
    index; the CLIP tower may also be a ``pytorch_model.bin`` checkpoint)."""
    directory = Path(directory)
    if not directory.is_dir():
        _fail(f"{kind} directory {directory} does not exist")
    if kind == "base":
        required, optional = BASE_REQUIRED_FILES, BASE_OPTIONAL_FILES
        weight_layouts = (("model.safetensors", False), ("model.safetensors.index.json", True))
    elif kind == "vision_tower":
        required, optional = CLIP_REQUIRED_FILES, ()
        weight_layouts = (("model.safetensors", False), ("model.safetensors.index.json", True),
                          ("pytorch_model.bin", False), ("pytorch_model.bin.index.json", True))
    else:
        raise ValueError(f"unknown asset kind {kind!r}")
    missing = [f for f in required if not (directory / f).is_file()]
    if missing:
        _fail(f"{kind} directory {directory} lacks {', '.join(missing)}")
    files = list(required) + [f for f in optional if (directory / f).is_file()]
    for name, sharded in weight_layouts:
        if (directory / name).is_file():
            files += _sharded(directory, name) if sharded else [name]
            break
    else:
        _fail(f"{kind} directory {directory} has no weights "
              f"({' / '.join(name for name, _ in weight_layouts)})")
    return sorted(set(files))


def fingerprint_asset(directory: PathLike, kind: str) -> Dict[str, Dict[str, Any]]:
    """``{relative path: {"size_bytes", "sha256"}}`` of :func:`asset_files` (hashes every byte)."""
    directory = Path(directory)
    out = {}
    for name in asset_files(directory, kind):
        path = directory / name
        if not path.is_file():
            _fail(f"{kind} directory {directory} lacks {name}")
        out[name] = _digest(path)
    return out


def _validate_file_digests(raw: Any, context: str) -> Dict[str, Dict[str, Any]]:
    raw = _mapping(raw, context)
    if not raw:
        _fail(f"{context} must list at least one file")
    out = {}
    for name, entry in raw.items():
        portable_relative_path(name, f"{context} entry")
        entry = _mapping(entry, f"{context}[{name!r}]")
        _strict_keys(entry, context=f"{context}[{name!r}]", required=("size_bytes", "sha256"))
        if not _is_int(entry["size_bytes"]) or entry["size_bytes"] < 0:
            _fail(f"{context}[{name!r}].size_bytes must be a non-negative integer")
        if not isinstance(entry["sha256"], str) or not _SHA256.fullmatch(entry["sha256"]):
            _fail(f"{context}[{name!r}].sha256 must be 64 lowercase hexadecimal characters")
        out[name] = {"size_bytes": entry["size_bytes"], "sha256": entry["sha256"]}
    return out


def verify_asset(files: Mapping[str, Mapping[str, Any]], directory: PathLike, kind: str, *,
                 hash_small: bool = True, hash_large: bool = False) -> None:
    """Check a base or CLIP directory against the fingerprint of a manifest.

    The directory must hold exactly the fingerprinted files (:func:`asset_files`), each with the
    recorded size; the SHA-256 of the small files (configuration, tokenizer, preprocessor) is
    checked with ``hash_small`` and that of the weight files with ``hash_large``."""
    directory = Path(directory)
    found = set(asset_files(directory, kind))
    recorded = set(files)
    if found != recorded:
        _fail(f"the {kind} directory {directory} does not hold the files the checkpoint was made with: "
              f"missing {sorted(recorded - found)}, unexpected {sorted(found - recorded)}")
    for name in sorted(recorded):
        path = directory / name
        if not path.is_file():
            _fail(f"{kind} file {name} is missing from {directory}")
        size = path.stat().st_size
        if size != files[name]["size_bytes"]:
            _fail(f"{kind} file {name} has {size} bytes, the checkpoint was made with {files[name]['size_bytes']}")
        if hash_large if _is_weight_file(name) else hash_small:
            digest = sha256_file(path)
            if digest != files[name]["sha256"]:
                _fail(f"SHA-256 mismatch for {kind} file {name}: expected {files[name]['sha256']}, got {digest}")


# ----------------------------------------------------------------------------------------------
# Recipe-derived contracts
# ----------------------------------------------------------------------------------------------

def topology_from_recipe(cfg: Config) -> Dict[str, Any]:
    """The manifest ``topology`` block of a recipe (loop fields are null for SFT)."""
    loop, vis = cfg.model.loop, cfg.model.vision
    looped = loop.mode != "none"
    return {
        "variant": cfg.variant,
        "loop_mode": loop.mode,
        "loop_span": [int(loop.band[0]), int(loop.band[1])] if looped else None,
        "train_depth": int(loop.steps) if looped else 1,
        "step_embed": loop.step_embed if looped else None,
        "readout": loop.readout if looped else None,
        "fused_readout": bool(loop.fused_readout) if looped else None,
        "deep_sup_weight": loop.deep_sup_weight if looped else None,
        "condition": ({"include_bos": True, "max_tokens": int(cfg.data.condition_max_length),
                       "dim": int(loop.condition_dim)} if looped else None),
        "adaln_hidden_dim": int(loop.adaln_hidden_dim) if looped else None,
        "step_embed_max_steps": int(loop.step_embed_max_steps) if looped else None,
        "vision": {"input_res": int(vis.input_res), "spatial_pool": int(vis.spatial_pool),
                   "select_layer": int(vis.select_layer), "select_feature": vis.select_feature},
        "projector": cfg.model.projector,
    }


def lora_from_recipe(cfg: Config) -> Dict[str, Any]:
    """The manifest ``lora`` block of a recipe."""
    lo = cfg.lora
    return {"r": int(lo.r), "alpha": int(lo.alpha), "dropout": float(lo.dropout), "bias": lo.bias,
            "target_modules": list(lo.target_modules), "scope": lo.target_scope}


def runtime_from_recipe(cfg: Config) -> Dict[str, Any]:
    """The ``adaloop_runtime.json`` contract of a recipe."""
    vis = cfg.model.vision
    return {
        "format_version": FORMAT_VERSION,
        "backend": BACKEND,
        "conversation": cfg.model.conv_template,
        "model_max_length": int(cfg.model.model_max_length),
        "image": {"input_res": int(vis.input_res), "spatial_pool": int(vis.spatial_pool),
                  "aspect_ratio": cfg.data.image_aspect_ratio, "select_layer": int(vis.select_layer),
                  "select_feature": vis.select_feature},
        "condition": (None if cfg.model.loop.mode == "none" else
                      {"include_bos": True, "max_tokens": int(cfg.data.condition_max_length)}),
        "decode": dict(DECODE_POLICY),
    }


def portable_recipe_dict(cfg: Config) -> Dict[str, Any]:
    """The recipe as ``adaloop_config.yaml`` stores it: every key, with the local paths
    (``paths.*`` and the dataset locations ``data.visro_root``, ``data.replay_jsonl``,
    ``data.replay_image_root``) recorded as null and a resume directory in ``train.resume``
    recorded as ``auto``."""
    data = to_dict(cfg)
    data["paths"] = {key: None for key in data["paths"]}
    for key in DATA_LOCATION_KEYS:
        data["data"][key] = None
    if data["train"]["resume"] not in ("auto", "never"):
        data["train"]["resume"] = "auto"
    _require_no_local_paths(data, "the recipe")
    return data


def load_checkpoint_recipe(path: PathLike) -> Config:
    """Read and validate a checkpoint's ``adaloop_config.yaml`` (see :func:`portable_recipe_dict`)."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise CheckpointValidationError(f"could not read {path.name}: {exc}") from exc
    raw = dict(_mapping(raw, path.name))
    paths = _mapping(raw.get("paths"), f"{path.name} paths")
    data = _mapping(raw.get("data"), f"{path.name} data")
    recorded = {f"paths.{k}": v for k, v in paths.items()}
    recorded.update({f"data.{k}": data.get(k) for k in DATA_LOCATION_KEYS})
    local = sorted(k for k, v in recorded.items() if v is not None)
    if local:
        _fail(f"{path.name} records local paths ({', '.join(local)}); they must be null")
    reference = portable_recipe_dict(Config())
    if flatten_keys(raw) != flatten_keys(reference):
        missing = sorted(flatten_keys(reference) - flatten_keys(raw))
        extra = sorted(flatten_keys(raw) - flatten_keys(reference))
        _fail(f"{path.name} must list every recipe key: missing {missing[:5]}, unknown {extra[:5]}")
    raw["paths"] = {}
    try:
        cfg = build_dataclass(Config, raw)
        resolve_loop(cfg)
        validate(cfg)
    except ConfigValidationError as exc:
        raise CheckpointValidationError(f"{path.name}: {exc}") from exc
    return cfg


def _as_recipe(recipe: Union[Config, PathLike]) -> Config:
    if isinstance(recipe, Config):
        cfg = copy.deepcopy(recipe)
        try:
            resolve_loop(cfg)
            validate(cfg)
        except ConfigValidationError as exc:
            raise CheckpointValidationError(f"recipe: {exc}") from exc
        return cfg
    return load_config(str(recipe))


# ----------------------------------------------------------------------------------------------
# The model config.json and the PEFT adapter config
# ----------------------------------------------------------------------------------------------

def _expected_model_fields(cfg: Config) -> Dict[str, Any]:
    """``config.json`` values implied by a recipe (keys under their internal names)."""
    m, loop, vis = cfg.model, cfg.model.loop, cfg.model.vision
    out: Dict[str, Any] = {
        "loop_mode": loop.mode,
        "mm_projector_type": m.projector,
        "mm_vision_select_layer": int(vis.select_layer),
        "mm_vision_select_feature": vis.select_feature,
        "vision_input_res": int(vis.input_res),
        "vision_spatial_pool": int(vis.spatial_pool),
        "image_aspect_ratio": cfg.data.image_aspect_ratio,
        "mm_use_im_start_end": bool(m.mm_use_im_start_end),
        "mm_use_im_patch_token": bool(m.mm_use_im_patch_token),
    }
    if loop.mode != "none":
        out.update(
            loop_start=int(loop.band[0]), loop_end=int(loop.band[1]), loop_steps=int(loop.steps),
            loop_step_embed=loop.step_embed, loop_deep_sup_weight=loop.deep_sup_weight,
            loop_fused_readout=bool(loop.fused_readout), loop_readout_mode=loop.readout,
            loop_use_adaln=bool(loop.use_adaln), condition_max_length=int(cfg.data.condition_max_length),
            loop_condition_dim=int(loop.condition_dim), loop_adaln_hidden_dim=int(loop.adaln_hidden_dim),
            loop_step_embed_max_steps=int(loop.step_embed_max_steps))
        if loop.readout == "answer_trajectory":
            out.update(loop_traj_text=True, loop_draft_max_new_tokens=TRAJ_DRAFT_TOKENS,
                       loop_traj_sep_token_id=TRAJ_SEPARATOR_TOKEN_ID)
    return out


def model_config_fields(conf: Mapping[str, Any]) -> Dict[str, Any]:
    """The topology a ``config.json`` describes, read the way the model code reads it (absent loop
    keys take the model code's defaults; the loop mode comes from ``infer_loop_mode``)."""
    try:
        mode = infer_loop_mode(types.SimpleNamespace(**conf))
    except (ValueError, TypeError) as exc:
        raise CheckpointValidationError(f"config.json: {exc}") from exc
    out: Dict[str, Any] = {
        "loop_mode": mode,
        "mm_projector_type": conf.get("mm_projector_type"),
        "mm_vision_select_layer": conf.get("mm_vision_select_layer"),
        "mm_vision_select_feature": conf.get("mm_vision_select_feature") or "patch",
        "vision_input_res": conf.get("vision_input_res"),
        "vision_spatial_pool": conf.get("vision_spatial_pool"),
        "image_aspect_ratio": conf.get("image_aspect_ratio"),
        "mm_use_im_start_end": bool(conf.get("mm_use_im_start_end", False)),
        "mm_use_im_patch_token": bool(conf.get("mm_use_im_patch_token", False)),
    }
    if mode != "none":
        out["loop_start"] = conf.get("loop_start")
        out["loop_end"] = conf.get("loop_end")
        out["loop_deep_sup_weight"] = conf.get("loop_deep_sup_weight") or ""
        for key, default in LOOP_CONFIG_DEFAULTS.items():
            value = conf.get(key)
            out[key] = default if value is None else value
        for key in ("loop_traj_text", "loop_draft_max_new_tokens", "loop_traj_sep_token_id"):
            out[key] = conf.get(key)
    return out


def _same(got: Any, want: Any) -> bool:
    if isinstance(want, bool):
        return isinstance(got, bool) and got == want
    if _is_int(want):
        if isinstance(got, bool):
            return False
        try:
            return int(str(got)) == want
        except ValueError:
            return False
    if isinstance(want, str) and isinstance(got, str):
        return got.strip().lower() == want.strip().lower()
    return got == want


def check_model_config(conf: Mapping[str, Any], cfg: Config, *, context: str = "config.json") -> None:
    """Compare a ``config.json`` with a recipe field by field: loop mode, band, steps, step
    embedding, deep supervision, fused readout, readout, AdaLN use, condition length and
    dimensions, step-embedding size, vision geometry, projector and image-token settings."""
    got = model_config_fields(conf)
    mismatches = [f"{key}: {context} has {got.get(key)!r}, the recipe has {want!r}"
                  for key, want in _expected_model_fields(cfg).items() if not _same(got.get(key), want)]
    tml = conf.get("tokenizer_model_max_length")
    if tml is not None and not _same(tml, int(cfg.model.model_max_length)):
        mismatches.append(f"tokenizer_model_max_length: {context} has {tml!r}, "
                          f"the recipe has {cfg.model.model_max_length!r}")
    if conf.get("model_type") != "llava_llama":
        mismatches.append(f"model_type: {context} has {conf.get('model_type')!r}, expected 'llava_llama'")
    if mismatches:
        _fail(f"{context} does not match the recipe ({cfg.variant}): " + "; ".join(mismatches))


def _lm_value(conf: Mapping[str, Any], key: str) -> Any:
    if key == "num_key_value_heads":
        return conf.get(key) or conf.get("num_attention_heads")
    if key == "head_dim":
        heads, hidden = conf.get("num_attention_heads"), conf.get("hidden_size")
        return conf.get(key) or (int(hidden) // int(heads) if heads and hidden else None)
    return conf.get(key)


def check_lm_architecture(conf: Mapping[str, Any], base_conf: Mapping[str, Any]) -> None:
    """The checkpoint's ``config.json`` and the base's must describe the same language model and
    projector (keys missing from either side are not compared)."""
    mismatches = []
    for key in _LM_ARCH_KEYS:
        a, b = _lm_value(conf, key), _lm_value(base_conf, key)
        if a is not None and b is not None and a != b:
            mismatches.append(f"{key}: checkpoint {a!r}, base {b!r}")
    if mismatches:
        _fail("the checkpoint's config.json and the base model's config.json describe different "
              "models: " + "; ".join(mismatches))


def check_vision_width(conf: Mapping[str, Any], clip_conf: Mapping[str, Any]) -> None:
    """``mm_hidden_size`` must be the CLIP tower's width (when the CLIP config states it)."""
    vision = clip_conf.get("vision_config") if isinstance(clip_conf.get("vision_config"), dict) else clip_conf
    width = vision.get("hidden_size")
    if width is not None and conf.get("mm_hidden_size") is not None and int(width) != int(conf["mm_hidden_size"]):
        _fail(f"the CLIP tower has width {width}, the checkpoint was trained with mm_hidden_size "
              f"{conf['mm_hidden_size']}")


def check_adapter_config(acfg: Mapping[str, Any], cfg: Config, *, context: str = "adapter_config.json") -> None:
    """Compare a PEFT adapter config with the recipe's LoRA settings: type, rank, alpha, dropout,
    bias, target modules (the training code's regex, or the list of projection names older adapter
    directories record) and PEFT's defaults for every option that changes the adapted modules."""
    lo = cfg.lora
    problems = []
    if acfg.get("peft_type") != "LORA":
        problems.append(f"peft_type {acfg.get('peft_type')!r} (expected 'LORA')")
    if acfg.get("task_type") not in (None, "CAUSAL_LM"):
        problems.append(f"task_type {acfg.get('task_type')!r} (expected 'CAUSAL_LM')")
    for key, want in (("r", int(lo.r)), ("lora_alpha", int(lo.alpha))):
        if not _same(acfg.get(key), want):
            problems.append(f"{key} {acfg.get(key)!r} (recipe {want!r})")
    dropout = acfg.get("lora_dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or float(dropout) != float(lo.dropout):
        problems.append(f"lora_dropout {dropout!r} (recipe {lo.dropout!r})")
    if acfg.get("bias", "none") != lo.bias:
        problems.append(f"bias {acfg.get('bias')!r} (recipe {lo.bias!r})")
    targets = acfg.get("target_modules")
    if isinstance(targets, str):
        ok = targets == lm_lora_target_regex(lo.target_modules)
    elif isinstance(targets, (list, tuple)):
        ok = sorted(targets) == sorted(lo.target_modules) and len(set(targets)) == len(targets)
    else:
        ok = False
    if not ok:
        problems.append(f"target_modules {targets!r} (recipe {list(lo.target_modules)}, "
                        f"regex {lm_lora_target_regex(lo.target_modules)!r})")
    for key, default in _DEFAULT_LORA_OPTIONS.items():
        if key in acfg and acfg[key] != default and not (default == {} and acfg[key] is None):
            problems.append(f"{key} {acfg[key]!r} (only the PEFT default {default!r} is supported)")
    if problems:
        _fail(f"{context} does not match the recipe's LoRA settings: " + "; ".join(problems))


# ----------------------------------------------------------------------------------------------
# Tensor inventories
# ----------------------------------------------------------------------------------------------

def _lm_dims(conf: Mapping[str, Any]) -> Tuple[int, int, int, int, int, int]:
    try:
        hidden, heads = int(conf["hidden_size"]), int(conf["num_attention_heads"])
        kv_heads = int(conf.get("num_key_value_heads") or heads)
        head_dim = int(conf.get("head_dim") or hidden // heads)
        return hidden, heads, kv_heads, head_dim, int(conf["intermediate_size"]), int(conf["num_hidden_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"config.json lacks the language-model dimensions: {exc}") from exc


def expected_adapter_inventory(conf: Mapping[str, Any], cfg: Config) -> Dict[str, Tuple[int, ...]]:
    """Names and shapes of the LoRA tensors: one A/B pair for each target projection of every
    language-model layer, under the names PEFT saves."""
    hidden, heads, kv_heads, head_dim, ffn, layers = _lm_dims(conf)
    r = int(cfg.lora.r)
    io = {"q_proj": ("self_attn", hidden, heads * head_dim), "k_proj": ("self_attn", hidden, kv_heads * head_dim),
          "v_proj": ("self_attn", hidden, kv_heads * head_dim), "o_proj": ("self_attn", heads * head_dim, hidden),
          "gate_proj": ("mlp", hidden, ffn), "up_proj": ("mlp", hidden, ffn), "down_proj": ("mlp", ffn, hidden)}
    out: Dict[str, Tuple[int, ...]] = {}
    for layer in range(layers):
        for target in cfg.lora.target_modules:
            block, fan_in, fan_out = io[target]
            prefix = f"base_model.model.model.layers.{layer}.{block}.{target}"
            out[f"{prefix}.lora_A.weight"] = (r, fan_in)
            out[f"{prefix}.lora_B.weight"] = (fan_out, r)
    return out


def expected_sidecar_inventory(conf: Mapping[str, Any], cfg: Config) -> Dict[str, Tuple[int, ...]]:
    """Names and shapes of the non-LoRA trainables: the ``mlp2x_gelu`` projector and, for loop
    recipes, the loop core (one AdaLN generator per band layer, the question pooler, with the
    sinusoidal step embedding the step embedder, and for Variant C the draft feedback), under the
    names the training code saves."""
    try:
        hidden, vision = int(conf["hidden_size"]), int(conf["mm_hidden_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"config.json lacks hidden_size / mm_hidden_size: {exc}") from exc
    lm_hidden = int(conf.get("lm_hidden_size") or hidden)
    p = "base_model.model.model."
    out: Dict[str, Tuple[int, ...]] = {
        p + "mm_projector.0.weight": (hidden, vision), p + "mm_projector.0.bias": (hidden,),
        p + "mm_projector.2.weight": (hidden, hidden), p + "mm_projector.2.bias": (hidden,),
    }
    loop = cfg.model.loop
    if loop.mode == "none":
        return out
    cd, ah = int(loop.condition_dim), int(loop.adaln_hidden_dim)
    core = p + "loop_core."
    for i in range(int(loop.band[0]), int(loop.band[1])):
        a = f"{core}adaln.{i}."
        out.update({a + "condition_norm.weight": (cd,), a + "condition_norm.bias": (cd,),
                    a + "condition_proj.weight": (ah, cd), a + "condition_proj.bias": (ah,),
                    a + "to_adaln.weight": (6 * vision, ah), a + "to_adaln.bias": (6 * vision,)})
    mlp_hidden = max(1, int(round(cd * 2.0)))
    poolers = [core + "question_conditioner."]
    if loop.readout == "answer_trajectory":
        poolers.append(core + "draft_feedback.answer_pooler.")
        out.update({core + "draft_feedback.gate.weight": (1, 2 * cd), core + "draft_feedback.gate.bias": (1,)})
    for q in poolers:
        out.update({q + "query": (1, cd), q + "input_norm.weight": (lm_hidden,),
                    q + "q_proj.weight": (cd, cd), q + "k_proj.weight": (cd, cd), q + "v_proj.weight": (cd, cd),
                    q + "out_proj.weight": (cd, cd), q + "mlp_norm.weight": (cd,),
                    q + "mlp.0.weight": (mlp_hidden, cd), q + "mlp.0.bias": (mlp_hidden,),
                    q + "mlp.2.weight": (cd, mlp_hidden), q + "mlp.2.bias": (cd,)})
        if lm_hidden != cd:
            out[q + "input_proj.weight"] = (cd, lm_hidden)
    if loop.step_embed == "sinusoidal":
        s = core + "loop_step_embedder.mlp."
        out.update({s + "0.weight": (cd, 64), s + "0.bias": (cd,), s + "2.weight": (cd, cd), s + "2.bias": (cd,)})
    return out


def check_inventory(found: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Sequence[int]],
                    what: str) -> None:
    """``found`` (``{name: {"shape", "dtype"}}``) must hold exactly the ``expected`` names and
    shapes, in floating-point dtypes."""
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    shaped = sorted(n for n in set(found) & set(expected) if list(found[n]["shape"]) != list(expected[n]))
    typed = sorted(n for n, v in found.items() if v["dtype"] not in _FLOAT_DTYPES)
    if missing or extra or shaped or typed:
        _fail(f"{what} does not hold exactly the tensors of the recipe's model: missing {missing[:4]}, "
              f"unexpected {extra[:4]}, mis-shaped {shaped[:4]}, non-floating {typed[:4]}")


# ----------------------------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------------------------

_TOPOLOGY_KEYS = ("variant", "loop_mode", "loop_span", "train_depth", "step_embed", "readout", "fused_readout",
                  "deep_sup_weight", "condition", "adaln_hidden_dim", "step_embed_max_steps", "vision", "projector")


@dataclass(frozen=True)
class CheckpointManifest:
    """A validated ``adaloop_manifest.json`` (see the module docstring)."""

    format_version: int
    backend: str
    variant: str
    base_model: Dict[str, Any]
    vision_tower: Dict[str, Any]
    topology: Dict[str, Any]
    lora: Dict[str, Any]
    sidecar: Dict[str, Dict[str, Any]]
    files: Dict[str, Dict[str, Any]]

    @classmethod
    def from_dict(cls, raw: Any) -> "CheckpointManifest":
        raw = _mapping(raw, "checkpoint manifest")
        fields = ("format_version", "backend", "variant", "base_model", "vision_tower", "topology", "lora",
                  "sidecar", "files")
        _strict_keys(raw, context="checkpoint manifest", required=fields)
        if raw["format_version"] != FORMAT_VERSION:
            _fail(f"format_version must be {FORMAT_VERSION}, got {raw['format_version']!r}")
        if raw["backend"] != BACKEND:
            _fail(f"backend must be {BACKEND!r}, got {raw['backend']!r}")
        variant = raw["variant"]
        if variant not in VARIANTS:
            _fail(f"variant must be one of {VARIANTS}, got {variant!r}")

        base = _mapping(raw["base_model"], "base_model")
        _strict_keys(base, context="base_model", required=("id", "format", "files"))
        if not isinstance(base["id"], str) or not base["id"] or not isinstance(base["format"], str):
            _fail("base_model.id and base_model.format must be strings")
        _require_no_local_paths({"id": base["id"]}, "base_model.id")
        base_files = _validate_file_digests(base["files"], "base_model.files")
        vision = _mapping(raw["vision_tower"], "vision_tower")
        _strict_keys(vision, context="vision_tower", required=("id", "revision", "files"))
        if not isinstance(vision["id"], str) or not vision["id"]:
            _fail("vision_tower.id must be a string")
        _require_no_local_paths({"id": vision["id"]}, "vision_tower.id")
        if not isinstance(vision["revision"], str) or not _REVISION.fullmatch(vision["revision"]):
            _fail("vision_tower.revision must be an immutable 40-64 character lowercase hexadecimal revision")
        vision_files = _validate_file_digests(vision["files"], "vision_tower.files")

        topology = dict(_mapping(raw["topology"], "topology"))
        _strict_keys(topology, context="topology", required=_TOPOLOGY_KEYS)
        expected_mode, expected_readout = next(k for k, v in VARIANT_FOR_LOOP.items() if v == variant)
        if topology["variant"] != variant or topology["loop_mode"] != expected_mode:
            _fail(f"topology variant/loop_mode {topology['variant']!r}/{topology['loop_mode']!r} contradict "
                  f"variant {variant!r} (expected loop mode {expected_mode!r})")
        if expected_mode != "none" and topology["readout"] != expected_readout:
            _fail(f"variant {variant!r} requires readout {expected_readout!r}, got {topology['readout']!r}")
        span = topology["loop_span"]
        if expected_mode == "none":
            if span is not None or topology["train_depth"] != 1:
                _fail("an sft checkpoint has loop_span null and train_depth 1")
        else:
            if not (isinstance(span, list) and len(span) == 2 and all(_is_int(x) for x in span)
                    and 0 <= span[0] < span[1]):
                _fail(f"topology.loop_span must be a half-open [start, end), got {span!r}")
            depth = topology["train_depth"]
            if not _is_int(depth) or depth < 1 or (expected_mode == "single_pass") != (depth == 1):
                _fail(f"topology.train_depth {depth!r} contradicts loop mode {expected_mode!r}")
            step_embed = {"single_pass": "none", "adaloop": "sinusoidal"}[expected_mode]
            if topology["step_embed"] != step_embed:
                _fail(f"variant {variant!r} requires step_embed {step_embed!r}, got {topology['step_embed']!r}")
            condition = _mapping(topology["condition"], "topology.condition")
            _strict_keys(condition, context="topology.condition", required=("include_bos", "max_tokens", "dim"))
            if condition["include_bos"] is not True:
                _fail("the LLaVA question condition starts with BOS (topology.condition.include_bos true)")
        vis = _mapping(topology["vision"], "topology.vision")
        _strict_keys(vis, context="topology.vision", required=("input_res", "spatial_pool", "select_layer",
                                                               "select_feature"))

        lora = dict(_mapping(raw["lora"], "lora"))
        _strict_keys(lora, context="lora", required=("r", "alpha", "dropout", "bias", "target_modules", "scope"))

        sidecar = {}
        for name, entry in _mapping(raw["sidecar"], "sidecar").items():
            if not isinstance(name, str) or not name:
                _fail("sidecar tensor names must be non-empty strings")
            entry = _mapping(entry, f"sidecar[{name!r}]")
            _strict_keys(entry, context=f"sidecar[{name!r}]", required=("shape", "dtype"))
            if not isinstance(entry["shape"], list) or not all(_is_int(x) and x >= 0 for x in entry["shape"]):
                _fail(f"sidecar[{name!r}].shape must be a list of non-negative integers")
            if entry["dtype"] not in _FLOAT_DTYPES:
                _fail(f"sidecar[{name!r}].dtype must be one of {_FLOAT_DTYPES}, got {entry['dtype']!r}")
            sidecar[name] = {"shape": list(entry["shape"]), "dtype": entry["dtype"]}

        files_raw = _mapping(raw["files"], "files")
        if set(files_raw) != set(FILE_ROLES):
            _fail(f"files must list exactly {sorted(FILE_ROLES)}, got {sorted(files_raw)}")
        files = {}
        for name, entry in files_raw.items():
            portable_relative_path(name, "files entry")
            entry = _mapping(entry, f"files[{name!r}]")
            _strict_keys(entry, context=f"files[{name!r}]", required=("role", "size_bytes", "sha256"))
            if entry["role"] != FILE_ROLES[name]:
                _fail(f"files[{name!r}].role must be {FILE_ROLES[name]!r}, got {entry['role']!r}")
            digest = _validate_file_digests({name: {"size_bytes": entry["size_bytes"], "sha256": entry["sha256"]}},
                                            "files")[name]
            files[name] = {"role": entry["role"], **digest}
        return cls(format_version=FORMAT_VERSION, backend=BACKEND, variant=variant,
                   base_model={"id": base["id"], "format": base["format"], "files": base_files},
                   vision_tower={"id": vision["id"], "revision": vision["revision"], "files": vision_files},
                   topology=topology, lora=lora, sidecar=sidecar, files=files)

    @classmethod
    def from_file(cls, path: PathLike) -> "CheckpointManifest":
        return cls.from_dict(_read_json(Path(path), "manifest"))

    def to_dict(self) -> Dict[str, Any]:
        return {"format_version": self.format_version, "backend": self.backend, "variant": self.variant,
                "base_model": copy.deepcopy(self.base_model), "vision_tower": copy.deepcopy(self.vision_tower),
                "topology": copy.deepcopy(self.topology), "lora": copy.deepcopy(self.lora),
                "sidecar": copy.deepcopy(self.sidecar), "files": copy.deepcopy(self.files)}

    @property
    def train_depth(self) -> int:
        return int(self.topology["train_depth"])

    def validate_files(self, root: PathLike, *, verify_hashes: bool = True) -> None:
        """Every checkpoint file exists inside ``root`` as a regular file with the recorded size
        (and SHA-256 with ``verify_hashes``); no file that would shadow a canonical one exists."""
        root = Path(root).resolve()
        for name, entry in sorted(self.files.items()):
            path = root / name
            if path.is_symlink() or not path.is_file():
                _fail(f"checkpoint file {name} is missing (or not a regular file)")
            if root not in path.resolve().parents:
                _fail(f"checkpoint file {name} escapes the checkpoint root")
            size = path.stat().st_size
            if size != entry["size_bytes"]:
                _fail(f"checkpoint file {name} has {size} bytes, the manifest records {entry['size_bytes']}")
            if verify_hashes:
                digest = sha256_file(path)
                if digest != entry["sha256"]:
                    _fail(f"SHA-256 mismatch for {name}: expected {entry['sha256']}, got {digest}")
        shadowing = [name for name in SHADOWING_FILES if (root / name).exists()]
        if shadowing:
            _fail(f"the checkpoint contains {', '.join(shadowing)}, which the loaders would read instead of "
                  "the files the manifest covers")


# ----------------------------------------------------------------------------------------------
# Verification and loading
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifiedLlavaCheckpoint:
    """A portable checkpoint whose own files passed :meth:`LlavaCheckpointLoader.verify`."""

    root: Path
    manifest: CheckpointManifest
    recipe: Config
    runtime: Dict[str, Any]
    model_config: Dict[str, Any]


@dataclass
class LoadedLlavaCheckpoint:
    """The result of :meth:`LlavaCheckpointLoader.from_pretrained`."""

    model: Any
    tokenizer: Any
    image_processor: Any
    recipe: Config
    runtime: Dict[str, Any]
    manifest: CheckpointManifest
    checkpoint_dir: Path


def _cross_check(manifest: CheckpointManifest, recipe: Config, runtime: Any, conf: Mapping[str, Any],
                 acfg: Mapping[str, Any]) -> None:
    problems = []
    if manifest.variant != recipe.variant:
        problems.append(f"variant: manifest {manifest.variant!r}, recipe {recipe.variant!r}")
    for key, want in (("id", recipe.base_model.id), ("format", recipe.base_model.format)):
        if manifest.base_model[key] != want:
            problems.append(f"base_model.{key}: manifest {manifest.base_model[key]!r}, recipe {want!r}")
    for key, want in (("id", recipe.vision_tower.id), ("revision", recipe.vision_tower.revision)):
        if manifest.vision_tower[key] != want:
            problems.append(f"vision_tower.{key}: manifest {manifest.vision_tower[key]!r}, recipe {want!r}")
    topology = topology_from_recipe(recipe)
    for key in _TOPOLOGY_KEYS:
        if manifest.topology.get(key) != topology[key]:
            problems.append(f"topology.{key}: manifest {manifest.topology.get(key)!r}, recipe {topology[key]!r}")
    lora = lora_from_recipe(recipe)
    if manifest.lora != lora:
        problems.append(f"lora: manifest {manifest.lora!r}, recipe {lora!r}")
    if problems:
        _fail("checkpoint manifest and adaloop_config.yaml disagree: " + "; ".join(problems))
    if runtime != runtime_from_recipe(recipe):
        _fail(f"{RUNTIME_FILE} does not match the recipe: got {runtime!r}, expected {runtime_from_recipe(recipe)!r}")
    check_model_config(conf, recipe)
    if conf.get("mm_vision_tower") != recipe.vision_tower.id:
        _fail(f"config.json mm_vision_tower must be the CLIP Hub id {recipe.vision_tower.id!r}, "
              f"got {conf.get('mm_vision_tower')!r}")
    if "_name_or_path" in conf:
        _fail("config.json must not record _name_or_path")
    _require_no_local_paths(conf, "config.json")
    check_adapter_config(acfg, recipe)
    if acfg.get("target_modules") != lm_lora_target_regex(recipe.lora.target_modules):
        _fail(f"adapter_config.json target_modules must be the language-model regex "
              f"{lm_lora_target_regex(recipe.lora.target_modules)!r}, got {acfg.get('target_modules')!r}")
    if acfg.get("base_model_name_or_path") != recipe.base_model.id:
        _fail(f"adapter_config.json base_model_name_or_path must be {recipe.base_model.id!r}, "
              f"got {acfg.get('base_model_name_or_path')!r}")
    _require_no_local_paths(acfg, "adapter_config.json")


class LlavaCheckpointLoader:
    """Validate and load portable LLaVA-1.5 AdaLoop checkpoints (local directories)."""

    @staticmethod
    def inspect(checkpoint: PathLike) -> CheckpointManifest:
        """Parse and validate the manifest only (no file is hashed)."""
        root = Path(checkpoint)
        path = root / MANIFEST_FILE
        if not path.is_file():
            _fail(f"{root} is not a portable checkpoint (no {MANIFEST_FILE}); adapter directories "
                  "are converted with adaloop-llava-convert")
        return CheckpointManifest.from_file(path)

    @staticmethod
    def verify(checkpoint: PathLike, *, verify_hashes: bool = True) -> VerifiedLlavaCheckpoint:
        """Validate a checkpoint's own files, without torch:

        1. the manifest, its portable paths and the size (and, with ``verify_hashes``, SHA-256)
           of every file;
        2. the topology, LoRA settings and identity blocks of the manifest against
           ``adaloop_config.yaml``; ``adaloop_runtime.json`` against the recipe; the ``loop_*`` and
           vision keys of ``config.json`` and the PEFT settings of ``adapter_config.json`` against
           the recipe;
        3. the exact tensor inventories of ``adapter_model.safetensors`` and
           ``non_lora_trainables.safetensors`` (the latter also against the manifest's dtypes).
        """
        root = Path(checkpoint).resolve()
        manifest = LlavaCheckpointLoader.inspect(root)
        manifest.validate_files(root, verify_hashes=verify_hashes)
        recipe = load_checkpoint_recipe(root / RECIPE_FILE)
        runtime = _read_json(root / RUNTIME_FILE, "runtime contract")
        conf = _read_json(root / MODEL_CONFIG_FILE, "model config")
        acfg = _read_json(root / ADAPTER_CONFIG_FILE, "adapter config")
        if not isinstance(conf, dict) or not isinstance(acfg, dict):
            _fail("config.json and adapter_config.json must hold JSON objects")
        _cross_check(manifest, recipe, runtime, conf, acfg)

        adapter = read_safetensors_inventory(root / ADAPTER_WEIGHTS_FILE)
        check_inventory(adapter, expected_adapter_inventory(conf, recipe), ADAPTER_WEIGHTS_FILE)
        sidecar = read_safetensors_inventory(root / SIDECAR_FILE)
        check_inventory(sidecar, expected_sidecar_inventory(conf, recipe), SIDECAR_FILE)
        if sidecar != manifest.sidecar:
            differ = sorted(n for n in set(sidecar) | set(manifest.sidecar)
                            if sidecar.get(n) != manifest.sidecar.get(n))
            _fail(f"{SIDECAR_FILE} does not match the manifest's sidecar inventory: {differ[:4]}")
        return VerifiedLlavaCheckpoint(root=root, manifest=manifest, recipe=recipe, runtime=runtime,
                                       model_config=conf)

    @staticmethod
    def verify_external(verified: VerifiedLlavaCheckpoint, *, base_model: PathLike, vision_tower: PathLike,
                        verify_hashes: bool = True, verify_base_hashes: bool = False) -> None:
        """Check the supplied base and CLIP directories against the manifest: the same files with
        the same sizes, the SHA-256 of the small files (unless ``verify_hashes`` is false) and, with
        ``verify_base_hashes``, of the weights too; and the same language-model architecture and
        vision width as the checkpoint's ``config.json``."""
        m = verified.manifest
        small = verify_hashes or verify_base_hashes
        verify_asset(m.base_model["files"], base_model, "base", hash_small=small, hash_large=verify_base_hashes)
        verify_asset(m.vision_tower["files"], vision_tower, "vision_tower", hash_small=small,
                     hash_large=verify_base_hashes)
        check_lm_architecture(verified.model_config, _read_json(Path(base_model) / "config.json", "base config"))
        check_vision_width(verified.model_config, _read_json(Path(vision_tower) / "config.json", "CLIP config"))

    @classmethod
    def from_pretrained(cls, checkpoint: PathLike, *, base_model: PathLike, vision_tower: PathLike,
                        device: str = "cuda", dtype: Any = "float16", attn_implementation: Optional[str] = None,
                        verify_hashes: bool = True, verify_base_hashes: bool = False,
                        with_tokenizer: bool = True) -> LoadedLlavaCheckpoint:
        """Verify a portable checkpoint (:meth:`verify`, :meth:`verify_external`) and load it with
        :func:`adaloop_llava.model.loading.load_for_eval`: base weights, the sidecar, the LoRA
        adapter merged into the weights, the CLIP tower, and the loop core reloaded in fp32.
        ``dtype`` (a ``torch.dtype`` or its name) is the precision of the language model and CLIP."""
        verified = cls.verify(checkpoint, verify_hashes=verify_hashes)
        cls.verify_external(verified, base_model=base_model, vision_tower=vision_tower,
                            verify_hashes=verify_hashes, verify_base_hashes=verify_base_hashes)
        import torch

        from .model.loading import load_for_eval
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"dtype must be a torch dtype or its name, got {dtype!r}")
        tokenizer, model, image_processor = load_for_eval(
            str(base_model), str(verified.root), str(vision_tower), dtype=torch_dtype, device=device,
            attn_implementation=attn_implementation, with_tokenizer=with_tokenizer)
        if model.config.loop_mode != verified.recipe.model.loop.mode:
            _fail(f"the loaded model has loop mode {model.config.loop_mode!r}, the recipe "
                  f"{verified.recipe.model.loop.mode!r}")
        return LoadedLlavaCheckpoint(model=model, tokenizer=tokenizer, image_processor=image_processor,
                                     recipe=verified.recipe, runtime=verified.runtime, manifest=verified.manifest,
                                     checkpoint_dir=verified.root)


# ----------------------------------------------------------------------------------------------
# Staged, atomic publication
# ----------------------------------------------------------------------------------------------

def _check_destination(root: Path, *, overwrite: bool) -> bool:
    """Whether ``root`` is a non-empty directory; refuses symlinks, files and (without
    ``overwrite``) non-empty directories."""
    if root.is_symlink():
        _fail(f"checkpoint destination must not be a symlink: {root}")
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"checkpoint destination is not a directory: {root}")
    nonempty = root.is_dir() and any(root.iterdir())
    if nonempty and not overwrite:
        raise FileExistsError(f"checkpoint destination is not empty: {root}; pass overwrite=True (--overwrite) "
                              "only to replace its checkpoint files")
    return nonempty


@contextlib.contextmanager
def _staged_output(output_dir: PathLike, *, overwrite: bool) -> Iterator[Tuple[Path, Path]]:
    """Yield ``(destination, staging directory)``; on success the staged files are published.

    An empty or missing destination is replaced by the staging directory in one rename. A
    non-empty one (with ``overwrite``) first loses its manifest, then receives the staged files,
    the manifest last, so a reader sees either a complete checkpoint or none; other files in it
    are kept.
    """
    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        _fail(f"checkpoint destination must not be a symlink: {requested}")
    root = requested.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    _check_destination(root, overwrite=overwrite)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent)))
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(stage, 0o777 & ~umask)          # mkdtemp creates the directory private (0700)
    try:
        yield root, stage
        if _check_destination(root, overwrite=overwrite):
            (root / MANIFEST_FILE).unlink(missing_ok=True)
            names = sorted(p.name for p in stage.iterdir() if p.name != MANIFEST_FILE)
            for name in names + [MANIFEST_FILE]:
                target = root / name
                if target.is_symlink() or target.is_dir():
                    _fail(f"cannot replace {target}: it is a symlink or a directory")
                os.replace(stage / name, target)
        else:
            if root.exists():
                root.rmdir()
            os.replace(stage, root)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


# ----------------------------------------------------------------------------------------------
# Conversion and export
# ----------------------------------------------------------------------------------------------

def _load_fingerprint(base_fingerprint: Union[None, PathLike, Mapping[str, Any]]) -> Dict[str, Any]:
    if base_fingerprint is None:
        return {}
    if isinstance(base_fingerprint, Mapping):
        raw = base_fingerprint
    else:
        raw = _read_json(Path(base_fingerprint), "base fingerprint")
    raw = _mapping(raw, "base fingerprint")
    out = {}
    if "base_model" in raw:
        block = _mapping(raw["base_model"], "base fingerprint base_model")
        out["base_model"] = {"id": block.get("id"), "format": block.get("format"),
                             "files": _validate_file_digests(block.get("files"), "base fingerprint base_model.files")}
    if "vision_tower" in raw:
        block = _mapping(raw["vision_tower"], "base fingerprint vision_tower")
        out["vision_tower"] = {"id": block.get("id"), "revision": block.get("revision"),
                               "files": _validate_file_digests(block.get("files"),
                                                               "base fingerprint vision_tower.files")}
    if not out:
        _fail("the base fingerprint has neither a base_model nor a vision_tower block")
    return out


def _asset_block(kind: str, directory: PathLike, identity: Dict[str, Any],
                 reused: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if reused is None:
        return {**identity, "files": fingerprint_asset(directory, kind)}
    for key, want in identity.items():
        if reused.get(key) != want:
            _fail(f"the base fingerprint's {kind} {key} is {reused.get(key)!r}, the recipe's {want!r}")
    verify_asset(reused["files"], directory, kind, hash_small=True, hash_large=False)
    return {**identity, "files": copy.deepcopy(dict(reused["files"]))}


def _write_sidecar(state: Mapping[str, Any], path: Path) -> Dict[str, Dict[str, Any]]:
    """Write the sidecar tensors to safetensors under their own names and prove the file
    reloads to the same dtypes and values (``torch.equal``); returns its inventory."""
    import torch
    from safetensors.torch import load_file, save_file

    tensors = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            _fail(f"{SIDECAR_BIN_FILE} entry {name!r} is not a tensor")
        tensors[name] = value.detach().cpu().clone().contiguous()
    save_file(tensors, str(path), metadata={"format": "pt"})
    reloaded = load_file(str(path))
    if set(reloaded) != set(state):
        _fail(f"{path.name} does not reload with the same tensor names")
    for name, value in state.items():
        got = reloaded[name]
        if got.dtype != value.dtype or got.shape != value.shape or not torch.equal(got, value.detach().cpu()):
            _fail(f"{path.name}: tensor {name!r} does not reload bit-exactly")
    return {name: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
            for name, t in sorted(reloaded.items())}


def convert_adapter_checkpoint(adapter_dir: PathLike, recipe: Union[Config, PathLike], output_dir: PathLike, *,
                              base_model: PathLike, vision_tower: PathLike,
                              base_fingerprint: Union[None, PathLike, Mapping[str, Any]] = None,
                              overwrite: bool = False) -> Path:
    """Convert an adapter directory written by the training code into a portable checkpoint.

    ``adapter_dir`` holds ``adapter_config.json``, ``adapter_model.safetensors``,
    ``non_lora_trainables.bin`` and ``config.json``. Steps:

    1. ``config.json`` is read through :func:`~adaloop_llava.model.loading.normalize_model_config`
       (an absent ``loop_use_adaln`` means true; SFT's ``loop_start: null`` with an inert
       ``loop_steps: 3`` means no loop; ``loop_steps == 1`` means single pass) and compared field by
       field with ``recipe`` (:func:`check_model_config`); ``adapter_config.json`` is compared with
       the recipe's LoRA settings (:func:`check_adapter_config`).
    2. The base and CLIP directories are fingerprinted (size and SHA-256 of every file), or checked
       against ``base_fingerprint`` (a JSON file or mapping with ``base_model`` / ``vision_tower``
       blocks, e.g. the manifest of an earlier conversion against the same base), and their
       ``config.json`` must agree with the checkpoint's language-model architecture and width.
    3. ``non_lora_trainables.bin`` is loaded with ``torch.load(weights_only=True)``, checked
       against the exact inventory of the recipe's model, and written to
       ``non_lora_trainables.safetensors`` under the same names; every tensor must reload with
       the same dtype and ``torch.equal`` values. ``adapter_model.safetensors`` is copied byte for
       byte after its inventory check.
    4. ``config.json`` is written with ``mm_vision_tower`` set to the CLIP Hub id, without
       ``_name_or_path``, with ``model_type: llava_llama`` (the paper runs recorded ``llama`` for
       their ``LlavaLlamaForCausalLM``), and with the keys ``normalize_model_config`` resolves
       made explicit;
       ``adapter_config.json`` with ``base_model_name_or_path`` set to the base id and
       ``target_modules`` as the training code's language-model regex. (Adapter directories of
       the paper runs list projection names, which PEFT also matches against the question
       pooler's ``q/k/v_proj`` and wraps with LoRA weights the adapter file does not hold; their
       ``B`` is zero, so merging them changes nothing, and the loop core is reloaded from the
       sidecar anyway. With the regex the loader builds exactly the adapted modules.)
    5. Everything is hashed into the manifest, written last, and the staged checkpoint is verified
       (:meth:`LlavaCheckpointLoader.verify`) before it is published atomically.

    A non-empty ``output_dir`` is refused unless ``overwrite``. Returns the manifest path.
    """
    src = Path(adapter_dir).resolve()
    cfg = _as_recipe(recipe)
    if Path(output_dir).expanduser().resolve() == src:
        _fail("the output directory must differ from the adapter directory")
    for name in (MODEL_CONFIG_FILE, ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE, SIDECAR_BIN_FILE):
        if not (src / name).is_file():
            _fail(f"{src} lacks {name}; an adapter directory holds {MODEL_CONFIG_FILE}, {ADAPTER_CONFIG_FILE}, "
                  f"{ADAPTER_WEIGHTS_FILE} and {SIDECAR_BIN_FILE}")

    # 1. config.json and adapter_config.json against the recipe
    from .model.loading import normalize_model_config
    raw_conf = _read_json(src / MODEL_CONFIG_FILE, "model config")
    if not isinstance(raw_conf, dict):
        _fail("config.json must hold a JSON object")
    try:
        upgraded = normalize_model_config(types.SimpleNamespace(**raw_conf))
    except ValueError as exc:
        raise CheckpointValidationError(f"config.json: {exc}") from exc
    conf = dict(raw_conf)
    conf.pop("_name_or_path", None)
    if conf.get("model_type") == "llama" and conf.get("architectures") == ["LlavaLlamaForCausalLM"]:
        conf["model_type"] = "llava_llama"      # the paper runs saved their LLaVA config as 'llama'
    conf["mm_vision_tower"] = cfg.vision_tower.id
    conf["loop_mode"] = upgraded.loop_mode
    if upgraded.loop_mode != "none":
        conf["loop_use_adaln"] = upgraded.loop_use_adaln
        conf["loop_step_embed"] = upgraded.loop_step_embed
    check_model_config(conf, cfg, context=f"{src.name}/config.json")
    _require_no_local_paths(conf, "config.json (after mm_vision_tower and _name_or_path are replaced)")
    acfg = _read_json(src / ADAPTER_CONFIG_FILE, "adapter config")
    if not isinstance(acfg, dict):
        _fail("adapter_config.json must hold a JSON object")
    check_adapter_config(acfg, cfg, context=f"{src.name}/adapter_config.json")
    acfg = dict(acfg)
    acfg["base_model_name_or_path"] = cfg.base_model.id
    acfg["target_modules"] = lm_lora_target_regex(cfg.lora.target_modules)
    _require_no_local_paths(acfg, "adapter_config.json (after base_model_name_or_path is replaced)")

    # 2. external assets
    reused = _load_fingerprint(base_fingerprint)
    base_block = _asset_block("base", base_model, {"id": cfg.base_model.id, "format": cfg.base_model.format},
                              reused.get("base_model"))
    clip_block = _asset_block("vision_tower", vision_tower,
                              {"id": cfg.vision_tower.id, "revision": cfg.vision_tower.revision},
                              reused.get("vision_tower"))
    check_lm_architecture(conf, _read_json(Path(base_model) / "config.json", "base config"))
    check_vision_width(conf, _read_json(Path(vision_tower) / "config.json", "CLIP config"))

    # 3. tensors
    check_inventory(read_safetensors_inventory(src / ADAPTER_WEIGHTS_FILE), expected_adapter_inventory(conf, cfg),
                    f"{src.name}/{ADAPTER_WEIGHTS_FILE}")
    import torch
    state = torch.load(src / SIDECAR_BIN_FILE, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        _fail(f"{SIDECAR_BIN_FILE} must hold a dict of tensors")
    check_inventory({k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                     for k, v in state.items()}, expected_sidecar_inventory(conf, cfg),
                    f"{src.name}/{SIDECAR_BIN_FILE}")

    # 4.-5. stage, hash, verify, publish
    with _staged_output(output_dir, overwrite=overwrite) as (root, stage):
        sidecar = _write_sidecar(state, stage / SIDECAR_FILE)
        shutil.copyfile(src / ADAPTER_WEIGHTS_FILE, stage / ADAPTER_WEIGHTS_FILE)
        _write_json(stage / MODEL_CONFIG_FILE, conf)
        _write_json(stage / ADAPTER_CONFIG_FILE, acfg)
        with open(stage / RECIPE_FILE, "w", encoding="utf-8") as handle:
            yaml.safe_dump(portable_recipe_dict(cfg), handle, sort_keys=False)
        _write_json(stage / RUNTIME_FILE, runtime_from_recipe(cfg))
        manifest = {
            "format_version": FORMAT_VERSION,
            "backend": BACKEND,
            "variant": cfg.variant,
            "base_model": base_block,
            "vision_tower": clip_block,
            "topology": topology_from_recipe(cfg),
            "lora": lora_from_recipe(cfg),
            "sidecar": sidecar,
            "files": {name: {"role": role, **_digest(stage / name)} for name, role in sorted(FILE_ROLES.items())},
        }
        CheckpointManifest.from_dict(manifest)
        _write_json(stage / MANIFEST_FILE, manifest)
        LlavaCheckpointLoader.verify(stage, verify_hashes=True)
    return root / MANIFEST_FILE


def export_release_checkpoint(training_output_dir: PathLike, output_dir: Optional[PathLike] = None, *,
                              recipe: Union[None, Config, PathLike] = None, base_model: Optional[PathLike] = None,
                              vision_tower: Optional[PathLike] = None,
                              base_fingerprint: Union[None, PathLike, Mapping[str, Any]] = None,
                              overwrite: bool = False) -> Path:
    """Export a finished training run as a portable checkpoint: :func:`convert_adapter_checkpoint`
    applied to the run directory.

    ``recipe`` defaults to the run's ``resolved_config.yaml``, ``output_dir`` to
    ``<run>/release``, and the base and CLIP directories to the recipe's ``paths``. Returns the
    manifest path."""
    run = Path(training_output_dir)
    cfg = _as_recipe(recipe if recipe is not None else run / RESOLVED_RECIPE_FILE)
    destination = Path(output_dir) if output_dir is not None else run / RELEASE_SUBDIR
    return convert_adapter_checkpoint(run, cfg, destination,
                                     base_model=base_model if base_model is not None else cfg.paths.base_model,
                                     vision_tower=vision_tower if vision_tower is not None else cfg.paths.vision_tower,
                                     base_fingerprint=base_fingerprint, overwrite=overwrite)


__all__ = [
    "CheckpointManifest",
    "DECODE_POLICY",
    "FORMAT_VERSION",
    "LlavaCheckpointLoader",
    "LoadedLlavaCheckpoint",
    "MANIFEST_FILE",
    "VerifiedLlavaCheckpoint",
    "convert_adapter_checkpoint",
    "export_release_checkpoint",
    "fingerprint_asset",
    "load_checkpoint_recipe",
    "runtime_from_recipe",
    "topology_from_recipe",
]
