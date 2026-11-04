"""Recipe and evaluation-config schema of the LLaVA-1.5 backend.

* Typed dataclasses for training recipes (:class:`Config`) and paper evaluation configs
  (:class:`EvalConfig`). The shipped files (``configs/{sft,adaln,variant_a}.yaml`` and
  ``configs/eval/*.yaml``) are flat and list every key.
* A recipe starts with ``schema_version: 1``, ``backend: llava-1.5`` and ``variant``
  (``sft`` | ``adaln`` | ``a`` | ``b`` | ``c``), followed by the read-only identity blocks
  ``base_model`` and ``vision_tower``. The variant must agree with ``model.loop.mode`` and
  ``model.loop.readout`` (``sft`` <-> ``none``, ``adaln`` <-> ``single_pass``, and with mode
  ``adaloop``: ``a`` <-> readout ``last``, ``b`` <-> ``image_prefix``, ``c`` <->
  ``answer_trajectory``).
* Command-line overrides use dotted keys, ``model.loop.steps=3``; values are parsed with
  ``yaml.safe_load``.
* ``model.loop.mode`` (``none`` | ``single_pass`` | ``adaloop``) selects the model topology.
  :func:`resolve_loop` fills loop fields left ``null`` from the mode's preset and
  :func:`validate` rejects contradictions.
* Every validation error is a :class:`~adaloop_llava.errors.ConfigValidationError`, which is a
  ``ValueError``.

This module does not import torch or transformers.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

from .errors import ConfigValidationError
from .model.loop_utils import (LOOP_MODES, READOUT_MODES, TRAJ_DRAFT_TOKENS, TRAJ_SEPARATOR_TOKEN_ID,
                               check_mode_consistency, check_readout, clip_exec_layers, deep_supervision_weights)

#: Recipe schema of this backend (top-level ``schema_version`` and ``backend`` keys).
SCHEMA_VERSION = 1
BACKEND = "llava-1.5"

#: The recipe variant of each (loop mode, readout) pair.
VARIANT_FOR_LOOP: Dict[Tuple[str, str], str] = {
    ("none", "last"): "sft", ("single_pass", "last"): "adaln", ("adaloop", "last"): "a",
    ("adaloop", "image_prefix"): "b", ("adaloop", "answer_trajectory"): "c"}
VARIANTS = tuple(VARIANT_FOR_LOOP.values())

#: Identity of the stage-2 base and of the CLIP tower the recipes were trained on. The recipes
#: only name them; the files are pinned by size and sha256 in each portable checkpoint manifest.
BASE_MODEL_ID = "llava-v1.5-7b-448-s2"
BASE_MODEL_FORMAT = "llava_hf_safetensors"
VISION_TOWER_ID = "openai/clip-vit-large-patch14-336"
VISION_TOWER_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

#: Language-model projections LoRA may target (all linear layers of a LLaMA decoder block).
LORA_ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")

#: Values filled in by :func:`resolve_loop` for loop fields left ``null``.
LOOP_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {},
    "single_pass": {"steps": 1, "step_embed": "none", "fused_readout": False},
    "adaloop": {"steps": 3, "step_embed": "sinusoidal", "fused_readout": True},
}

ATTN_IMPLEMENTATIONS = ("flash_attention_2", "sdpa", "eager")
#: Values of ``train.loss_normalization`` (see ``LlavaLlamaForCausalLM.set_loss_normalization``).
LOSS_NORMALIZATIONS = ("rank_mean", "token_mean")


# ----------------------------------------------------------------------------------------------
# Training config
# ----------------------------------------------------------------------------------------------

@dataclass
class BaseModelIdentity:
    """The stage-2 LLaVA-1.5 base the recipe trains from (read-only identity)."""
    id: str = BASE_MODEL_ID
    format: str = BASE_MODEL_FORMAT


@dataclass
class VisionTowerIdentity:
    """The CLIP tower of the recipe: Hugging Face Hub id and immutable revision (read-only)."""
    id: str = VISION_TOWER_ID
    revision: str = VISION_TOWER_REVISION


@dataclass
class PathsConfig:
    base_model: str = "checkpoints/llava-v1.5-7b-448-s2"
    vision_tower: str = "checkpoints/clip-vit-large-patch14-336"
    output_root: str = "outputs"
    output_dir: Optional[str] = None        # null: <output_root>/<run_name>


#: ``data`` keys that name local files; portable checkpoints record them as null.
DATA_LOCATION_KEYS = ("visro_root", "replay_jsonl", "replay_image_root")


@dataclass
class DataConfig:
    visro_root: Optional[str] = "data/VisRO"
    tasks: str = "core6"
    difficulty: Optional[str] = None
    replay_jsonl: Optional[str] = "data/llava_replay/replay.jsonl"
    replay_image_root: Optional[str] = None
    data_limit: int = 0
    image_aspect_ratio: str = "pad"
    condition_max_length: int = 128


@dataclass
class VisionConfig:
    select_layer: int = -2
    select_feature: str = "patch"
    input_res: int = 448
    spatial_pool: int = 2


@dataclass
class LoopConfig:
    mode: str = "none"
    band: List[int] = field(default_factory=lambda: [2, 8])
    steps: Optional[int] = None
    step_embed: Optional[str] = None
    fused_readout: Optional[bool] = None
    readout: str = "last"
    deep_sup_weight: str = "geometric:0.6"
    use_adaln: bool = True
    condition_dim: int = 512
    adaln_hidden_dim: int = 128
    step_embed_max_steps: int = 16


@dataclass
class ModelConfig:
    conv_template: str = "v1"
    model_max_length: int = 2048
    attn_implementation: str = "flash_attention_2"
    vision: VisionConfig = field(default_factory=VisionConfig)
    projector: str = "mlp2x_gelu"
    loop: LoopConfig = field(default_factory=LoopConfig)
    mm_use_im_start_end: bool = False
    mm_use_im_patch_token: bool = False


@dataclass
class LoRAConfig:
    enable: bool = True
    r: int = 128
    alpha: int = 256
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: list(LORA_ATTN_TARGETS + LORA_MLP_TARGETS))
    target_scope: str = "language_model"


@dataclass
class OptimConfig:
    name: str = "adamw_torch"
    lr: float = 1.0e-4
    mm_projector_lr: float = 5.0e-5
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1.0e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03


@dataclass
class TrainConfig:
    seed: int = 42
    max_steps: int = 40000
    per_device_batch_size: int = 8
    grad_accum: int = 1
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 1000
    save_total_limit: int = 1
    save_safetensors: bool = True
    logging_steps: int = 1
    dataloader_num_workers: int = 4
    group_by_modality_length: bool = False
    loss_normalization: str = "rank_mean"
    rank_mean_scale: float = 8.0
    ddp_find_unused_parameters: bool = False
    resume: str = "auto"
    debug_grad_arrival: bool = False


@dataclass
class Config:
    schema_version: int = SCHEMA_VERSION
    backend: str = BACKEND
    variant: str = "sft"
    base_model: BaseModelIdentity = field(default_factory=BaseModelIdentity)
    vision_tower: VisionTowerIdentity = field(default_factory=VisionTowerIdentity)
    run_name: Optional[str] = None
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def output_dir(self) -> str:
        """``paths.output_dir`` when set, else ``<paths.output_root>/<run_name>``."""
        if self.paths.output_dir:
            return self.paths.output_dir
        if not self.run_name:
            raise ConfigValidationError("run_name is not set")
        return os.path.join(self.paths.output_root, self.run_name)


# ----------------------------------------------------------------------------------------------
# Evaluation config
# ----------------------------------------------------------------------------------------------

@dataclass
class EvalPathsConfig:
    base_model: str = "checkpoints/llava-v1.5-7b-448-s2"
    vision_tower: str = "checkpoints/clip-vit-large-patch14-336"


@dataclass
class EvalRunConfig:
    name: Optional[str] = None
    adapter: Optional[str] = None
    visro_root: str = "data/VisRO"
    tasks: str = "core6"
    splits: List[str] = field(default_factory=lambda: ["eval", "ood"])
    per_cell: int = 0
    prompt_style: str = "question"
    max_depth: int = 1
    p1_max_depth: int = 0
    dtype: str = "float16"
    attn_implementation: str = "sdpa"
    batch_size: int = 16
    max_new_tokens: int = 32
    do_sample: bool = False
    padding_side: str = "left"
    shard: str = "batch"
    output_dir: str = "outputs/eval/${eval.name}"


@dataclass
class EvalConfig:
    paths: EvalPathsConfig = field(default_factory=EvalPathsConfig)
    eval: EvalRunConfig = field(default_factory=EvalRunConfig)


# ----------------------------------------------------------------------------------------------
# YAML loading and overrides
# ----------------------------------------------------------------------------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def flatten_keys(d: Dict[str, Any], prefix: str = "") -> Set[str]:
    """Dotted paths of all leaves of a nested dict (an empty dict counts as a leaf)."""
    keys: Set[str] = set()
    for k, v in d.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            keys |= flatten_keys(v, path + ".")
        else:
            keys.add(path)
    return keys


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigValidationError(f"{path}: the top level of a config file must be a mapping")
    return data


def parse_overrides(overrides: Iterable[str]) -> Dict[str, Any]:
    """``["a.b=1", "c=[2, 8]"]`` -> nested dict; values are parsed with ``yaml.safe_load``."""
    out: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ConfigValidationError(f"Override {item!r} must look like key.path=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key or any(not part for part in key.split(".")):
            raise ConfigValidationError(f"Override {item!r} has an empty key")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigValidationError(f"Override {item!r} conflicts with another override")
        node[parts[-1]] = yaml.safe_load(value) if value.strip() else None
    return out


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """Check ``value`` against the annotation ``tp`` (with light, lossless coercions)."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            if type(None) in args:
                return None
            raise ConfigValidationError(f"{path}: must not be null")
        return _coerce(value, non_none[0], path)
    if value is None:
        raise ConfigValidationError(f"{path}: must not be null")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigValidationError(f"{path}: expected a mapping, got {value!r}")
        return build_dataclass(tp, value, path)
    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise ConfigValidationError(f"{path}: expected a list, got {value!r}")
        return [_coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ConfigValidationError(f"{path}: expected a mapping, got {value!r}")
        return {str(k): _coerce(v, args[1], f"{path}.{k}") for k, v in value.items()}
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigValidationError(f"{path}: expected true/false, got {value!r}")
    if tp is int:
        if isinstance(value, bool):
            raise ConfigValidationError(f"{path}: expected an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise ConfigValidationError(f"{path}: expected an integer, got {value!r}")
    if tp is float:
        if isinstance(value, bool):
            raise ConfigValidationError(f"{path}: expected a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):          # YAML 1.1 reads "1e-4" as a string
            try:
                return float(value)
            except ValueError:
                pass
        raise ConfigValidationError(f"{path}: expected a number, got {value!r}")
    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigValidationError(f"{path}: expected a string, got {value!r}")
    if tp is Any:
        return value
    raise TypeError(f"{path}: unsupported annotation {tp!r}")


def build_dataclass(cls, data: Dict[str, Any], path: str = ""):
    """Instantiate dataclass ``cls`` from a nested dict; unknown keys are errors."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        where = path or "<root>"
        raise ConfigValidationError(f"Unknown config key(s) under {where}: {unknown}")
    kwargs = {k: _coerce(v, hints[k], f"{path}.{k}" if path else k) for k, v in data.items()}
    return cls(**kwargs)


def to_dict(cfg) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def dump_config(cfg, path: str) -> None:
    """Write the fully resolved config (e.g. ``<output_dir>/resolved_config.yaml``)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)


# ----------------------------------------------------------------------------------------------
# Mode presets and validation
# ----------------------------------------------------------------------------------------------

def resolve_loop(cfg: Config) -> Config:
    """Fill loop fields left ``null`` from the preset of ``model.loop.mode`` (returns ``cfg``)."""
    loop = cfg.model.loop
    if loop.mode not in LOOP_PRESETS:
        raise ConfigValidationError(f"model.loop.mode must be one of {LOOP_MODES}, got {loop.mode!r}")
    for k, v in LOOP_PRESETS[loop.mode].items():
        if getattr(loop, k) is None:
            setattr(loop, k, v)
    return cfg


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigValidationError(msg)


def _checked(fn, *args):
    """Call a loop helper of :mod:`adaloop_llava.model.loop_utils`; its ``ValueError`` becomes a
    :class:`ConfigValidationError`."""
    try:
        return fn(*args)
    except ConfigValidationError:
        raise
    except ValueError as e:
        raise ConfigValidationError(str(e)) from e


def _check_loop_span(start: int, end: int) -> None:
    """The band as the shared topology contract: a zero-indexed, half-open
    :class:`adaloop_common.LoopSpan` (imported lazily)."""
    from adaloop_common import LoopSpan
    try:
        LoopSpan(start, end)
    except ValueError as e:
        raise ConfigValidationError(f"model.loop.band: {e}") from e


def validate(cfg: Config, *, explicit_keys: Iterable[str] = (), for_training: bool = True) -> Config:
    """Reject contradictory or unsupported settings. ``explicit_keys`` are the dotted keys set by
    command-line overrides."""
    m, loop, vis = cfg.model, cfg.model.loop, cfg.model.vision
    _require(loop.mode in LOOP_MODES, f"model.loop.mode must be one of {LOOP_MODES}, got {loop.mode!r}")

    # -- recipe identity --
    _require(cfg.schema_version == SCHEMA_VERSION,
             f"schema_version must be {SCHEMA_VERSION}, got {cfg.schema_version!r}")
    _require(cfg.backend == BACKEND, f"backend must be {BACKEND!r}, got {cfg.backend!r}")
    _require(cfg.variant in VARIANTS, f"variant must be one of {VARIANTS}, got {cfg.variant!r}")
    _require(loop.readout in READOUT_MODES,
             f"model.loop.readout must be one of {READOUT_MODES}, got {loop.readout!r}")
    _require(VARIANT_FOR_LOOP.get((loop.mode, loop.readout)) == cfg.variant,
             f"variant {cfg.variant!r} contradicts model.loop.mode {loop.mode!r} with readout {loop.readout!r} "
             "(sft <-> none, adaln <-> single_pass, a / b / c <-> adaloop with readout last / image_prefix / "
             "answer_trajectory)")
    _require(cfg.base_model == BaseModelIdentity(),
             f"base_model is a read-only identity block, expected {dataclasses.asdict(BaseModelIdentity())}, "
             f"got {dataclasses.asdict(cfg.base_model)}")
    _require(cfg.vision_tower == VisionTowerIdentity(),
             f"vision_tower is a read-only identity block, expected {dataclasses.asdict(VisionTowerIdentity())}, "
             f"got {dataclasses.asdict(cfg.vision_tower)}")
    _require(cfg.paths.output_dir is None or bool(str(cfg.paths.output_dir).strip()),
             "paths.output_dir must be null or a non-empty path")

    # -- the setting switch --
    if loop.mode == "none":
        stray = sorted(k for k in explicit_keys
                       if k.startswith("model.loop.") and k != "model.loop.mode")
        _require(not stray, f"model.loop.mode is 'none' but overrides set loop keys {stray}; "
                            "they would have no effect")
    else:
        _require(loop.steps is not None and loop.step_embed is not None and loop.fused_readout is not None,
                 "loop fields are unresolved; call resolve_loop() before validate()")
        _checked(check_mode_consistency, loop.mode, loop.steps, loop.step_embed, loop.fused_readout)
        _require(isinstance(loop.band, list) and len(loop.band) == 2,
                 f"model.loop.band must be [start, end], got {loop.band!r}")
        exec_layers = clip_exec_layers(vis.select_layer)
        s, e = loop.band
        _require(0 <= s < e <= exec_layers,
                 f"model.loop.band [{s},{e}) must satisfy 0 <= start < end <= {exec_layers} "
                 f"(the executed CLIP layers for select_layer={vis.select_layer})")
        _check_loop_span(s, e)
        _checked(check_readout, loop.mode, loop.readout, loop.deep_sup_weight, loop.fused_readout)
        if loop.readout == "last":
            _require(bool(str(loop.deep_sup_weight or "").strip()),
                     "model.loop.deep_sup_weight must be set for loop modes (training always uses the "
                     "deep-supervision path, also for one step)")
            _checked(deep_supervision_weights, loop.deep_sup_weight, loop.steps)
        else:
            _require(loop.step_embed == "sinusoidal",
                     f"model.loop.readout {loop.readout!r} requires step_embed 'sinusoidal', got {loop.step_embed!r}")
            _require(cfg.train.loss_normalization == "token_mean",
                     f"model.loop.readout {loop.readout!r} trains with one language-model pass whose loss is "
                     "the mean over the supervised tokens of the global batch; train.loss_normalization "
                     f"must be 'token_mean', got {cfg.train.loss_normalization!r}")
        _require(loop.use_adaln, "model.loop.use_adaln must be true (unconditioned loops are not supported)")
        _require(loop.condition_dim > 0 and loop.condition_dim % 8 == 0,
                 f"model.loop.condition_dim must be a positive multiple of 8, got {loop.condition_dim}")
        _require(loop.adaln_hidden_dim > 0, "model.loop.adaln_hidden_dim must be > 0")
        _require(loop.step_embed_max_steps >= 1, "model.loop.step_embed_max_steps must be >= 1")

    # -- model --
    _require(m.conv_template == "v1", f"model.conv_template must be 'v1', got {m.conv_template!r}")
    _require(m.projector == "mlp2x_gelu", f"model.projector must be 'mlp2x_gelu', got {m.projector!r}")
    _require(not m.mm_use_im_start_end and not m.mm_use_im_patch_token,
             "model.mm_use_im_start_end and model.mm_use_im_patch_token must be false")
    _require(m.attn_implementation in ATTN_IMPLEMENTATIONS,
             f"model.attn_implementation must be one of {ATTN_IMPLEMENTATIONS}, got {m.attn_implementation!r}")
    _require(m.model_max_length > 0, "model.model_max_length must be > 0")
    _require(vis.select_feature == "patch", f"model.vision.select_feature must be 'patch', got {vis.select_feature!r}")
    _require(vis.input_res > 0 and vis.input_res % 14 == 0,
             f"model.vision.input_res must be a positive multiple of 14, got {vis.input_res}")
    _require(vis.spatial_pool >= 1 and (vis.input_res // 14) % vis.spatial_pool == 0,
             f"model.vision.spatial_pool={vis.spatial_pool} must divide the {vis.input_res // 14}-patch grid")

    # -- data --
    _require(cfg.data.image_aspect_ratio == "pad",
             f"data.image_aspect_ratio must be 'pad', got {cfg.data.image_aspect_ratio!r}")
    _require(cfg.data.condition_max_length >= 1, "data.condition_max_length must be >= 1")
    _require(cfg.data.data_limit >= 0, "data.data_limit must be >= 0")
    _check_tasks(cfg.data.tasks, cfg.data.difficulty, "data")

    # -- LoRA --
    lo = cfg.lora
    if for_training:
        _require(lo.enable, "lora.enable must be true for training (full fine-tuning is not supported)")
    _require(lo.r > 0 and lo.alpha > 0, "lora.r and lora.alpha must be > 0")
    _require(0.0 <= lo.dropout < 1.0, f"lora.dropout must be in [0, 1), got {lo.dropout}")
    _require(lo.bias == "none", f"lora.bias must be 'none', got {lo.bias!r}")
    allowed = set(LORA_ATTN_TARGETS + LORA_MLP_TARGETS)
    _require(bool(lo.target_modules) and set(lo.target_modules) <= allowed
             and len(set(lo.target_modules)) == len(lo.target_modules),
             f"lora.target_modules must be distinct names from {sorted(allowed)}, got {lo.target_modules}")
    _require(lo.target_scope == "language_model",
             f"lora.target_scope must be 'language_model', got {lo.target_scope!r}")

    # -- optimiser --
    o = cfg.optim
    _require(o.name == "adamw_torch", f"optim.name must be 'adamw_torch', got {o.name!r}")
    _require(o.lr > 0 and o.mm_projector_lr > 0, "optim.lr and optim.mm_projector_lr must be > 0")
    _require(o.weight_decay >= 0, "optim.weight_decay must be >= 0")
    _require(len(o.betas) == 2 and all(0.0 <= b < 1.0 for b in o.betas), f"optim.betas invalid: {o.betas}")
    _require(o.eps > 0, "optim.eps must be > 0")
    _require(o.max_grad_norm >= 0, "optim.max_grad_norm must be >= 0")
    _require(0.0 <= o.warmup_ratio < 1.0, "optim.warmup_ratio must be in [0, 1)")

    # -- training loop --
    t = cfg.train
    _require(t.loss_normalization in LOSS_NORMALIZATIONS,
             f"train.loss_normalization must be one of {LOSS_NORMALIZATIONS}, got {t.loss_normalization!r}")
    _require(t.rank_mean_scale > 0, "train.rank_mean_scale must be > 0")
    _require(t.per_device_batch_size >= 1 and t.grad_accum >= 1, "batch size and grad_accum must be >= 1")
    _require(t.max_steps >= 1 and t.save_steps >= 1 and t.logging_steps >= 1,
             "train.max_steps / save_steps / logging_steps must be >= 1")
    _require(bool(t.resume), "train.resume must be 'auto', 'never' or a checkpoint directory")
    _require(not t.group_by_modality_length,
             "train.group_by_modality_length is not supported (training uses the default random sampler)")
    _require(t.dataloader_num_workers >= 0, "train.dataloader_num_workers must be >= 0")
    return cfg


def load_config(path: str, overrides: Sequence[str] = ()) -> Config:
    """Load, override, resolve and validate a training recipe.

    Keys set by ``overrides`` count as explicit for :func:`validate`; the file itself (a shipped
    flat recipe, a dumped resolved config) only provides values.
    """
    ov = parse_overrides(overrides)
    cfg = build_dataclass(Config, deep_merge(_read_yaml(path), ov))
    resolve_loop(cfg)
    validate(cfg, explicit_keys=flatten_keys(ov))
    return cfg


def load_eval_config(path: str, overrides: Sequence[str] = ()) -> EvalConfig:
    """Load and override an evaluation config; resolves ``${eval.name}`` in ``output_dir``."""
    raw = deep_merge(_read_yaml(path), parse_overrides(overrides))
    cfg = build_dataclass(EvalConfig, raw)
    ev = cfg.eval
    if "${eval.name}" in ev.output_dir:
        _require(bool(ev.name), "eval.output_dir uses ${eval.name} but eval.name is not set")
        ev.output_dir = ev.output_dir.replace("${eval.name}", ev.name)
    _require("${" not in ev.output_dir, f"Unsupported interpolation in eval.output_dir: {ev.output_dir!r}")
    _require(ev.max_depth >= 1, f"eval.max_depth must be >= 1, got {ev.max_depth}")
    _require(ev.p1_max_depth == 0 or 2 <= ev.p1_max_depth <= ev.max_depth,
             f"eval.p1_max_depth must be 0 (no p=1) or lie in [2, eval.max_depth={ev.max_depth}], "
             f"got {ev.p1_max_depth}")
    _require(bool(ev.splits) and all(s in ("eval", "ood") for s in ev.splits),
             f"eval.splits must list eval and/or ood, got {ev.splits}")
    _require(ev.per_cell >= 0, f"eval.per_cell must be >= 0, got {ev.per_cell}")
    _require(ev.prompt_style in ("question", "instruction"),
             f"eval.prompt_style must be 'question' or 'instruction', got {ev.prompt_style!r}")
    _check_tasks(ev.tasks, None, "eval")
    _require(ev.dtype in ("float16", "bfloat16", "float32"), f"eval.dtype invalid: {ev.dtype!r}")
    _require(ev.attn_implementation in ATTN_IMPLEMENTATIONS, f"eval.attn_implementation invalid: {ev.attn_implementation!r}")
    _require(ev.padding_side == "left", "eval.padding_side must be 'left' (batched generation)")
    _require(ev.shard in ("batch", "depth"), f"eval.shard must be 'batch' or 'depth', got {ev.shard!r}")
    _require(ev.batch_size >= 1 and ev.max_new_tokens >= 1, "eval.batch_size / max_new_tokens must be >= 1")
    _require(not ev.do_sample, "eval.do_sample must be false (greedy decoding)")
    return cfg


def _check_tasks(tasks: str, difficulty: Optional[str], section: str) -> None:
    """``<section>.tasks`` (and ``difficulty``) must name VisRO tasks (levels)."""
    from visro.dataset import DIFFICULTIES, resolve_tasks
    try:
        resolve_tasks(tasks)
    except ValueError as exc:
        raise ConfigValidationError(f"{section}.tasks: {exc}") from exc
    levels = [x.strip() for x in (difficulty or "").split(",") if x.strip()]
    bad = [x for x in levels if x not in DIFFICULTIES]
    _require(not bad, f"{section}.difficulty: unknown level(s) {bad}; levels are {DIFFICULTIES}")


# ----------------------------------------------------------------------------------------------
# Derived values
# ----------------------------------------------------------------------------------------------

def lm_lora_target_regex(target_modules: Sequence[str]) -> str:
    """Regex (PEFT ``target_modules`` string, matched with ``re.fullmatch``) selecting the given
    projections of the language-model decoder layers only.

    Matching by suffix (``["q_proj", ...]``) would also wrap the CLIP attention projections and the
    question pooler's ``q/k/v_proj`` whenever those modules exist when LoRA is attached; the
    explicit path makes the trainable set independent of construction order.
    """
    unknown = [t for t in target_modules if t not in LORA_ATTN_TARGETS + LORA_MLP_TARGETS]
    if unknown or not target_modules:
        raise ValueError(f"Unsupported LoRA targets {list(target_modules)}")
    attn = [t for t in LORA_ATTN_TARGETS if t in target_modules]
    mlp = [t for t in LORA_MLP_TARGETS if t in target_modules]
    parts = []
    if attn:
        parts.append(r"self_attn\.(?:" + "|".join(attn) + ")")
    if mlp:
        parts.append(r"mlp\.(?:" + "|".join(mlp) + ")")
    return r"model\.layers\.\d+\.(?:" + "|".join(parts) + ")"


def model_config_updates(cfg: Config) -> Dict[str, Any]:
    """Keys written into the model's ``config.json`` (internal names, read back by
    ``from_pretrained`` to rebuild the same tower)."""
    m, loop, vis = cfg.model, cfg.model.loop, cfg.model.vision
    out: Dict[str, Any] = dict(
        mm_vision_tower=cfg.paths.vision_tower,
        mm_vision_select_layer=int(vis.select_layer),
        mm_vision_select_feature=vis.select_feature,
        mm_projector_type=m.projector,
        mm_patch_merge_type="flat",
        mm_use_im_start_end=bool(m.mm_use_im_start_end),
        mm_use_im_patch_token=bool(m.mm_use_im_patch_token),
        vision_input_res=int(vis.input_res),
        vision_spatial_pool=int(vis.spatial_pool),
        image_aspect_ratio=cfg.data.image_aspect_ratio,
        condition_max_length=int(cfg.data.condition_max_length),
        loop_mode=loop.mode,
    )
    loop_keys = ("loop_start", "loop_end", "loop_steps", "loop_step_embed", "loop_deep_sup_weight",
                 "loop_fused_readout", "loop_readout_mode", "loop_use_adaln", "loop_condition_dim",
                 "loop_adaln_hidden_dim", "loop_step_embed_max_steps")
    if loop.mode == "none":
        out.update({k: None for k in loop_keys})
    else:
        out.update(
            loop_start=int(loop.band[0]), loop_end=int(loop.band[1]), loop_steps=int(loop.steps),
            loop_step_embed=loop.step_embed, loop_deep_sup_weight=loop.deep_sup_weight,
            loop_fused_readout=bool(loop.fused_readout), loop_readout_mode=loop.readout,
            loop_use_adaln=bool(loop.use_adaln), loop_condition_dim=int(loop.condition_dim),
            loop_adaln_hidden_dim=int(loop.adaln_hidden_dim),
            loop_step_embed_max_steps=int(loop.step_embed_max_steps))
        if loop.readout == "answer_trajectory":
            out.update(loop_traj_text=True, loop_draft_max_new_tokens=TRAJ_DRAFT_TOKENS,
                       loop_traj_sep_token_id=TRAJ_SEPARATOR_TOKEN_ID)
    return out


def rank_mean_factor(cfg: Config, world_size: int) -> float:
    """Factor applied to the loop loss under ``loss_normalization: rank_mean``.

    ``rank_mean_scale / (world_size * grad_accum)``: exactly 1.0 for the paper topology
    (8 GPUs x 1 accumulation step), and it keeps the effective objective at
    ``rank_mean_scale`` x the mean of per-rank means on other topologies.
    """
    return float(cfg.train.rank_mean_scale) / float(world_size * cfg.train.grad_accum)
