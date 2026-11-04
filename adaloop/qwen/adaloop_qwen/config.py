"""Strict, framework-free configuration for the Qwen2.5-VL AdaLoop backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from .errors import ConfigValidationError

BACKEND = "qwen2.5-vl"
SCHEMA_VERSION = 1
PAPER_MAX_CONTEXT_LENGTH = 2048
# The paper's Variant-C runs trained and evaluated interleaved trajectories
# with a 4096-token context; every other recipe used 2048.
PAPER_VARIANT_C_MAX_CONTEXT_LENGTH = 4096
PAPER_MIN_PIXELS = 12_544
PAPER_MAX_PIXELS = 451_584
# Token budget of the bare AdaLN question condition.
CONDITION_MAX_TOKENS = 128
# A Hub commit hash; branch and tag names are mutable and cannot be exported.
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
VARIANTS = frozenset({"sft", "adaln", "a", "b", "c"})
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
READOUT_FOR_VARIANT = {
    "sft": "single_pass",
    "adaln": "single_pass",
    "a": "current_image",
    "b": "accumulated_images",
    "c": "interleaved_drafts",
}


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader which refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConfigValidationError("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise ConfigValidationError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(
    value: Any, context: str, *, error: type[Exception] = ConfigValidationError
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{context} must be a mapping")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    context: str,
    allowed: set[str],
    required: set[str],
    error: type[Exception] = ConfigValidationError,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise error(
            f"{context} contains unknown field(s): {', '.join(sorted(unknown))}"
        )
    if missing:
        raise error(f"{context} is missing field(s): {', '.join(sorted(missing))}")


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigValidationError(f"{context} must be a positive integer")
    return value


def _positive_number(value: Any, context: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{context} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ConfigValidationError(f"{context} must be finite") from exc
    if not math.isfinite(result):
        raise ConfigValidationError(f"{context} must be finite")
    if result < 0 if allow_zero else result <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigValidationError(f"{context} must be {qualifier}")
    return result


def _span(value: Any, context: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigValidationError(f"{context} must be null or [start, end]")
    if isinstance(value[0], bool) or not isinstance(value[0], int) or value[0] < 0:
        raise ConfigValidationError(f"{context}[0] must be a non-negative integer")
    start = value[0]
    end = _positive_int(value[1], f"{context}[1]")
    if start >= end:
        raise ConfigValidationError(f"{context} must satisfy start < end")
    return start, end


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "LoRAConfig":
        raw = _mapping(raw, "lora")
        fields = {"rank", "alpha", "dropout", "target_modules"}
        _strict_keys(raw, context="lora", allowed=fields, required=fields)
        rank = _positive_int(raw["rank"], "lora.rank")
        alpha = _positive_int(raw["alpha"], "lora.alpha")
        dropout = _positive_number(raw["dropout"], "lora.dropout", allow_zero=True)
        if dropout >= 1:
            raise ConfigValidationError("lora.dropout must be smaller than 1")
        targets_raw = raw["target_modules"]
        if not isinstance(targets_raw, (list, tuple)):
            raise ConfigValidationError("lora.target_modules must be a list")
        targets = tuple(str(item) for item in targets_raw)
        if targets != LORA_TARGETS:
            raise ConfigValidationError(
                "Qwen2.5-VL release recipes require target_modules="
                f"{list(LORA_TARGETS)!r} in that order"
            )
        return cls(rank=rank, alpha=alpha, dropout=dropout, target_modules=targets)


@dataclass(frozen=True)
class ConditionerConfig:
    mode: str
    question_input: str | None
    include_bos: bool | None
    max_tokens: int | None
    hidden_size: int | None
    attention_heads: int | None
    constant_text: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "ConditionerConfig":
        raw = _mapping(raw, "conditioner")
        fields = {
            "mode",
            "question_input",
            "include_bos",
            "max_tokens",
            "hidden_size",
            "attention_heads",
            "constant_text",
        }
        _strict_keys(raw, context="conditioner", allowed=fields, required={"mode"})
        mode = raw["mode"]
        if mode not in {"none", "question", "constant_question"}:
            raise ConfigValidationError(
                "conditioner.mode must be none, question, or constant_question"
            )
        optional_values = {
            "question_input": raw.get("question_input"),
            "include_bos": raw.get("include_bos"),
            "max_tokens": raw.get("max_tokens"),
            "hidden_size": raw.get("hidden_size"),
            "attention_heads": raw.get("attention_heads"),
            "constant_text": raw.get("constant_text"),
        }
        if mode == "none":
            if any(value is not None for value in optional_values.values()):
                raise ConfigValidationError(
                    "conditioner fields other than mode must be omitted when mode=none"
                )
            return cls(mode=mode, **optional_values)

        required_condition_fields = {
            "question_input",
            "include_bos",
            "max_tokens",
            "hidden_size",
            "attention_heads",
        }
        missing = [
            name
            for name in required_condition_fields
            if optional_values[name] is None
        ]
        if missing:
            raise ConfigValidationError(
                "conditioner is missing field(s): " + ", ".join(sorted(missing))
            )
        if optional_values["question_input"] != "bare":
            raise ConfigValidationError("conditioner.question_input must be bare")
        if optional_values["include_bos"] is not False:
            raise ConfigValidationError("conditioner.include_bos must be false")
        max_tokens = _positive_int(optional_values["max_tokens"], "conditioner.max_tokens")
        if max_tokens != CONDITION_MAX_TOKENS:
            raise ConfigValidationError(
                f"conditioner.max_tokens must be {CONDITION_MAX_TOKENS}"
            )
        hidden_size = _positive_int(optional_values["hidden_size"], "conditioner.hidden_size")
        heads = _positive_int(
            optional_values["attention_heads"], "conditioner.attention_heads"
        )
        if hidden_size % heads:
            raise ConfigValidationError(
                "conditioner.hidden_size must be divisible by attention_heads"
            )
        constant_text = optional_values["constant_text"]
        if mode == "question":
            if constant_text is not None:
                raise ConfigValidationError(
                    "conditioner.constant_text is valid only for constant_question"
                )
        elif not isinstance(constant_text, str) or not constant_text.strip():
            raise ConfigValidationError(
                "constant_question requires a non-empty conditioner.constant_text"
            )
        return cls(
            mode=mode,
            question_input="bare",
            include_bos=False,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            attention_heads=heads,
            constant_text=(constant_text.strip() if constant_text is not None else None),
        )


@dataclass(frozen=True)
class ReadoutConfig:
    mode: str
    question_mode: str | None = None
    draft_max_tokens: int | None = None
    rollout_engine: str | None = None
    suppress_special_draft_tokens: bool | None = None
    separator: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "ReadoutConfig":
        raw = _mapping(raw, "readout")
        fields = {
            "mode",
            "question_mode",
            "draft_max_tokens",
            "rollout_engine",
            "suppress_special_draft_tokens",
            "separator",
        }
        _strict_keys(raw, context="readout", allowed=fields, required={"mode"})
        mode = raw["mode"]
        extra = {name: raw.get(name) for name in fields - {"mode"}}
        if mode != "interleaved_drafts":
            if any(value is not None for value in extra.values()):
                raise ConfigValidationError(
                    "draft readout fields are valid only for interleaved_drafts"
                )
            if mode not in {"single_pass", "current_image", "accumulated_images"}:
                raise ConfigValidationError(f"unsupported readout.mode: {mode!r}")
            return cls(mode=mode)

        missing = [name for name, value in extra.items() if value is None]
        if missing:
            raise ConfigValidationError(
                "interleaved_drafts readout is missing field(s): "
                + ", ".join(sorted(missing))
            )
        if extra["question_mode"] != "each_round":
            raise ConfigValidationError("Variant C requires question_mode=each_round")
        if extra["rollout_engine"] != "reprefill":
            raise ConfigValidationError("Variant C requires rollout_engine=reprefill")
        if extra["suppress_special_draft_tokens"] is not True:
            raise ConfigValidationError(
                "Variant C requires suppress_special_draft_tokens=true"
            )
        if extra["separator"] != "newline":
            raise ConfigValidationError("Variant C release recipe uses separator=newline")
        return cls(
            mode=mode,
            question_mode="each_round",
            draft_max_tokens=_positive_int(
                extra["draft_max_tokens"], "readout.draft_max_tokens"
            ),
            rollout_engine="reprefill",
            suppress_special_draft_tokens=True,
            separator="newline",
        )


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    max_grad_norm: float
    lora_lr: float
    merger_lr: float
    conditioner_lr: float

    @classmethod
    def from_dict(cls, raw: Any) -> "OptimizerConfig":
        raw = _mapping(raw, "training.optimizer")
        fields = {
            "name",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "max_grad_norm",
            "lora_lr",
            "merger_lr",
            "conditioner_lr",
        }
        _strict_keys(raw, context="training.optimizer", allowed=fields, required=fields)
        if raw["name"] != "adamw":
            raise ConfigValidationError("training.optimizer.name must be adamw")
        beta1 = _positive_number(raw["beta1"], "training.optimizer.beta1")
        beta2 = _positive_number(raw["beta2"], "training.optimizer.beta2")
        if beta1 >= 1 or beta2 >= 1:
            raise ConfigValidationError("optimizer beta values must be smaller than 1")
        return cls(
            name="adamw",
            beta1=beta1,
            beta2=beta2,
            epsilon=_positive_number(raw["epsilon"], "training.optimizer.epsilon"),
            weight_decay=_positive_number(
                raw["weight_decay"], "training.optimizer.weight_decay", allow_zero=True
            ),
            max_grad_norm=_positive_number(
                raw["max_grad_norm"], "training.optimizer.max_grad_norm"
            ),
            lora_lr=_positive_number(raw["lora_lr"], "training.optimizer.lora_lr"),
            merger_lr=_positive_number(
                raw["merger_lr"], "training.optimizer.merger_lr"
            ),
            conditioner_lr=_positive_number(
                raw["conditioner_lr"],
                "training.optimizer.conditioner_lr",
                allow_zero=True,
            ),
        )


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    global_batch_size: int
    precision: str
    loss_normalization: str
    supervision: str
    geometric_ratio: float | None
    optimizer: OptimizerConfig
    seed: int = 42
    warmup_steps: int = 200
    scheduler: str = "cosine"
    min_lr_ratio: float = 0.1
    gradient_checkpointing: bool = True
    train_vision_encoder: bool = False
    train_merger: bool = True

    @classmethod
    def from_dict(cls, raw: Any) -> "TrainingConfig":
        raw = _mapping(raw, "training")
        fields = {
            "steps",
            "global_batch_size",
            "precision",
            "loss_normalization",
            "supervision",
            "geometric_ratio",
            "optimizer",
            "seed",
            "warmup_steps",
            "scheduler",
            "min_lr_ratio",
            "gradient_checkpointing",
            "train_vision_encoder",
            "train_merger",
        }
        _strict_keys(
            raw,
            context="training",
            allowed=fields,
            required={
                "steps",
                "global_batch_size",
                "precision",
                "loss_normalization",
                "supervision",
                "geometric_ratio",
                "optimizer",
            },
        )
        # The paper trained in bf16; adaloop-qwen-train records fp32 for a
        # --mixed-precision no (CPU/debug) run so its checkpoints say so.
        precision = raw["precision"]
        if not isinstance(precision, str) or precision not in {"bf16", "fp32"}:
            raise ConfigValidationError("training.precision must be bf16 or fp32")
        if raw["loss_normalization"] != "per_example_answer_tokens":
            raise ConfigValidationError(
                "training.loss_normalization must be per_example_answer_tokens"
            )
        supervision = raw["supervision"]
        if supervision not in {"terminal", "deep_geometric", "sampled_geometric"}:
            raise ConfigValidationError(f"unsupported training.supervision: {supervision!r}")
        ratio_raw = raw["geometric_ratio"]
        if supervision == "terminal":
            if ratio_raw is not None:
                raise ConfigValidationError(
                    "training.geometric_ratio must be null for terminal supervision"
                )
            ratio = None
        else:
            ratio = _positive_number(ratio_raw, "training.geometric_ratio")
            if ratio > 1:
                raise ConfigValidationError(
                    "training.geometric_ratio must be in the interval (0, 1]"
                )
        seed = raw.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigValidationError("training.seed must be a non-negative integer")
        warmup_steps = raw.get("warmup_steps", 200)
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or warmup_steps < 0
        ):
            raise ConfigValidationError(
                "training.warmup_steps must be a non-negative integer"
            )
        scheduler = raw.get("scheduler", "cosine")
        if scheduler != "cosine":
            raise ConfigValidationError("training.scheduler must be cosine")
        min_lr_ratio = _positive_number(
            raw.get("min_lr_ratio", 0.1),
            "training.min_lr_ratio",
            allow_zero=True,
        )
        if min_lr_ratio > 1:
            raise ConfigValidationError("training.min_lr_ratio must be in [0, 1]")
        boolean_fields = {
            "gradient_checkpointing": raw.get("gradient_checkpointing", True),
            "train_vision_encoder": raw.get("train_vision_encoder", False),
            "train_merger": raw.get("train_merger", True),
        }
        for name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise ConfigValidationError(f"training.{name} must be boolean")
        if boolean_fields["train_vision_encoder"]:
            raise ConfigValidationError(
                "the paper release keeps the pretrained vision encoder frozen"
            )
        if not boolean_fields["train_merger"]:
            raise ConfigValidationError(
                "the paper release trains the Qwen vision-language merger"
            )
        return cls(
            steps=_positive_int(raw["steps"], "training.steps"),
            global_batch_size=_positive_int(
                raw["global_batch_size"], "training.global_batch_size"
            ),
            precision=precision,
            loss_normalization="per_example_answer_tokens",
            supervision=supervision,
            geometric_ratio=ratio,
            optimizer=OptimizerConfig.from_dict(raw["optimizer"]),
            seed=seed,
            warmup_steps=warmup_steps,
            scheduler=scheduler,
            min_lr_ratio=min_lr_ratio,
            gradient_checkpointing=boolean_fields["gradient_checkpointing"],
            train_vision_encoder=False,
            train_merger=True,
        )


@dataclass(frozen=True)
class GenerationConfig:
    """Deterministic decoding contract used by evaluation and draft rollout."""

    max_new_tokens: int = 64
    do_sample: bool = False

    @classmethod
    def from_dict(cls, raw: Any) -> "GenerationConfig":
        if raw is None:
            return cls()
        raw = _mapping(raw, "generation")
        fields = {"max_new_tokens", "do_sample"}
        _strict_keys(raw, context="generation", allowed=fields, required=fields)
        if raw["do_sample"] is not False:
            raise ConfigValidationError("generation.do_sample must be false")
        return cls(
            max_new_tokens=_positive_int(
                raw["max_new_tokens"], "generation.max_new_tokens"
            ),
            do_sample=False,
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    """Input limits used by the released Qwen experiments."""

    max_length: int = PAPER_MAX_CONTEXT_LENGTH
    min_pixels: int = PAPER_MIN_PIXELS
    max_pixels: int = PAPER_MAX_PIXELS

    @classmethod
    def from_dict(
        cls, raw: Any, *, default_max_length: int = PAPER_MAX_CONTEXT_LENGTH
    ) -> "PreprocessingConfig":
        if raw is None:
            return cls(max_length=default_max_length)
        raw = _mapping(raw, "preprocessing")
        fields = {"max_length", "min_pixels", "max_pixels"}
        _strict_keys(raw, context="preprocessing", allowed=fields, required=fields)
        maximum_length = _positive_int(
            raw["max_length"], "preprocessing.max_length"
        )
        minimum_pixels = _positive_int(
            raw["min_pixels"], "preprocessing.min_pixels"
        )
        maximum_pixels = _positive_int(
            raw["max_pixels"], "preprocessing.max_pixels"
        )
        if minimum_pixels > maximum_pixels:
            raise ConfigValidationError(
                "preprocessing.min_pixels must not exceed preprocessing.max_pixels"
            )
        return cls(
            max_length=maximum_length,
            min_pixels=minimum_pixels,
            max_pixels=maximum_pixels,
        )


@dataclass(frozen=True)
class QwenAdaLoopConfig:
    schema_version: int
    backend: str
    base_model: str
    base_revision: str
    variant: str
    adaptive_span: tuple[int, int] | None
    train_depth: int
    vision_round_mode: str
    lora: LoRAConfig
    conditioner: ConditionerConfig
    readout: ReadoutConfig
    training: TrainingConfig
    generation: GenerationConfig = GenerationConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()

    @classmethod
    def from_dict(cls, raw: Any) -> "QwenAdaLoopConfig":
        raw = _mapping(raw, "config")
        fields = {
            "schema_version",
            "backend",
            "base_model",
            "base_revision",
            "variant",
            "adaptive_span",
            "train_depth",
            "vision_round_mode",
            "lora",
            "conditioner",
            "readout",
            "training",
            "generation",
            "preprocessing",
        }
        _strict_keys(
            raw,
            context="config",
            allowed=fields,
            required=fields - {"generation", "preprocessing"},
        )
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ConfigValidationError(
                f"schema_version must be {SCHEMA_VERSION}, got {raw['schema_version']!r}"
            )
        if raw["backend"] != BACKEND:
            raise ConfigValidationError(
                f"backend must be {BACKEND!r}; Qwen3 backends are not supported"
            )
        base_model = raw["base_model"]
        if not isinstance(base_model, str) or "Qwen2.5-VL" not in base_model:
            raise ConfigValidationError(
                "base_model must identify a Qwen2.5-VL checkpoint"
            )
        base_revision = raw["base_revision"]
        if not isinstance(base_revision, str) or not IMMUTABLE_REVISION.fullmatch(
            base_revision
        ):
            # Export and the checkpoint manifest require the same pin; reject
            # it here instead of after a full training run.
            raise ConfigValidationError(
                "base_revision must pin an immutable 40-64 character lowercase "
                "hexadecimal commit hash, not a branch or tag name"
            )
        variant = str(raw["variant"]).lower()
        if variant not in VARIANTS:
            raise ConfigValidationError(
                f"variant must be one of {sorted(VARIANTS)}, got {variant!r}"
            )
        adaptive_span = _span(raw["adaptive_span"], "adaptive_span")
        train_depth = _positive_int(raw["train_depth"], "train_depth")
        if raw["vision_round_mode"] != "latent_carry":
            raise ConfigValidationError("vision_round_mode must be latent_carry")
        conditioner = ConditionerConfig.from_dict(raw["conditioner"])
        readout = ReadoutConfig.from_dict(raw["readout"])
        training = TrainingConfig.from_dict(raw["training"])
        generation = GenerationConfig.from_dict(raw.get("generation"))
        preprocessing = PreprocessingConfig.from_dict(
            raw.get("preprocessing"),
            default_max_length=(
                PAPER_VARIANT_C_MAX_CONTEXT_LENGTH
                if variant == "c"
                else PAPER_MAX_CONTEXT_LENGTH
            ),
        )

        if variant == "sft":
            if adaptive_span is not None or conditioner.mode != "none":
                raise ConfigValidationError(
                    "sft requires adaptive_span=null and conditioner.mode=none"
                )
        else:
            if adaptive_span is None:
                raise ConfigValidationError(f"variant {variant} requires adaptive_span")
            if conditioner.mode == "none" and variant != "a":
                raise ConfigValidationError(
                    "only Variant A supports the paper's no-modulation loop control"
                )
        expected_readout = READOUT_FOR_VARIANT[variant]
        if readout.mode != expected_readout:
            raise ConfigValidationError(
                f"variant {variant} requires readout.mode={expected_readout}"
            )
        if variant in {"sft", "adaln"}:
            if train_depth != 1 or training.supervision != "terminal":
                raise ConfigValidationError(
                    f"variant {variant} requires train_depth=1 and terminal supervision"
                )
        else:
            if train_depth < 2:
                raise ConfigValidationError(f"variant {variant} requires train_depth >= 2")
            if training.supervision == "terminal":
                raise ConfigValidationError(
                    f"variant {variant} requires deep or sampled geometric supervision"
                )
            if training.supervision == "sampled_geometric" and variant != "a":
                raise ConfigValidationError(
                    "sampled_geometric supervision is implemented only for Variant A"
                )
        if conditioner.mode == "none" and training.optimizer.conditioner_lr != 0:
            raise ConfigValidationError(
                "conditioner_lr must be 0 when conditioner.mode=none"
            )
        return cls(
            schema_version=SCHEMA_VERSION,
            backend=BACKEND,
            base_model=base_model,
            base_revision=base_revision,
            variant=variant,
            adaptive_span=adaptive_span,
            train_depth=train_depth,
            vision_round_mode="latent_carry",
            lora=LoRAConfig.from_dict(raw["lora"]),
            conditioner=conditioner,
            readout=readout,
            training=training,
            generation=generation,
            preprocessing=preprocessing,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "QwenAdaLoopConfig":
        path = Path(path)
        try:
            raw = yaml.load(
                path.read_text(encoding="utf-8"), Loader=_StrictSafeLoader
            )
        except (OSError, yaml.YAMLError, ConfigValidationError) as exc:
            raise ConfigValidationError(f"could not read config {path}: {exc}") from exc
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-serializable representation."""
        result = asdict(self)
        result["adaptive_span"] = (
            list(self.adaptive_span) if self.adaptive_span is not None else None
        )
        result["lora"]["target_modules"] = list(self.lora.target_modules)
        return result
