"""Strict, framework-free configuration for the PaliGemma2/JAX backend.

Only the model family and AdaLoop modes used by the paper are represented,
plus one release extension: ``conditioner.mode: constant_question`` (the
paper's constant-question control is Qwen-only).  Other switches are
deliberately not accepted as inert fields: an unknown key is an error,
and every topology-affecting choice is serialized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .constants import (
    GEMMA_VOCAB_SIZE,
    PAPER_ADALN_HIDDEN_DIM,
    PAPER_CONDITION_DIM,
    PAPER_CONDITION_HEADS,
    PAPER_DRAFT_MAX_TOKENS,
    PAPER_EVAL_MAX_TEXT_TOKENS,
    PAPER_EVAL_PROMPT_TEMPLATE,
    PAPER_GEOMETRIC_RATIO,
    PAPER_IMAGE_SIZE,
    PAPER_LANGUAGE_VARIANT,
    PAPER_LOOP_SPAN,
    PAPER_LORA_ALPHA,
    PAPER_LORA_RANK,
    PAPER_LORA_TARGETS,
    PAPER_MAX_NEW_TOKENS,
    PAPER_MAX_SEQUENCE_LENGTH,
    PAPER_MAX_TEXT_TOKENS,
    PAPER_QUESTION_MAX_TOKENS,
    PAPER_TRAIN_DEPTH,
    PAPER_TRAIN_PROMPT_TEMPLATE,
    PAPER_VISION_DEPTH,
    PAPER_VISION_VARIANT,
    RECURRENT_VARIANTS,
    VARIANTS,
)
from .errors import ConfigValidationError


SCHEMA_VERSION = 1
BACKEND = "paligemma2-jax"
BASE_MODEL_ID = "google/paligemma2-3b-mix-448"
BASE_MODEL_FORMAT = "big_vision_npz"
BASE_MODEL_SHA256 = "2d426f6d23ec5ecd9b49f01a815bfb3ce5a373fa98bd154568cc402dcf8fc049"
BASE_MODEL_SIZE = 6_065_970_800
TOKENIZER_FORMAT = "sentencepiece"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
TOKENIZER_SIZE = 4_264_023

READOUT_FOR_VARIANT = {
    "sft": "single_pass",
    "adaln": "single_pass",
    "a": "current_image",
    "b": "accumulated_images",
    "c": "interleaved_drafts",
}


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader which refuses duplicate mapping keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False):
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConfigValidationError("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise ConfigValidationError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{context} must be a mapping")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    context: str,
    allowed: set[str],
    required: set[str],
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ConfigValidationError(
            f"{context} contains unknown field(s): {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ConfigValidationError(
            f"{context} is missing field(s): {', '.join(sorted(missing))}"
        )


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigValidationError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigValidationError(f"{context} must be a non-negative integer")
    return value


def _number(value: Any, context: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigValidationError(f"{context} must be finite")
    invalid = result < 0 if allow_zero else result <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigValidationError(f"{context} must be {qualifier}")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{context} must be boolean")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigValidationError(f"{context} must be a non-empty string")
    return value


def _sha256_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ConfigValidationError(f"{context} must be a lowercase SHA-256 digest")
    if any(char not in "0123456789abcdef" for char in value):
        raise ConfigValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _span(value: Any, context: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigValidationError(f"{context} must be null or [start, end]")
    start = _nonnegative_int(value[0], f"{context}[0]")
    end = _positive_int(value[1], f"{context}[1]")
    if start >= end:
        raise ConfigValidationError(f"{context} must satisfy start < end")
    return start, end


@dataclass(frozen=True)
class AssetConfig:
    """Immutable external artifact identity; paths never enter the recipe."""

    format: str
    sha256: str
    size_bytes: int
    id: str | None = None

    @classmethod
    def from_dict(cls, raw: Any, *, kind: str) -> "AssetConfig":
        raw = _mapping(raw, kind)
        fields = {"id", "format", "sha256", "size_bytes"}
        required = {"format", "sha256", "size_bytes"}
        if kind == "base_model":
            required.add("id")
        _strict_keys(raw, context=kind, allowed=fields, required=required)
        identifier = raw.get("id")
        if identifier is not None and (
            not isinstance(identifier, str) or not identifier.strip()
        ):
            raise ConfigValidationError(f"{kind}.id must be a non-empty string")
        result = cls(
            id=identifier,
            format=_string(raw["format"], f"{kind}.format"),
            sha256=_sha256_digest(raw["sha256"], f"{kind}.sha256"),
            size_bytes=_positive_int(raw["size_bytes"], f"{kind}.size_bytes"),
        )
        if kind == "base_model":
            expected = (
                BASE_MODEL_ID,
                BASE_MODEL_FORMAT,
                BASE_MODEL_SHA256,
                BASE_MODEL_SIZE,
            )
            actual = (result.id, result.format, result.sha256, result.size_bytes)
            if actual != expected:
                raise ConfigValidationError(
                    "base_model must identify the pinned PaliGemma2-3B-Mix-448 "
                    "Big Vision artifact"
                )
        elif (result.format, result.sha256, result.size_bytes) != (
            TOKENIZER_FORMAT,
            TOKENIZER_SHA256,
            TOKENIZER_SIZE,
        ):
            raise ConfigValidationError(
                "tokenizer must identify the pinned PaliGemma SentencePiece artifact"
            )
        return result


@dataclass(frozen=True)
class BackboneConfig:
    image_size: int
    vision_variant: str
    vision_depth: int
    language_variant: str
    vocab_size: int

    @classmethod
    def from_dict(cls, raw: Any) -> "BackboneConfig":
        raw = _mapping(raw, "backbone")
        fields = {
            "image_size",
            "vision_variant",
            "vision_depth",
            "language_variant",
            "vocab_size",
        }
        _strict_keys(raw, context="backbone", allowed=fields, required=fields)
        result = cls(
            image_size=_positive_int(raw["image_size"], "backbone.image_size"),
            vision_variant=_string(raw["vision_variant"], "backbone.vision_variant"),
            vision_depth=_positive_int(raw["vision_depth"], "backbone.vision_depth"),
            language_variant=_string(
                raw["language_variant"], "backbone.language_variant"
            ),
            vocab_size=_positive_int(raw["vocab_size"], "backbone.vocab_size"),
        )
        if result != cls(
            image_size=PAPER_IMAGE_SIZE,
            vision_variant=PAPER_VISION_VARIANT,
            vision_depth=PAPER_VISION_DEPTH,
            language_variant=PAPER_LANGUAGE_VARIANT,
            vocab_size=GEMMA_VOCAB_SIZE,
        ):
            raise ConfigValidationError(
                "backbone must be PaliGemma2-3B-Mix-448 with So400m/14"
            )
        return result


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    layers: str
    seed: int

    @classmethod
    def from_dict(cls, raw: Any) -> "LoRAConfig":
        raw = _mapping(raw, "lora")
        fields = {"rank", "alpha", "dropout", "target_modules", "layers", "seed"}
        _strict_keys(raw, context="lora", allowed=fields, required=fields)
        targets_raw = raw["target_modules"]
        if not isinstance(targets_raw, (list, tuple)):
            raise ConfigValidationError("lora.target_modules must be a list")
        if any(not isinstance(item, str) or not item for item in targets_raw):
            raise ConfigValidationError(
                "lora.target_modules entries must be non-empty strings"
            )
        targets = tuple(targets_raw)
        if targets != PAPER_LORA_TARGETS:
            raise ConfigValidationError(
                f"PaliGemma release recipes require target_modules={list(PAPER_LORA_TARGETS)!r}"
            )
        if raw["layers"] != "all":
            raise ConfigValidationError("lora.layers must be all")
        dropout = _number(raw["dropout"], "lora.dropout", allow_zero=True)
        if dropout >= 1:
            raise ConfigValidationError("lora.dropout must be smaller than 1")
        result = cls(
            rank=_positive_int(raw["rank"], "lora.rank"),
            alpha=_positive_int(raw["alpha"], "lora.alpha"),
            dropout=dropout,
            target_modules=targets,
            layers="all",
            seed=_nonnegative_int(raw["seed"], "lora.seed"),
        )
        if (
            result.rank != PAPER_LORA_RANK
            or result.alpha != PAPER_LORA_ALPHA
            or result.dropout != 0.0
        ):
            raise ConfigValidationError(
                "paper PaliGemma recipes require LoRA rank=64, alpha=128, dropout=0"
            )
        return result


@dataclass(frozen=True)
class ConditionerConfig:
    mode: str
    question_input: str | None = None
    include_bos: bool | None = None
    max_tokens: int | None = None
    hidden_size: int | None = None
    attention_heads: int | None = None
    detach: bool | None = None
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
            "detach",
            "constant_text",
        }
        _strict_keys(raw, context="conditioner", allowed=fields, required={"mode"})
        mode = raw["mode"]
        if mode not in {"none", "question", "constant_question"}:
            raise ConfigValidationError(
                "conditioner.mode must be none, question, or constant_question"
            )
        optional = {name: raw.get(name) for name in fields - {"mode"}}
        if mode == "none":
            if any(value is not None for value in optional.values()):
                raise ConfigValidationError(
                    "conditioner fields other than mode must be omitted when mode=none"
                )
            return cls(mode="none")

        required = {
            "question_input",
            "include_bos",
            "max_tokens",
            "hidden_size",
            "attention_heads",
            "detach",
        }
        missing = sorted(name for name in required if optional[name] is None)
        if missing:
            raise ConfigValidationError(
                "conditioner is missing field(s): " + ", ".join(missing)
            )
        if optional["question_input"] != "bare":
            raise ConfigValidationError("conditioner.question_input must be bare")
        if optional["include_bos"] is not False:
            raise ConfigValidationError("conditioner.include_bos must be false")
        if optional["detach"] is not True:
            raise ConfigValidationError("conditioner.detach must be true")
        max_tokens = _positive_int(optional["max_tokens"], "conditioner.max_tokens")
        hidden_size = _positive_int(optional["hidden_size"], "conditioner.hidden_size")
        heads = _positive_int(
            optional["attention_heads"], "conditioner.attention_heads"
        )
        if (max_tokens, hidden_size, heads) != (
            PAPER_QUESTION_MAX_TOKENS,
            PAPER_CONDITION_DIM,
            PAPER_CONDITION_HEADS,
        ):
            raise ConfigValidationError(
                "paper question conditioning requires bare/no-BOS, 128 tokens, "
                "512 dimensions, and 8 attention heads"
            )
        constant_text = optional["constant_text"]
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
            detach=True,
            constant_text=constant_text.strip() if constant_text is not None else None,
        )


@dataclass(frozen=True)
class AdaLNConfig:
    mode: str
    hidden_size: int | None = None
    variant: str | None = None
    zero_init: bool | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "AdaLNConfig":
        raw = _mapping(raw, "adaln")
        fields = {"mode", "hidden_size", "variant", "zero_init"}
        _strict_keys(raw, context="adaln", allowed=fields, required={"mode"})
        mode = raw["mode"]
        extra = {name: raw.get(name) for name in fields - {"mode"}}
        if mode == "none":
            if any(value is not None for value in extra.values()):
                raise ConfigValidationError(
                    "adaln fields other than mode must be omitted when mode=none"
                )
            return cls(mode="none")
        if mode != "independent":
            raise ConfigValidationError("adaln.mode must be none or independent")
        missing = sorted(name for name, value in extra.items() if value is None)
        if missing:
            raise ConfigValidationError("adaln is missing field(s): " + ", ".join(missing))
        hidden_size = _positive_int(extra["hidden_size"], "adaln.hidden_size")
        if hidden_size != PAPER_ADALN_HIDDEN_DIM:
            raise ConfigValidationError("adaln.hidden_size must be 128")
        if extra["variant"] != "full":
            raise ConfigValidationError("adaln.variant must be full")
        if extra["zero_init"] is not True:
            raise ConfigValidationError("adaln.zero_init must be true")
        return cls(
            mode="independent", hidden_size=hidden_size, variant="full", zero_init=True
        )


@dataclass(frozen=True)
class ReadoutConfig:
    mode: str
    block_mode: str | None = None
    max_length: int | None = None
    question_mode: str | None = None
    draft_max_tokens: int | None = None
    condition_mode: str | None = None
    rollout_engine: str | None = None
    suppress_special_draft_tokens: bool | None = None
    separator: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "ReadoutConfig":
        raw = _mapping(raw, "readout")
        fields = {
            "mode",
            "block_mode",
            "max_length",
            "question_mode",
            "draft_max_tokens",
            "condition_mode",
            "rollout_engine",
            "suppress_special_draft_tokens",
            "separator",
        }
        _strict_keys(raw, context="readout", allowed=fields, required={"mode"})
        mode = raw["mode"]
        extra = {name: raw.get(name) for name in fields - {"mode"}}
        if mode in {"single_pass", "current_image"}:
            if any(value is not None for value in extra.values()):
                raise ConfigValidationError(
                    f"readout fields other than mode are invalid for {mode}"
                )
            return cls(mode=mode)
        if mode == "accumulated_images":
            allowed = {"block_mode", "max_length"}
            unexpected = [
                name for name, value in extra.items() if name not in allowed and value is not None
            ]
            if unexpected:
                raise ConfigValidationError(
                    "accumulated_images contains invalid field(s): "
                    + ", ".join(sorted(unexpected))
                )
            if extra["block_mode"] != "chained":
                raise ConfigValidationError("Variant B requires block_mode=chained")
            maximum = _positive_int(extra["max_length"], "readout.max_length")
            if maximum != PAPER_MAX_SEQUENCE_LENGTH:
                raise ConfigValidationError("Variant B requires max_length=8192")
            return cls(mode=mode, block_mode="chained", max_length=maximum)
        if mode != "interleaved_drafts":
            raise ConfigValidationError(f"unsupported readout.mode: {mode!r}")

        c_fields = {
            "max_length",
            "question_mode",
            "draft_max_tokens",
            "condition_mode",
            "rollout_engine",
            "suppress_special_draft_tokens",
            "separator",
        }
        if extra["block_mode"] is not None:
            raise ConfigValidationError(
                "readout.block_mode is invalid for interleaved_drafts"
            )
        missing = sorted(name for name in c_fields if extra[name] is None)
        if missing:
            raise ConfigValidationError(
                "interleaved_drafts is missing field(s): " + ", ".join(missing)
            )
        expected = {
            "question_mode": "each_round",
            "condition_mode": "question_only",
            "rollout_engine": "reprefill",
            "suppress_special_draft_tokens": True,
            "separator": "newline",
        }
        for name, value in expected.items():
            if extra[name] != value:
                raise ConfigValidationError(
                    f"Variant C requires readout.{name}={value!r}"
                )
        draft_max = _positive_int(
            extra["draft_max_tokens"], "readout.draft_max_tokens"
        )
        maximum = _positive_int(extra["max_length"], "readout.max_length")
        if (
            draft_max != PAPER_DRAFT_MAX_TOKENS
            or maximum != PAPER_MAX_SEQUENCE_LENGTH
        ):
            raise ConfigValidationError(
                "paper Variant C requires draft_max_tokens=24 and max_length=8192"
            )
        return cls(
            mode=mode,
            max_length=maximum,
            question_mode="each_round",
            draft_max_tokens=draft_max,
            condition_mode="question_only",
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
    connector_lr: float
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
            "connector_lr",
            "conditioner_lr",
        }
        _strict_keys(raw, context="training.optimizer", allowed=fields, required=fields)
        if raw["name"] != "adamw":
            raise ConfigValidationError("training.optimizer.name must be adamw")
        beta1 = _number(raw["beta1"], "training.optimizer.beta1")
        beta2 = _number(raw["beta2"], "training.optimizer.beta2")
        if beta1 >= 1 or beta2 >= 1:
            raise ConfigValidationError("optimizer beta values must be smaller than 1")
        return cls(
            name="adamw",
            beta1=beta1,
            beta2=beta2,
            epsilon=_number(raw["epsilon"], "training.optimizer.epsilon"),
            weight_decay=_number(
                raw["weight_decay"], "training.optimizer.weight_decay", allow_zero=True
            ),
            max_grad_norm=_number(
                raw["max_grad_norm"], "training.optimizer.max_grad_norm"
            ),
            lora_lr=_number(raw["lora_lr"], "training.optimizer.lora_lr"),
            connector_lr=_number(
                raw["connector_lr"], "training.optimizer.connector_lr"
            ),
            conditioner_lr=_number(
                raw["conditioner_lr"],
                "training.optimizer.conditioner_lr",
                allow_zero=True,
            ),
        )


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    global_batch_size: int
    seed: int
    warmup_steps: int
    scheduler: str
    min_lr_ratio: float
    precision: str
    loss_normalization: str
    supervision: str
    geometric_ratio: float | None
    train_vision_encoder: bool
    train_connector: bool
    optimizer: OptimizerConfig

    @classmethod
    def from_dict(cls, raw: Any) -> "TrainingConfig":
        raw = _mapping(raw, "training")
        fields = {
            "steps",
            "global_batch_size",
            "seed",
            "warmup_steps",
            "scheduler",
            "min_lr_ratio",
            "precision",
            "loss_normalization",
            "supervision",
            "geometric_ratio",
            "train_vision_encoder",
            "train_connector",
            "optimizer",
        }
        _strict_keys(raw, context="training", allowed=fields, required=fields)
        steps = _positive_int(raw["steps"], "training.steps")
        warmup = _nonnegative_int(raw["warmup_steps"], "training.warmup_steps")
        if warmup >= steps:
            raise ConfigValidationError("training.warmup_steps must be smaller than steps")
        if raw["scheduler"] != "cosine":
            raise ConfigValidationError("training.scheduler must be cosine")
        minimum_lr = _number(
            raw["min_lr_ratio"], "training.min_lr_ratio", allow_zero=True
        )
        if minimum_lr > 1:
            raise ConfigValidationError("training.min_lr_ratio must be in [0, 1]")
        if raw["precision"] != "bf16":
            raise ConfigValidationError("training.precision must be bf16")
        if raw["loss_normalization"] != "per_example":
            raise ConfigValidationError("training.loss_normalization must be per_example")
        supervision = raw["supervision"]
        if supervision not in {"terminal", "deep_geometric"}:
            raise ConfigValidationError(
                "training.supervision must be terminal or deep_geometric"
            )
        ratio = raw["geometric_ratio"]
        if supervision == "terminal":
            if ratio is not None:
                raise ConfigValidationError(
                    "training.geometric_ratio must be null for terminal supervision"
                )
            parsed_ratio = None
        else:
            parsed_ratio = _number(ratio, "training.geometric_ratio")
            if parsed_ratio != PAPER_GEOMETRIC_RATIO:
                raise ConfigValidationError(
                    "paper deep supervision requires geometric_ratio=0.8"
                )
        if _boolean(raw["train_vision_encoder"], "training.train_vision_encoder"):
            raise ConfigValidationError("paper recipes keep the vision encoder frozen")
        if not _boolean(raw["train_connector"], "training.train_connector"):
            raise ConfigValidationError("paper recipes train the image connector")
        return cls(
            steps=steps,
            global_batch_size=_positive_int(
                raw["global_batch_size"], "training.global_batch_size"
            ),
            seed=_nonnegative_int(raw["seed"], "training.seed"),
            warmup_steps=warmup,
            scheduler="cosine",
            min_lr_ratio=minimum_lr,
            precision="bf16",
            loss_normalization="per_example",
            supervision=supervision,
            geometric_ratio=parsed_ratio,
            train_vision_encoder=False,
            train_connector=True,
            optimizer=OptimizerConfig.from_dict(raw["optimizer"]),
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int
    interpolation: str
    normalization: str
    max_text_tokens: int
    eval_max_text_tokens: int
    prompt_template: str
    eval_prompt_template: str
    separator: str

    @classmethod
    def from_dict(cls, raw: Any) -> "PreprocessingConfig":
        raw = _mapping(raw, "preprocessing")
        fields = {
            "image_size",
            "interpolation",
            "normalization",
            "max_text_tokens",
            "eval_max_text_tokens",
            "prompt_template",
            "eval_prompt_template",
            "separator",
        }
        _strict_keys(raw, context="preprocessing", allowed=fields, required=fields)
        result = cls(
            image_size=_positive_int(raw["image_size"], "preprocessing.image_size"),
            interpolation=_string(
                raw["interpolation"], "preprocessing.interpolation"
            ),
            normalization=_string(
                raw["normalization"], "preprocessing.normalization"
            ),
            max_text_tokens=_positive_int(
                raw["max_text_tokens"], "preprocessing.max_text_tokens"
            ),
            eval_max_text_tokens=_positive_int(
                raw["eval_max_text_tokens"],
                "preprocessing.eval_max_text_tokens",
            ),
            prompt_template=_string(
                raw["prompt_template"], "preprocessing.prompt_template"
            ),
            eval_prompt_template=_string(
                raw["eval_prompt_template"],
                "preprocessing.eval_prompt_template",
            ),
            separator=_string(raw["separator"], "preprocessing.separator"),
        )
        if result != cls(
            image_size=PAPER_IMAGE_SIZE,
            interpolation="bicubic",
            normalization="minus_one_to_one",
            max_text_tokens=PAPER_MAX_TEXT_TOKENS,
            eval_max_text_tokens=PAPER_EVAL_MAX_TEXT_TOKENS,
            prompt_template=PAPER_TRAIN_PROMPT_TEMPLATE,
            eval_prompt_template=PAPER_EVAL_PROMPT_TEMPLATE,
            separator="newline",
        ):
            raise ConfigValidationError(
                "preprocessing must match the signed PaliGemma paper contract"
            )
        return result


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    do_sample: bool

    @classmethod
    def from_dict(cls, raw: Any) -> "GenerationConfig":
        raw = _mapping(raw, "generation")
        fields = {"max_new_tokens", "do_sample"}
        _strict_keys(raw, context="generation", allowed=fields, required=fields)
        maximum = _positive_int(raw["max_new_tokens"], "generation.max_new_tokens")
        if maximum != PAPER_MAX_NEW_TOKENS:
            raise ConfigValidationError("generation.max_new_tokens must be 128")
        if raw["do_sample"] is not False:
            raise ConfigValidationError("generation.do_sample must be false")
        return cls(max_new_tokens=maximum, do_sample=False)


@dataclass(frozen=True)
class PaliGemmaAdaLoopConfig:
    schema_version: int
    backend: str
    base_model: AssetConfig
    tokenizer: AssetConfig
    backbone: BackboneConfig
    variant: str
    adaptive_span: tuple[int, int] | None
    train_depth: int
    vision_round_mode: str
    lora: LoRAConfig
    conditioner: ConditionerConfig
    adaln: AdaLNConfig
    readout: ReadoutConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    generation: GenerationConfig

    @classmethod
    def from_dict(cls, raw: Any) -> "PaliGemmaAdaLoopConfig":
        raw = _mapping(raw, "config")
        fields = {
            "schema_version",
            "backend",
            "base_model",
            "tokenizer",
            "backbone",
            "variant",
            "adaptive_span",
            "train_depth",
            "vision_round_mode",
            "lora",
            "conditioner",
            "adaln",
            "readout",
            "preprocessing",
            "training",
            "generation",
        }
        _strict_keys(raw, context="config", allowed=fields, required=fields)
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ConfigValidationError(
                f"unsupported schema_version: {raw['schema_version']!r}"
            )
        if raw["backend"] != BACKEND:
            raise ConfigValidationError(f"backend must be {BACKEND!r}")
        variant = raw["variant"]
        if variant not in VARIANTS:
            raise ConfigValidationError(f"unsupported variant: {variant!r}")
        span = _span(raw["adaptive_span"], "adaptive_span")
        depth = _positive_int(raw["train_depth"], "train_depth")
        if raw["vision_round_mode"] != "latent_carry":
            raise ConfigValidationError("vision_round_mode must be latent_carry")

        conditioner = ConditionerConfig.from_dict(raw["conditioner"])
        adaln = AdaLNConfig.from_dict(raw["adaln"])
        readout = ReadoutConfig.from_dict(raw["readout"])
        training = TrainingConfig.from_dict(raw["training"])
        expected_readout = READOUT_FOR_VARIANT[variant]
        if readout.mode != expected_readout:
            raise ConfigValidationError(
                f"variant {variant!r} requires readout.mode={expected_readout!r}"
            )
        if variant == "sft":
            if span is not None or depth != 1:
                raise ConfigValidationError("sft requires adaptive_span=null and train_depth=1")
            if conditioner.mode != "none" or adaln.mode != "none":
                raise ConfigValidationError("sft must not contain a conditioner or AdaLN")
        else:
            if span != PAPER_LOOP_SPAN:
                raise ConfigValidationError("conditioned recipes require adaptive_span=[3, 15]")
            expected_depth = 1 if variant == "adaln" else PAPER_TRAIN_DEPTH
            if depth != expected_depth:
                raise ConfigValidationError(
                    f"variant {variant!r} requires train_depth={expected_depth}"
                )
            if conditioner.mode not in {"question", "constant_question"}:
                raise ConfigValidationError(
                    f"variant {variant!r} requires question or constant_question conditioning"
                )
            if adaln.mode != "independent":
                raise ConfigValidationError(f"variant {variant!r} requires independent AdaLN")
        if variant not in RECURRENT_VARIANTS:
            if training.supervision != "terminal":
                raise ConfigValidationError(
                    f"variant {variant!r} requires terminal supervision"
                )
        elif training.supervision != "deep_geometric":
            raise ConfigValidationError(
                f"variant {variant!r} requires deep_geometric supervision"
            )
        if conditioner.mode == "none" and training.optimizer.conditioner_lr != 0.0:
            raise ConfigValidationError(
                "training.optimizer.conditioner_lr must be zero without conditioning"
            )
        if conditioner.mode != "none" and training.optimizer.conditioner_lr <= 0.0:
            raise ConfigValidationError(
                "training.optimizer.conditioner_lr must be positive with conditioning"
            )

        return cls(
            schema_version=SCHEMA_VERSION,
            backend=BACKEND,
            base_model=AssetConfig.from_dict(raw["base_model"], kind="base_model"),
            tokenizer=AssetConfig.from_dict(raw["tokenizer"], kind="tokenizer"),
            backbone=BackboneConfig.from_dict(raw["backbone"]),
            variant=variant,
            adaptive_span=span,
            train_depth=depth,
            vision_round_mode="latent_carry",
            lora=LoRAConfig.from_dict(raw["lora"]),
            conditioner=conditioner,
            adaln=adaln,
            readout=readout,
            preprocessing=PreprocessingConfig.from_dict(raw["preprocessing"]),
            training=training,
            generation=GenerationConfig.from_dict(raw["generation"]),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "PaliGemmaAdaLoopConfig":
        path = Path(path)
        try:
            raw = yaml.load(
                path.read_text(encoding="utf-8"), Loader=_StrictSafeLoader
            )
        except ConfigValidationError:
            raise
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ConfigValidationError(f"could not read config {path}: {exc}") from exc
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["adaptive_span"] = (
            list(self.adaptive_span) if self.adaptive_span is not None else None
        )
        result["lora"]["target_modules"] = list(self.lora.target_modules)
        # Omit optional fields from mode-specific blocks. This keeps the public
        # YAML canonical and lets strict parsing distinguish absent from null.
        result["base_model"] = {
            key: value for key, value in result["base_model"].items() if value is not None
        }
        for block in ("conditioner", "adaln", "readout"):
            result[block] = {
                key: value for key, value in result[block].items() if value is not None
            }
        return result

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
