"""AdaLoop for LLaVA-1.5: a question-conditioned looped CLIP vision encoder.

The backend covers the three settings the paper trained on LLaVA-1.5-7B (SFT, single-pass AdaLN
and Variant A), Variants B and C, and zero-shot evaluation. A recipe (``configs/sft.yaml``,
``configs/adaln.yaml`` or ``configs/variant_{a,b,c}.yaml``) selects the setting with ``variant``
and the matching ``model.loop.mode`` and ``model.loop.readout`` (see :mod:`adaloop_llava.config`).

Importing the package loads only the torch-free surface: the recipe schema, the typed errors and
the portable checkpoint metadata (:mod:`adaloop_llava.checkpoint`). The model classes
(``LlavaConfig``, ``LlavaLlamaForCausalLM``) and the inference pipeline
(``AdaLoopLlavaPipeline``) are imported on first access.
"""

from importlib import import_module

from .config import (
    BACKEND,
    SCHEMA_VERSION,
    VARIANTS,
    Config,
    EvalConfig,
    load_config,
    load_eval_config,
)
from .checkpoint import CheckpointManifest, LlavaCheckpointLoader
from .errors import AdaLoopLlavaError, CheckpointValidationError, ConfigValidationError

__version__ = "1.0.0"

#: Exports whose modules import torch / transformers, by defining module.
_LAZY_EXPORTS = {"LlavaConfig": ".model", "LlavaLlamaForCausalLM": ".model",
                 "AdaLoopLlavaPipeline": ".inference"}


def __getattr__(name: str):
    """Import the model classes and the pipeline (torch, transformers) only when they are used."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "AdaLoopLlavaError",
    "AdaLoopLlavaPipeline",
    "BACKEND",
    "CheckpointManifest",
    "CheckpointValidationError",
    "Config",
    "ConfigValidationError",
    "EvalConfig",
    "LlavaCheckpointLoader",
    "LlavaConfig",
    "LlavaLlamaForCausalLM",
    "SCHEMA_VERSION",
    "VARIANTS",
    "load_config",
    "load_eval_config",
]
