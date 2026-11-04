"""Public contracts for the PaliGemma2/JAX AdaLoop backend.

Importing the package does not initialize JAX or select an accelerator. Model
and pipeline symbols are loaded lazily on first access.
"""

from importlib import import_module

from .checkpoint import (
    PaliGemmaCheckpointLoader,
    VerifiedPaliGemmaCheckpoint,
    export_parameter_tree,
)
from .config import PaliGemmaAdaLoopConfig
from .errors import (
    AdaLoopPaliGemmaError,
    CheckpointValidationError,
    ConfigValidationError,
    ReconstructionError,
)

__version__ = "1.0.0"

# Symbols that need JAX, by the submodule that defines them.
_LAZY_EXPORTS = {
    "AdaLoopOutput": ".model",
    "AdaLoopPaliGemmaPipeline": ".inference",
    "GenerationPrefill": ".model",
    "PaliGemmaAdaLoop": ".model",
    "PaliGemmaTokenIds": ".model",
    "build_model_from_config": ".model",
}


def __getattr__(name: str):
    """Load JAX symbols without making metadata imports select a device."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "AdaLoopOutput",
    "AdaLoopPaliGemmaError",
    "AdaLoopPaliGemmaPipeline",
    "CheckpointValidationError",
    "ConfigValidationError",
    "GenerationPrefill",
    "PaliGemmaAdaLoop",
    "PaliGemmaAdaLoopConfig",
    "PaliGemmaCheckpointLoader",
    "PaliGemmaTokenIds",
    "ReconstructionError",
    "VerifiedPaliGemmaCheckpoint",
    "build_model_from_config",
    "export_parameter_tree",
]
