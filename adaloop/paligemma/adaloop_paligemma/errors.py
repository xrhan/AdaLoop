"""Typed public exceptions for the PaliGemma2/JAX backend."""


class AdaLoopPaliGemmaError(Exception):
    """Base class for public PaliGemma backend errors."""


class ConfigValidationError(AdaLoopPaliGemmaError, ValueError):
    """Raised when a PaliGemma AdaLoop recipe is malformed."""


class CheckpointValidationError(AdaLoopPaliGemmaError, ValueError):
    """Raised when a checkpoint or an export is invalid."""


class ReconstructionError(AdaLoopPaliGemmaError, RuntimeError):
    """Raised when a runtime hook cannot reconstruct the pinned model."""
