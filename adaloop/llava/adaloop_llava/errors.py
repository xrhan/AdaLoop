"""Typed public exceptions for the LLaVA-1.5 backend."""


class AdaLoopLlavaError(Exception):
    """Base class for public LLaVA backend errors."""


class ConfigValidationError(AdaLoopLlavaError, ValueError):
    """Raised when a LLaVA recipe or evaluation config is malformed or internally inconsistent."""


class CheckpointValidationError(AdaLoopLlavaError, ValueError):
    """Raised when portable checkpoint metadata or files are invalid."""
