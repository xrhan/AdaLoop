"""Typed public exceptions for the Qwen backend."""


class AdaLoopQwenError(Exception):
    """Base class for public Qwen backend errors."""


class ConfigValidationError(AdaLoopQwenError, ValueError):
    """Raised when an AdaLoop recipe is malformed or internally inconsistent."""


class CheckpointValidationError(AdaLoopQwenError, ValueError):
    """Raised when released checkpoint metadata or files are invalid."""
