"""Qwen2.5-VL model primitives used by the public AdaLoop backend."""

from .conditioning import AdaLNZero, QuestionAttentionPooler
from .losses import causal_cross_entropy_per_example
from .modeling import (
    AdaptiveConditionalGenerationMixin,
    AdaptiveQwen2_5_VLForConditionalGeneration,
    restore_missing_adaptive_initializers,
)
from .vision import (
    AdaLNQwen2_5_VLVisionBlock,
    AdaLNResidualBlockMixin,
    AdaptiveQwen2_5_VisionTransformer,
    AdaptiveVisionRuntimeMixin,
    AdaptiveVisionSpec,
    VisionRuntimeState,
)

__all__ = [
    "AdaLNZero",
    "AdaLNQwen2_5_VLVisionBlock",
    "AdaLNResidualBlockMixin",
    "AdaptiveConditionalGenerationMixin",
    "AdaptiveQwen2_5_VLForConditionalGeneration",
    "AdaptiveQwen2_5_VisionTransformer",
    "AdaptiveVisionRuntimeMixin",
    "AdaptiveVisionSpec",
    "QuestionAttentionPooler",
    "VisionRuntimeState",
    "causal_cross_entropy_per_example",
    "restore_missing_adaptive_initializers",
]
