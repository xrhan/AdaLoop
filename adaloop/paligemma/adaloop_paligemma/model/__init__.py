"""Public PaliGemma/JAX AdaLoop model primitives."""

from .conditioning import (
    AttentionPooler,
    IndependentAdaLNZero,
    QuestionConditioner,
)
from .gemma_lora import Gemma2LoRA, GemmaCore, add_lora_params
from .losses import (
    causal_cross_entropy_per_example,
    combine_deep_supervision,
)
from .modeling import (
    AdaLoopOutput,
    GenerationPrefill,
    PaliGemmaAdaLoop,
    PaliGemmaTokenIds,
    build_model_from_config,
)
from .readouts import (
    PackedReadout,
    make_attn_mask,
    pack_accumulated_images,
    pack_current_image,
)
from .trajectory import (
    TrajectoryBatch,
    gather_text_slots,
    greedy_reprefill_drafts,
    pack_trajectory,
    rollout_reprefill_with_apply,
)
from .vision import (
    AdaptiveEncoder1DBlock,
    LoopedEncoder,
    LoopedSigLIPVision,
    unstack_scanned_vision_params,
)

__all__ = [
    "AdaLoopOutput",
    "AdaptiveEncoder1DBlock",
    "AttentionPooler",
    "Gemma2LoRA",
    "GemmaCore",
    "GenerationPrefill",
    "IndependentAdaLNZero",
    "LoopedEncoder",
    "LoopedSigLIPVision",
    "PackedReadout",
    "PaliGemmaAdaLoop",
    "PaliGemmaTokenIds",
    "QuestionConditioner",
    "TrajectoryBatch",
    "add_lora_params",
    "build_model_from_config",
    "causal_cross_entropy_per_example",
    "combine_deep_supervision",
    "gather_text_slots",
    "greedy_reprefill_drafts",
    "make_attn_mask",
    "pack_accumulated_images",
    "pack_current_image",
    "pack_trajectory",
    "rollout_reprefill_with_apply",
    "unstack_scanned_vision_params",
]
