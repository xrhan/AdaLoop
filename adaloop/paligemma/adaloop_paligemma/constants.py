"""Fixed dimensions and recipe values of the released PaliGemma2 AdaLoop models.

This module is framework-free: config validation, data preparation,
checkpoint inventories, and the Flax modules all read these values from here.
"""

# PaliGemma2-3B-Mix-448 vision tower (SigLIP So400m/14).
PAPER_IMAGE_SIZE = 448
PAPER_PATCH_SIZE = (14, 14)
PAPER_IMAGE_TOKENS = 1024
PAPER_VISION_VARIANT = "So400m/14"
PAPER_VISION_WIDTH = 1152
PAPER_VISION_DEPTH = 27
PAPER_VISION_MLP_DIM = 4304
PAPER_VISION_HEADS = 16

# Gemma-2 2B language model.
PAPER_LANGUAGE_VARIANT = "gemma2_2b"
GEMMA_WIDTH = 2304
GEMMA_DEPTH = 26
GEMMA_MLP_DIM = 9216
GEMMA_HEADS = 8
GEMMA_KV_HEADS = 4
GEMMA_HEAD_DIM = 256
GEMMA_VOCAB_SIZE = 257_152

# Model variants: plain SFT, single-pass AdaLN, and AdaLoop Variants A/B/C.
VARIANTS = frozenset({"sft", "adaln", "a", "b", "c"})
# A/B/C loop the tied vision band and supervise every depth.
RECURRENT_VARIANTS = frozenset({"a", "b", "c"})
# B/C pack the image readout of every depth into one language sequence.
MULTI_IMAGE_VARIANTS = frozenset({"b", "c"})

# AdaLoop recipe.
PAPER_LOOP_SPAN = (3, 15)  # tied SigLIP blocks [start, end)
PAPER_TRAIN_DEPTH = 3
PAPER_LORA_RANK = 64
PAPER_LORA_ALPHA = 128
PAPER_LORA_TARGETS = ("q", "kv", "o")
PAPER_QUESTION_MAX_TOKENS = 128
PAPER_CONDITION_DIM = 512
PAPER_CONDITION_HEADS = 8
PAPER_ADALN_HIDDEN_DIM = 128
PAPER_GEOMETRIC_RATIO = 0.8
PAPER_DRAFT_MAX_TOKENS = 24
# Bound on the packed Variant-B/C language sequence.  The paper's depth-5/6
# evaluations raised the training value (6144) to 8192: with 1024 image and
# 256 prompt slots, Variant C at depth 6 needs 7805 prompt slots.
PAPER_MAX_SEQUENCE_LENGTH = 8192

# Text preprocessing and generation.
PAPER_MAX_TEXT_TOKENS = 256
PAPER_EVAL_MAX_TEXT_TOKENS = 256
PAPER_MAX_NEW_TOKENS = 128
PAPER_TRAIN_PROMPT_TEMPLATE = "answer en {question}"
PAPER_EVAL_PROMPT_TEMPLATE = "{question}"
