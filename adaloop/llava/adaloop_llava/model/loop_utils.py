"""Pure-Python helpers for the looped vision encoder (no torch dependency)."""

from __future__ import annotations

import math
from typing import Tuple

#: The single setting switch (``model.loop.mode`` in the YAML config, ``loop_mode`` in config.json).
LOOP_MODES = ("none", "single_pass", "adaloop")

#: Step-embedding modes supported by this release.
STEP_EMBED_MODES = ("sinusoidal", "none")

#: How the language model reads the loop (``model.loop.readout``, ``loop_readout_mode`` in
#: config.json): the current round only (Variant A), every round's readout as one image prefix
#: (Variant B), or every round's readout interleaved with the drafts of the earlier rounds
#: (Variant C).
READOUT_MODES = ("last", "image_prefix", "answer_trajectory")

#: Variant C: greedy draft tokens decoded after every round but the last, and the token that
#: separates a draft from the next round (13 is "\n" in the LLaMA vocabulary).
TRAJ_DRAFT_TOKENS = 16
TRAJ_SEPARATOR_TOKEN_ID = 13

#: Values of the loop keys a model config (``config.json``) does not set, or sets to null: the
#: settings the released checkpoints were trained with. ``loop_step_embed_max_steps`` is recorded
#: for provenance only; the sinusoidal step embedding has no cap on the step index.
LOOP_CONFIG_DEFAULTS = {
    "loop_steps": 3,
    "loop_step_embed": "sinusoidal",
    "loop_fused_readout": False,
    "loop_readout_mode": "last",
    "loop_use_adaln": True,
    "loop_condition_dim": 512,
    "loop_adaln_hidden_dim": 128,
    "loop_step_embed_max_steps": 16,
    "condition_max_length": 128,
}

#: Number of transformer layers of the CLIP ViT-L/14 tower used by the release.
CLIP_NUM_LAYERS = 24


def loop_config_value(config, key: str):
    """``config.<key>``, or its :data:`LOOP_CONFIG_DEFAULTS` value when absent or null."""
    value = getattr(config, key, None)
    return LOOP_CONFIG_DEFAULTS[key] if value is None else value


def clip_exec_layers(select_layer: int, num_layers: int = CLIP_NUM_LAYERS) -> int:
    """Number of CLIP layers executed to read ``hidden_states[select_layer]``."""
    return num_layers + 1 + select_layer if select_layer < 0 else select_layer


def deep_supervision_weights(spec: str, steps: int) -> Tuple[float, ...]:
    """Normalised per-iteration loss weights, shallowest iteration first.

    ``geometric:GAMMA`` gives ``raw_i = GAMMA ** (steps - 1 - i)`` (the last iteration has the
    largest weight), normalised to sum to one: ``geometric:0.6`` with 3 steps gives
    ``(9/49, 15/49, 25/49)`` and with 1 step exactly ``(1.0,)``. ``uniform`` and
    ``explicit:W1,...,WK`` are also accepted.
    """
    if steps < 1:
        raise ValueError(f"Deep supervision requires at least one step, got {steps}")
    value = str(spec).strip().lower()
    if value == "uniform":
        raw = [1.0] * steps
    elif value.startswith("geometric:"):
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("Deep-supervision geometric format is geometric:GAMMA")
        gamma = float(parts[1])
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError(f"Deep-supervision gamma must be finite and > 0, got {gamma}")
        raw = [gamma ** (steps - 1 - index) for index in range(steps)]
    elif value.startswith("explicit:"):
        raw = [float(item.strip()) for item in value[len("explicit:"):].split(",")]
        if len(raw) != steps:
            raise ValueError(f"Explicit deep-supervision weights need {steps} values, got {len(raw)}")
        if any(not math.isfinite(w) or w < 0.0 for w in raw):
            raise ValueError("Explicit deep-supervision weights must be finite and non-negative")
    else:
        raise ValueError("deep_sup_weight must be uniform, geometric:GAMMA or explicit:W1,...,WK; "
                         f"got {spec!r}")
    total = sum(raw)
    if total <= 0.0:
        raise ValueError("Deep-supervision weights must contain a positive value")
    return tuple(w / total for w in raw)


def check_mode_consistency(mode: str, steps, step_embed, fused_readout) -> None:
    """Reject loop settings that contradict ``mode``.

    * ``single_pass``: exactly one band application, no step embedding (a sinusoidal embedding
      at step 0 trains only half of its first layer), no fused readout.
    * ``adaloop``: at least two band applications.
    """
    if mode not in LOOP_MODES:
        raise ValueError(f"loop mode must be one of {LOOP_MODES}, got {mode!r}")
    if mode == "none":
        return
    if step_embed not in STEP_EMBED_MODES:
        raise ValueError(f"step_embed must be one of {STEP_EMBED_MODES}, got {step_embed!r}")
    if mode == "single_pass":
        if int(steps) != 1:
            raise ValueError(f"loop mode 'single_pass' requires steps == 1, got {steps}")
        if step_embed != "none":
            raise ValueError(f"loop mode 'single_pass' requires step_embed 'none', got {step_embed!r}")
        if bool(fused_readout):
            raise ValueError("loop mode 'single_pass' requires fused_readout false (there is one readout)")
    elif mode == "adaloop":
        if int(steps) < 2:
            raise ValueError(f"loop mode 'adaloop' requires steps >= 2, got {steps} "
                             "(use mode 'single_pass' for one band application)")


def check_readout(mode: str, readout: str, deep_sup_weight, fused_readout) -> None:
    """Reject a readout that contradicts the loop settings.

    * ``last`` (Variant A, and the single-pass control): deep supervision over the rounds.
    * ``image_prefix`` (Variant B) and ``answer_trajectory`` (Variant C): an AdaLoop mode, one
      language-model pass over all rounds, hence no deep supervision and no fused readout.
    """
    if readout not in READOUT_MODES:
        raise ValueError(f"readout must be one of {READOUT_MODES}, got {readout!r}")
    if readout == "last":
        return
    if mode != "adaloop":
        raise ValueError(f"readout {readout!r} requires loop mode 'adaloop', got {mode!r}")
    if str(deep_sup_weight or "").strip():
        raise ValueError(f"readout {readout!r} trains with one language-model pass; deep_sup_weight must be "
                         f"empty, got {deep_sup_weight!r}")
    if bool(fused_readout):
        raise ValueError(f"readout {readout!r} requires fused_readout false")


def infer_loop_mode(config) -> str:
    """Loop mode of a model config (``LlavaConfig`` or any object with the same attributes).

    New checkpoints carry ``loop_mode`` explicitly. Older ones only have ``loop_start``: it is
    ``None`` for plain LLaVA / SFT checkpoints (which may still carry inert ``loop_steps``), and
    the number of band applications then separates single-pass from AdaLoop.
    """
    mode = getattr(config, "loop_mode", None)
    loop_start = getattr(config, "loop_start", None)
    if mode is not None:
        if mode not in LOOP_MODES:
            raise ValueError(f"config.loop_mode must be one of {LOOP_MODES}, got {mode!r}")
        if (mode == "none") != (loop_start is None):
            raise ValueError(f"config.loop_mode={mode!r} contradicts config.loop_start={loop_start!r}")
        return mode
    if loop_start is None:
        return "none"
    return "single_pass" if int(loop_config_value(config, "loop_steps")) == 1 else "adaloop"
