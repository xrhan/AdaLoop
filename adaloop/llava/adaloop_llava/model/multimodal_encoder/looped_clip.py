# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
# Copyright 2026 The AdaLoop Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the AdaLoop Authors from Transformers 4.57.6
# (https://github.com/huggingface/transformers) src/transformers/models/clip/modeling_clip.py: the
# forward of CLIPEncoderLayer gained an identity-preserving AdaLN-Zero modulation of both residual
# branches (AdaLNCLIPEncoderLayer). The question pooler, the round embedding, the loop core and
# the looped tower are new.

"""Question-conditioned looped CLIP vision tower.

The executed CLIP layers ``[0, exec_layers)`` (``exec_layers = 23`` for ViT-L/14 with
``select_layer=-2``) are split into three segments::

    prelude  layers [0, loop_start)            run once
    band     layers [loop_start, loop_end)     run K times with shared (frozen) weights
    coda     layers [loop_end, exec_layers)    run once on the readout state

Every band layer is an :class:`AdaLNCLIPEncoderLayer`: the frozen CLIP block plus an
AdaLN-Zero modulation driven by a question condition (the question tokens pooled by
:class:`QuestionAttentionPooler`, plus an optional sinusoidal step embedding). The modulation
is zero-initialised and gated as ``(1 + g)``, so at initialisation one band application is
bit-identical to the original CLIP block and the depth-1 output equals vanilla CLIP.

All new trainable parameters live in :class:`LoopCore`, which is registered on the LLaVA model
(``model.loop_core``); the tower only borrows the same module objects.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

from ..fp32 import FP32Module, fp32_layer_norm, fp32_linear, fp32_rms_norm
from ..loop_utils import (READOUT_MODES, STEP_EMBED_MODES, check_mode_consistency, check_readout,
                          clip_exec_layers, deep_supervision_weights, infer_loop_mode, loop_config_value)
from .clip_encoder import CLIPVisionTower, spatial_pool


# ----------------------------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------------------------

class AdaLNZero(FP32Module):
    """Condition -> six modulation vectors (shift/scale/gate for attention and MLP).

    LayerNorm -> Linear(condition_dim, hidden_dim) -> SiLU -> Linear(hidden_dim, 6 * width).
    The last projection is zero-initialised, so every modulation starts at exactly zero.
    """

    def __init__(self, condition_dim: int, vision_width: int, hidden_dim: int = 128):
        super().__init__()
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.to_adaln = nn.Linear(hidden_dim, 6 * vision_width)
        nn.init.zeros_(self.to_adaln.weight)
        nn.init.zeros_(self.to_adaln.bias)

    def forward(self, condition: torch.Tensor):
        h = fp32_linear(fp32_layer_norm(condition, self.condition_norm), self.condition_proj)
        return fp32_linear(F.silu(h), self.to_adaln).chunk(6, dim=-1)


class LoopStepEmbedder(FP32Module):
    """Sinusoidal embedding of the (0-based) band iteration index, added to the condition.

    ``[sin(step * f), cos(step * f)]`` with ``f = exp(-ln(max_period) * arange(F/2) / (F/2 - 1))``
    followed by Linear(F, D) -> SiLU -> Linear(D, D). There is no cap on the step index, so
    evaluating deeper than the training depth extrapolates the embedding.
    """

    def __init__(self, condition_dim: int, *, frequency_dim: int = 64, max_period: float = 32.0):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.frequency_dim, self.max_period = int(frequency_dim), float(max_period)
        self.mlp = nn.Sequential(nn.Linear(self.frequency_dim, self.condition_dim), nn.SiLU(),
                                 nn.Linear(self.condition_dim, self.condition_dim))
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def _sinusoidal(self, step: int, device) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(-math.log(self.max_period)
                          * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1))
        phase = float(step) * freqs
        emb = torch.cat((phase.sin(), phase.cos()), dim=0)
        if emb.numel() < self.frequency_dim:
            emb = F.pad(emb, (0, self.frequency_dim - emb.numel()))
        return emb

    def forward(self, step: int, *, device, dtype) -> torch.Tensor:
        if step < 0:
            raise ValueError(f"Loop step must be non-negative, got {step}")
        h = fp32_linear(self._sinusoidal(step, device), self.mlp[0])
        out = fp32_linear(F.silu(h), self.mlp[2])
        return out.to(device=device, dtype=dtype)


class QuestionAttentionPooler(FP32Module):
    """Pool question token embeddings into one ``[B, condition_dim]`` condition vector.

    RMSNorm -> Linear(input_dim, condition_dim) -> one learned query attends over the tokens
    (multi-head, masked) -> query residual -> RMSNorm + GELU MLP residual. Runs entirely in fp32
    with autocast disabled.
    """

    def __init__(self, input_dim: int, condition_dim: int = 512, num_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        if condition_dim % num_heads:
            raise ValueError("condition_dim must be divisible by num_heads")
        self.condition_dim, self.num_heads = int(condition_dim), int(num_heads)
        self.head_dim = self.condition_dim // self.num_heads
        self.input_norm = nn.RMSNorm(input_dim, eps=1e-6)
        self.input_proj = nn.Identity() if input_dim == condition_dim else nn.Linear(input_dim, condition_dim, bias=False)
        self.query = nn.Parameter(torch.empty(1, condition_dim))
        self.q_proj = nn.Linear(condition_dim, condition_dim, bias=False)
        self.k_proj = nn.Linear(condition_dim, condition_dim, bias=False)
        self.v_proj = nn.Linear(condition_dim, condition_dim, bias=False)
        self.out_proj = nn.Linear(condition_dim, condition_dim, bias=False)
        hidden = max(1, int(round(condition_dim * mlp_ratio)))
        self.mlp_norm = nn.RMSNorm(condition_dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(condition_dim, hidden), nn.GELU(), nn.Linear(hidden, condition_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, token_embeddings: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if token_embeddings.ndim != 3:
            raise ValueError(f"Question embeddings must be [B, T, D], got {tuple(token_embeddings.shape)}")
        if token_mask.shape != token_embeddings.shape[:2]:
            raise ValueError("Question mask does not match embeddings")
        token_mask = token_mask.bool()
        if torch.any(~token_mask.any(dim=1)):
            raise ValueError("Every question condition must contain at least one token")
        with torch.autocast(device_type=token_embeddings.device.type, enabled=False):
            x = fp32_rms_norm(token_embeddings, self.input_norm)
            x = x if isinstance(self.input_proj, nn.Identity) else fp32_linear(x, self.input_proj)
            b, t, _ = x.shape
            # The un-projected query is used both as the attention query and as a residual.
            query = self.query.float().unsqueeze(0).expand(b, -1, -1)
            q = fp32_linear(query, self.q_proj).view(b, 1, self.num_heads, self.head_dim)
            k = fp32_linear(x, self.k_proj).view(b, t, self.num_heads, self.head_dim)
            v = fp32_linear(x, self.v_proj).view(b, t, self.num_heads, self.head_dim)
            logits = torch.einsum("bqhd,bthd->bhqt", q, k) / math.sqrt(self.head_dim)
            logits = logits.masked_fill(~token_mask[:, None, None, :], torch.finfo(logits.dtype).min)
            w = torch.softmax(logits, dim=-1)
            pooled = fp32_linear(torch.einsum("bhqt,bthd->bqhd", w, v).reshape(b, 1, self.condition_dim),
                                 self.out_proj)
            pooled = query + pooled
            h = fp32_linear(F.gelu(fp32_linear(fp32_rms_norm(pooled, self.mlp_norm), self.mlp[0])), self.mlp[2])
            pooled = pooled + h
            return pooled.squeeze(1)


class DraftConditionFeedback(FP32Module):
    """Variant C: fold the previous round's draft into the next round's condition.

    ``a = answer_pooler(draft token embeddings)``, ``g = tanh(gate([q, a]))`` and the condition is
    ``q + g * a``. The gate is zero-initialised, so the condition starts as the question condition
    ``q``; a row whose draft ended at once (no kept token) gets exactly ``q``. Runs in fp32.
    """

    def __init__(self, input_dim: int, condition_dim: int = 512, num_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.input_dim, self.condition_dim = int(input_dim), int(condition_dim)
        self.answer_pooler = QuestionAttentionPooler(input_dim=self.input_dim, condition_dim=self.condition_dim,
                                                     num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.gate = nn.Linear(2 * self.condition_dim, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, question_condition: torch.Tensor, draft_token_embeddings: torch.Tensor,
                draft_token_mask: torch.Tensor) -> torch.Tensor:
        draft_token_mask = draft_token_mask.bool()
        has_draft = draft_token_mask.any(dim=1)
        # A row without draft tokens pools one masked-in slot (so every parameter takes part in the
        # graph on every process) and its pooled vector is then multiplied by zero.
        safe_mask = draft_token_mask.clone()
        safe_mask[~has_draft, 0] = True
        pooled = self.answer_pooler(draft_token_embeddings, safe_mask)
        with torch.autocast(device_type=draft_token_embeddings.device.type, enabled=False):
            q = question_condition.float()
            draft_condition = pooled * has_draft.to(dtype=pooled.dtype).unsqueeze(1)
            gate = torch.tanh(fp32_linear(torch.cat((q, draft_condition), dim=-1), self.gate))
            return q + gate * draft_condition


class LoopCore(nn.Module):
    """All trainable parameters of the loop, owned by the LLaVA model as ``model.loop_core``.

    * ``adaln[str(i)]`` -- one :class:`AdaLNZero` per band layer ``i``, shared by all K iterations;
    * ``question_conditioner`` -- :class:`QuestionAttentionPooler` over LM token embeddings;
    * ``loop_step_embedder`` -- :class:`LoopStepEmbedder`, or ``None`` when ``step_embed='none'``;
    * ``draft_feedback`` -- :class:`DraftConditionFeedback` for Variant C
      (``readout_mode='answer_trajectory'``), else ``None``.

    The tower adopts these same objects in ``load_model``. Keeping them on the model (registered
    before ``vision_tower``) makes their state-dict keys ``model.loop_core.*``, which is what
    checkpoints store and what ``from_pretrained`` reloads.
    """

    def __init__(self, spec: Dict[str, Any], vision_width: int):
        super().__init__()
        self.adaln = nn.ModuleDict({
            str(i): AdaLNZero(spec["condition_dim"], int(vision_width), spec["adaln_hidden_dim"])
            for i in range(spec["loop_start"], spec["loop_end"])})
        # The attribute names are the checkpoint key names: question_conditioner is the question
        # pooler, and draft_feedback.answer_pooler pools the draft tokens.
        self.question_conditioner = QuestionAttentionPooler(
            input_dim=int(spec["lm_hidden_size"]), condition_dim=spec["condition_dim"])
        self.loop_step_embedder = LoopStepEmbedder(spec["condition_dim"]) if spec["step_embed"] == "sinusoidal" else None
        # Built last, so the modules above draw the same initial values in every variant.
        self.draft_feedback = (DraftConditionFeedback(input_dim=int(spec["lm_hidden_size"]),
                                                      condition_dim=spec["condition_dim"])
                               if spec["readout_mode"] == "answer_trajectory" else None)
        # from_pretrained(torch_dtype=fp16/bf16) constructs modules under a low-precision default
        # dtype; the core must be born fp32 so that loading fp32 weights does not round them.
        self.float()


class AdaLNCLIPEncoderLayer(CLIPEncoderLayer):
    """CLIP encoder block with an identity-preserving AdaLN modulation on both residual branches.

    ``x = x + (1 + g_a) * attn(LN1(x) * (1 + s_a) + b_a)``
    ``x = x + (1 + g_m) * mlp(LN2(x) * (1 + s_m) + b_m)``

    With all modulations zero this is exactly the original pre-norm CLIP block. Called without
    ``vision_condition`` it falls back to the plain CLIP forward. ``adaln`` is the generator of
    this band layer owned by :class:`LoopCore`.
    """

    def __init__(self, config, adaln: AdaLNZero):
        super().__init__(config)
        self.adaln = adaln

    def __call__(self, *args, **kwargs):
        # Bypass transformers' GradientCheckpointingLayer.__call__. With gradient checkpointing
        # enabled it would wrap this layer in a checkpoint that only tracks positional inputs,
        # while the condition arrives as a keyword argument; with reentrant checkpointing the
        # loop core would silently receive no gradient. The band layers are small, so they are
        # simply not checkpointed (prelude/coda/LM layers still are).
        return nn.Module.__call__(self, *args, **kwargs)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale) + shift

    # Adapted from ``CLIPEncoderLayer.forward`` in Hugging Face transformers (Copyright The
    # HuggingFace Team, Apache-2.0): the same pre-norm block, with AdaLN-Zero modulation added.
    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None,
                output_attentions=False, vision_condition: Optional[torch.Tensor] = None):
        if vision_condition is None:
            return super().forward(hidden_states, attention_mask, causal_attention_mask, output_attentions)
        # The modulation generator runs in fp32 with autocast off; its six outputs are cast to
        # the residual-stream dtype at the boundary.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            mods = self.adaln(vision_condition.float())
        sa, sc_a, g_a, sm, sc_m, g_m = (m.to(hidden_states.dtype) for m in mods)
        residual = hidden_states
        h = self._modulate(self.layer_norm1(hidden_states), sa, sc_a)
        h, attn_w = self.self_attn(hidden_states=h, attention_mask=attention_mask,
                                   causal_attention_mask=causal_attention_mask,
                                   output_attentions=output_attentions)
        hidden_states = residual + (1 + g_a) * h
        residual = hidden_states
        h = self._modulate(self.layer_norm2(hidden_states), sm, sc_m)
        hidden_states = residual + (1 + g_m) * self.mlp(h)
        return (hidden_states, attn_w) if output_attentions else (hidden_states,)


# ----------------------------------------------------------------------------------------------
# Loop specification
# ----------------------------------------------------------------------------------------------

def make_clip_loop_spec(*, mode: str, loop_start: int, loop_end: int, loop_steps: int, select_layer: int,
                        condition_dim: int, adaln_hidden_dim: int, step_embed: str, condition_max_length: int,
                        readout_mode: str, deep_sup_weight: str, input_res: int, spatial_pool: int,
                        lm_hidden_size: int, fused_readout: bool) -> Dict[str, Any]:
    """Validate the loop settings and return the tower's ``loop_spec`` dict.

    ``exec_layers`` assumes the release's 24-layer CLIP tower here; ``load_model`` recomputes it
    from the CLIP checkpoint that is actually loaded.
    """
    exec_layers = clip_exec_layers(select_layer)
    if not 0 <= loop_start < loop_end <= exec_layers:
        raise ValueError(f"Loop band [{loop_start},{loop_end}) must lie inside the executed layers [0,{exec_layers})")
    if loop_steps < 1:
        raise ValueError(f"loop_steps must be >= 1, got {loop_steps}")
    if step_embed not in STEP_EMBED_MODES:
        raise ValueError(f"step_embed must be one of {STEP_EMBED_MODES}, got {step_embed!r}")
    if readout_mode not in READOUT_MODES:
        raise ValueError(f"readout_mode must be one of {READOUT_MODES}, got {readout_mode!r}")
    if readout_mode == "last":
        if not str(deep_sup_weight or "").strip():
            raise ValueError("deep_sup_weight must be set for loop models (e.g. 'geometric:0.6')")
        deep_supervision_weights(deep_sup_weight, loop_steps)    # fail early on a malformed spec
    if condition_max_length < 1:
        raise ValueError(f"condition_max_length must be >= 1, got {condition_max_length}")
    if input_res and input_res % 14:
        raise ValueError(f"input_res must be divisible by the patch size 14, got {input_res}")
    if spatial_pool > 1 and input_res:
        g = input_res // 14
        if g % spatial_pool:
            raise ValueError(f"{input_res}px gives a {g}x{g} patch grid, not divisible by spatial_pool={spatial_pool}")
    check_mode_consistency(mode, loop_steps, step_embed, fused_readout)
    check_readout(mode, readout_mode, deep_sup_weight, fused_readout)
    return dict(mode=str(mode), loop_start=int(loop_start), loop_end=int(loop_end), loop_steps=int(loop_steps),
                exec_layers=int(exec_layers), condition_dim=int(condition_dim),
                adaln_hidden_dim=int(adaln_hidden_dim), step_embed=str(step_embed),
                condition_max_length=int(condition_max_length), readout_mode=str(readout_mode),
                deep_sup_weight=str(deep_sup_weight), input_res=int(input_res), spatial_pool=int(spatial_pool),
                lm_hidden_size=int(lm_hidden_size))


def loop_spec_from_config(config) -> Dict[str, Any]:
    """Build the ``loop_spec`` from a model config (internal ``loop_*`` key names).

    Missing loop keys take :data:`~adaloop_llava.model.loop_utils.LOOP_CONFIG_DEFAULTS`.
    """
    mode = infer_loop_mode(config)
    if mode == "none":
        raise ValueError("loop_spec_from_config called on a config without a loop (loop mode 'none')")
    return make_clip_loop_spec(
        mode=mode, loop_start=int(config.loop_start), loop_end=int(config.loop_end),
        loop_steps=int(loop_config_value(config, "loop_steps")),
        select_layer=int(getattr(config, "mm_vision_select_layer", -2)),
        condition_dim=int(loop_config_value(config, "loop_condition_dim")),
        adaln_hidden_dim=int(loop_config_value(config, "loop_adaln_hidden_dim")),
        step_embed=str(loop_config_value(config, "loop_step_embed")),
        condition_max_length=int(loop_config_value(config, "condition_max_length")),
        readout_mode=str(loop_config_value(config, "loop_readout_mode")),
        deep_sup_weight=str(getattr(config, "loop_deep_sup_weight", None) or ""),
        input_res=int(getattr(config, "vision_input_res", 0) or 0),
        spatial_pool=int(getattr(config, "vision_spatial_pool", 1) or 1),
        lm_hidden_size=int(getattr(config, "lm_hidden_size", None) or getattr(config, "hidden_size", 4096)),
        fused_readout=bool(loop_config_value(config, "loop_fused_readout")))


# ----------------------------------------------------------------------------------------------
# Tower
# ----------------------------------------------------------------------------------------------

class LoopedCLIPVisionTower(CLIPVisionTower):
    """CLIP tower with a question-conditioned, weight-tied band (single pass when K = 1).

    The caller (``LlavaMetaForCausalLM.encode_images``) passes only images; the condition and the
    depth are set beforehand with :meth:`set_active_condition` / :meth:`set_active_depth`.
    """

    def __init__(self, vision_tower, args, delay_load=False):
        # The core belongs to the LLaVA model; keep only a reference that nn.Module does not register.
        object.__setattr__(self, "_loop_core", None)
        self.loop_spec = loop_spec_from_config(args)
        self._active_condition: Optional[torch.Tensor] = None
        self._active_depth: Optional[int] = None
        self._collect = False
        self._last_step_features: Optional[List[torch.Tensor]] = None
        super().__init__(vision_tower, args, delay_load=delay_load)

    def attach_loop_core(self, core: LoopCore):
        """Attach the model-owned core; must happen before ``load_model`` (which adopts it)."""
        if self.is_loaded:
            raise RuntimeError("attach_loop_core must be called before load_model (adoption happens there)")
        object.__setattr__(self, "_loop_core", core)

    def load_model(self, device_map=None):
        """Load CLIP, then swap the band layers for AdaLN layers that adopt the core's generators."""
        if self.is_loaded:
            return
        core = self._loop_core
        if core is None:
            raise RuntimeError("attach_loop_core() must be called before load_model()")
        super().load_model(device_map=device_map)           # frozen CLIP, retargeted to input_res
        spec, vm = self.loop_spec, self.vision_tower.vision_model
        cfg = self.vision_tower.config
        # exec_layers follows the CLIP checkpoint that was actually loaded.
        spec["exec_layers"] = clip_exec_layers(int(self.select_layer), cfg.num_hidden_layers)
        if not 0 <= spec["loop_start"] < spec["loop_end"] <= spec["exec_layers"]:
            raise ValueError(f"Loop band [{spec['loop_start']},{spec['loop_end']}) must lie inside the executed "
                             f"layers [0,{spec['exec_layers']}) of a {cfg.num_hidden_layers}-layer tower")
        res = int(spec.get("input_res", 0) or 0)
        if res and cfg.image_size != res:
            raise RuntimeError(f"input_res={res} but the tower's image_size is {cfg.image_size}: "
                               "vision_input_res did not reach the tower")
        dev = vm.embeddings.patch_embedding.weight.device
        core.to(dev)                                         # device only; the core stays fp32
        for i in range(spec["loop_start"], spec["loop_end"]):
            new = AdaLNCLIPEncoderLayer(cfg, core.adaln[str(i)])
            missing, unexpected = new.load_state_dict(vm.encoder.layers[i].state_dict(), strict=False)
            assert not unexpected and all(k.startswith("adaln.") for k in missing), (missing, unexpected)
            new.to(device=vm.encoder.layers[i].layer_norm1.weight.device,
                   dtype=vm.encoder.layers[i].layer_norm1.weight.dtype)   # adaln is fp32-pinned
            for n, p in new.named_parameters():
                if not n.startswith("adaln."):
                    p.requires_grad = False                    # CLIP weights stay frozen
            vm.encoder.layers[i] = new
        # Borrow (not copy) the core's modules.
        self.question_conditioner = core.question_conditioner
        self.loop_step_embedder = core.loop_step_embedder

    # -- runtime state (reset by the model on every forward / generate) --
    def set_active_condition(self, condition):
        self._active_condition = condition

    def set_active_depth(self, depth: Optional[int]):
        """Number of band applications of the next forward; ``None`` means the trained ``loop_steps``."""
        self._active_depth = depth

    def set_collect_loop_step_features(self, flag):
        self._collect = bool(flag)

    def take_loop_step_features(self):
        out, self._last_step_features = self._last_step_features, None
        return out

    @property
    def active_depth(self) -> int:
        return int(self._active_depth or self.loop_spec["loop_steps"])

    def _cond_for_step(self, cond, step):
        if self.loop_step_embedder is None:
            return cond
        return cond + self.loop_step_embedder(step, device=cond.device, dtype=cond.dtype).unsqueeze(0)

    # -- segments --
    def prelude(self, images):
        vm = self.vision_tower.vision_model
        x = images.to(device=self.device, dtype=self.dtype)
        h = vm.pre_layrnorm(vm.embeddings(x))
        for i in range(self.loop_spec["loop_start"]):
            h = vm.encoder.layers[i](h, None, None)[0]
        return h

    def band_step(self, h, cond, step: int):
        """One application of the band. ``cond``: ``[B, condition_dim]``; ``step`` is 0-based."""
        spec = self.loop_spec
        layers = self.vision_tower.vision_model.encoder.layers
        cond = cond.to(device=h.device, dtype=torch.float32)
        c = self._cond_for_step(cond, step)
        # The condition is broadcast to every token and kept fp32; the layer casts at its boundary.
        pc = c[:, None, :].expand(h.shape[0], h.shape[1], -1)
        for i in range(spec["loop_start"], spec["loop_end"]):
            h = layers[i](h, None, None, vision_condition=pc)[0]
        return h

    def coda(self, h, out_dtype=None):
        spec = self.loop_spec
        layers = self.vision_tower.vision_model.encoder.layers
        for i in range(spec["loop_end"], spec["exec_layers"]):
            h = layers[i](h, None, None)[0]
        r = self._select(h)
        return r if out_dtype is None else r.to(out_dtype)

    def forward(self, images):
        """Run the tower. Returns the readout of the last iteration.

        With feature collection on (deep supervision), the coda also runs on every intermediate
        iterate and the K readouts are kept for :meth:`take_loop_step_features`.
        """
        cond = self._active_condition
        if cond is None:
            raise RuntimeError("No question condition set: call set_active_condition() before forward()")
        h = self.prelude(images)
        core_states = []
        for step in range(self.active_depth):
            h = self.band_step(h, cond, step)
            if self._collect:
                core_states.append(h)
        if self._collect:
            self._last_step_features = [self.coda(s, images.dtype) for s in core_states]
            return self._last_step_features[-1]
        return self.coda(h, images.dtype)

    def _select(self, hidden):
        return spatial_pool(hidden[:, 1:], self.spatial_pool)
