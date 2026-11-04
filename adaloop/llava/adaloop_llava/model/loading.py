# Copyright 2023 Haotian Liu
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
# Modified by the AdaLoop Authors from LLaVA (https://github.com/haotian-liu/LLaVA)
# llava/model/builder.py: loading for training and for evaluation (LoRA adapter plus non-LoRA
# trainables, read from non_lora_trainables.bin or non_lora_trainables.safetensors, or a full
# model), reload and verification of the fp32 loop core, normalization of checkpoint configs
# without loop_mode, and LoRA target selection.

"""Model loading for training and evaluation."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

from ..config import lm_lora_target_regex
from .llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from .loop_utils import (TRAJ_DRAFT_TOKENS, TRAJ_SEPARATOR_TOKEN_ID, check_mode_consistency, check_readout,
                         infer_loop_mode, loop_config_value)

logger = logging.getLogger(__name__)

NON_LORA_FILE = "non_lora_trainables.bin"
#: The same tensors in safetensors format (portable checkpoints; read when the .bin is absent).
NON_LORA_SAFETENSORS_FILE = "non_lora_trainables.safetensors"


# ----------------------------------------------------------------------------------------------
# LoRA
# ----------------------------------------------------------------------------------------------

def make_lora_config(cfg):
    """``peft.LoraConfig`` for a release config (``adaloop_llava.config.Config``)."""
    from peft import LoraConfig
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        target_modules=lm_lora_target_regex(cfg.lora.target_modules),
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )


# ----------------------------------------------------------------------------------------------
# Artifact keys
# ----------------------------------------------------------------------------------------------

def non_lora_trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    """Trainable non-LoRA parameters (projector + loop core) as saved in ``non_lora_trainables.bin``.

    Names are those of ``model.named_parameters()`` on the PEFT-wrapped model, e.g.
    ``base_model.model.model.loop_core.adaln.2.to_adaln.weight``.
    """
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters()
            if p.requires_grad and "lora_" not in k}


def normalize_artifact_keys(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map keys saved from a PEFT-wrapped model onto the bare ``LlavaLlamaForCausalLM``:
    strip ``base_model.``, then one leading ``model.`` if any key starts with ``model.model.``."""
    out = {(k[11:] if k.startswith("base_model.") else k): v for k, v in raw.items()}
    if any(k.startswith("model.model.") for k in out):
        out = {(k[6:] if k.startswith("model.") else k): v for k, v in out.items()}
    return out


def core_key(k: str) -> Optional[str]:
    """Map an artifact key ``...loop_core.<rest>`` to ``<rest>`` (relative to ``LoopCore``);
    ``None`` if it is not a core key."""
    if ".loop_core." in k or k.startswith("loop_core."):
        return k.split("loop_core.", 1)[1]
    return None


# ----------------------------------------------------------------------------------------------
# Config compatibility
# ----------------------------------------------------------------------------------------------

# Config keys of features outside this release (including the other image layouts and the image
# start/end and patch tokens); they must hold their inert value.
_INERT_CONFIG_KEYS = {
    "loop_truncated_bptt": False,
    "loop_prefix_deep_sup": False,
    "loop_traj_deep_sup": False,
    "loop_adaln_untied": False,
    "loop_per_row_loss": False,
    "loop_eval_steps": 0,
    "tune_vision": False,
    "image_aspect_ratio": "pad",
    "mm_use_im_start_end": False,
    "mm_use_im_patch_token": False,
}

# Variant C keys: the drafts enter the language-model context (traj_text), with this draft length
# and separator; any other value describes a model this release does not implement.
_TRAJECTORY_CONFIG_KEYS = {
    "loop_traj_text": True,
    "loop_draft_max_new_tokens": TRAJ_DRAFT_TOKENS,
    "loop_traj_sep_token_id": TRAJ_SEPARATOR_TOKEN_ID,
}


def _check_readout_keys(config, mode: str) -> None:
    """``loop_readout_mode`` and the Variant C keys must describe a readout of this release."""
    readout = loop_config_value(config, "loop_readout_mode")
    if mode != "none":
        check_readout(mode, readout, getattr(config, "loop_deep_sup_weight", None),
                      loop_config_value(config, "loop_fused_readout"))
    elif readout != "last":
        raise ValueError(f"Checkpoint config has loop_readout_mode={readout!r} without a loop")
    if readout == "answer_trajectory":
        for key, want in _TRAJECTORY_CONFIG_KEYS.items():
            value = getattr(config, key, None)
            if value != want:
                raise ValueError(f"Checkpoint config has {key}={value!r}; Variant C requires {want!r}")
    elif getattr(config, "loop_traj_text", None) not in (None, False):
        # The draft length and separator keys are inert outside Variant C.
        raise ValueError(f"Checkpoint config has loop_traj_text=True but loop_readout_mode={readout!r}")


def normalize_model_config(config):
    """Normalise a checkpoint config in place and return it.

    * ``loop_mode`` is added: ``loop_start is None`` -> ``none`` (even with inert ``loop_steps``),
      ``loop_steps == 1`` -> ``single_pass``, otherwise ``adaloop``.
    * Missing ``loop_use_adaln`` -> ``True``; missing ``loop_step_embed`` -> ``sinusoidal`` (their
      :data:`~adaloop_llava.model.loop_utils.LOOP_CONFIG_DEFAULTS`); ``loop_train_steps_dist`` must
      be ``fixed``.
    * Keys of features outside this release must hold their inert values; ``loop_readout_mode``
      and the Variant C keys must describe Variant A, B or C.
    """
    mode = infer_loop_mode(config)
    for key, inert in _INERT_CONFIG_KEYS.items():
        value = getattr(config, key, None)
        if value is not None and value != inert:
            raise ValueError(f"Checkpoint config has {key}={value!r}; this release only supports {inert!r}")
    _check_readout_keys(config, mode)
    if mode != "none":
        config.loop_use_adaln = loop_config_value(config, "loop_use_adaln")
        if not config.loop_use_adaln:
            raise ValueError("Checkpoint config has loop_use_adaln=False (unconditioned loop); not supported")
        steps_dist = getattr(config, "loop_train_steps_dist", None)
        if steps_dist not in (None, "fixed"):
            raise ValueError(f"Checkpoint config has loop_train_steps_dist={steps_dist!r}; only 'fixed' is supported")
        config.loop_step_embed = loop_config_value(config, "loop_step_embed")
        check_mode_consistency(mode, loop_config_value(config, "loop_steps"), config.loop_step_embed,
                               loop_config_value(config, "loop_fused_readout"))
    config.loop_mode = mode
    return config


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------

def load_base_for_train(cfg, torch_dtype=torch.bfloat16):
    """Load the stage-2 LLaVA base for training.

    The base config is patched in memory (never on disk): ``mm_vision_tower`` points to
    ``cfg.paths.vision_tower`` and the tower geometry comes from the release config. If the base
    config already records a geometry it must agree. The base must be a plain LLaVA model; the
    loop modules are added later by ``initialize_vision_modules``. Loading in ``torch_dtype`` with
    ``low_cpu_mem_usage`` materialises the projector in that dtype (it is upcast to fp32 later).
    """
    base = cfg.paths.base_model
    config = LlavaConfig.from_pretrained(base)
    if infer_loop_mode(config) != "none":
        raise ValueError(f"{base} already contains a loop tower; training starts from a plain LLaVA base")
    if not getattr(config, "mm_vision_tower", None):
        raise ValueError(f"{base} is not a LLaVA checkpoint (no mm_vision_tower in its config)")
    if getattr(config, "mm_projector_type", None) != cfg.model.projector:
        raise ValueError(f"{base} has mm_projector_type={getattr(config, 'mm_projector_type', None)!r}, "
                         f"the config asks for {cfg.model.projector!r}")
    config.mm_vision_tower = cfg.paths.vision_tower
    vis = cfg.model.vision
    for key, want in (("vision_input_res", vis.input_res), ("vision_spatial_pool", vis.spatial_pool),
                      ("mm_vision_select_layer", vis.select_layer), ("mm_vision_select_feature", vis.select_feature)):
        have = getattr(config, key, None)
        if have is not None and str(have) != str(want):
            raise ValueError(f"{base}/config.json has {key}={have!r} but the config asks for {want!r}")
        setattr(config, key, want)
    return LlavaLlamaForCausalLM.from_pretrained(
        base, config=config, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        attn_implementation=cfg.model.attn_implementation)


def mark_vision_trainables(model) -> None:
    """Make the projector and the loop core trainable (attaching LoRA freezes every other parameter);
    the CLIP weights stay frozen."""
    inner = model.get_model()
    for p in inner.mm_projector.parameters():
        p.requires_grad = True
    core = getattr(inner, "loop_core", None)
    if core is not None:
        for p in core.parameters():
            p.requires_grad = True


def assert_spec_matches(cfg, tower) -> None:
    """The tower that was built must be the one the config describes."""
    loop = cfg.model.loop
    spec = getattr(tower, "loop_spec", None)
    if loop.mode == "none":
        if spec is not None:
            raise RuntimeError("loop mode 'none' but the model has a loop tower")
        return
    want = dict(mode=loop.mode, loop_start=loop.band[0], loop_end=loop.band[1], loop_steps=loop.steps,
                step_embed=loop.step_embed, readout_mode=loop.readout,
                input_res=cfg.model.vision.input_res, spatial_pool=cfg.model.vision.spatial_pool,
                deep_sup_weight=loop.deep_sup_weight, condition_max_length=cfg.data.condition_max_length)
    got = {k: (spec or {}).get(k) for k in want}
    if got != want:
        raise RuntimeError(f"Loop tower spec {got} does not match the config {want}")


def training_dtypes(bf16: bool) -> Tuple[torch.dtype, torch.dtype]:
    """``(language-model dtype, CLIP tower dtype)`` of a training run, as in the paper recipe:
    bf16 for both with ``train.bf16``; otherwise an fp32 language model with an fp16 CLIP tower
    (the tower is cast to fp16 whenever bf16 is off). The projector, the loop core
    and the LoRA adapters are fp32 either way."""
    if bf16:
        return torch.bfloat16, torch.bfloat16
    return torch.float32, torch.float16


def build_model_for_training(cfg, *, world_size: int, torch_dtype=torch.bfloat16,
                             vision_dtype: Optional[torch.dtype] = None):
    """Build the trainable model for a release config; returns the PEFT-wrapped model.

    Order: base model (``torch_dtype``) -> ``use_cache=False`` -> input-grad hook -> cast to
    ``torch_dtype`` (rounds the rotary ``inv_freq`` buffer) -> LoRA on the LM projections ->
    vision modules (loop core + retargeted CLIP; projector trainable in fp32) -> CLIP cast to
    ``vision_dtype`` (default ``torch_dtype``; the loop core stays fp32) -> spec check ->
    non-reentrant gradient checkpointing on every checkpointable layer, including the CLIP layers
    created after LoRA was attached (the band layers opt out, see
    ``AdaLNCLIPEncoderLayer.__call__``).

    ``world_size`` (number of data-parallel processes) sets the loss normalisation of the loop
    path, see :meth:`LlavaLlamaForCausalLM.set_loss_normalization`. The training entry point
    passes ``vision_dtype`` as the paper recipe chose it (:func:`training_dtypes`).
    """
    from peft import get_peft_model

    from ..config import rank_mean_factor

    model = load_base_for_train(cfg, torch_dtype=torch_dtype)
    model.config.use_cache = False
    model.config.tokenizer_padding_side = "right"
    model.config.tokenizer_model_max_length = int(cfg.model.model_max_length)
    model.set_loss_normalization(cfg.train.loss_normalization, rank_mean_factor(cfg, world_size))
    if cfg.train.gradient_checkpointing:
        model.enable_input_require_grads()
    # Cast to the training dtype before attaching LoRA, as the paper recipe did. The weights are
    # already in that dtype; the cast also rounds floating buffers, notably the language model's
    # rotary inv_freq (fp32 after from_pretrained) to bf16, which the trained checkpoints saw. The
    # fp32-pinned projector only gets upcast (losslessly) by it.
    model.to(torch_dtype)
    model = get_peft_model(model, make_lora_config(cfg))
    base = model.get_base_model()
    base.get_model().initialize_vision_modules(cfg)
    tower = base.get_vision_tower()
    tower.to(dtype=torch_dtype if vision_dtype is None else vision_dtype)
    mark_vision_trainables(base)
    assert_spec_matches(cfg, tower)
    if cfg.train.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------

def reload_loop_core(model, raw: Dict[str, torch.Tensor], src: str = "artifact") -> int:
    """Reload the loop core in fp32 from artifact tensors and verify it; returns the tensor count.

    A low-precision ``from_pretrained`` materialises the core in that precision, and loading fp32
    weights into it would round them. The core is therefore moved back to fp32 and reloaded with a
    strict key check, after which the parameters must equal the file bit for bit. An all-zero
    ``to_adaln`` (an untrained core) is refused, and the tower must use the core's own modules.
    """
    core = getattr(model.get_model(), "loop_core", None)
    tower = model.get_vision_tower()
    if core is None:
        raise RuntimeError("The model has no loop_core (loop_start missing from its config)")
    file_sd = {}
    for k, v in raw.items():
        ck = core_key(k)
        if ck is not None:
            file_sd[ck] = v
    n_core = len(core.state_dict())
    if len(file_sd) != n_core:
        raise RuntimeError(f"Loop-core key count mismatch: {src} maps to {len(file_sd)} core tensors, "
                           f"the model's core has {n_core}")
    core.float()
    core.load_state_dict({k: v.float() for k, v in file_sd.items()}, strict=True)
    cur = core.state_dict()
    bad = [k for k, v in file_sd.items() if not torch.equal(cur[k].detach().cpu().float(), v.float())]
    if bad:
        raise RuntimeError(f"Loop core differs from {src} after reload: {bad[:3]}")
    mx = max(m.to_adaln.weight.abs().max().item() for m in core.adaln.values())
    if mx == 0.0:
        raise RuntimeError("Loop core to_adaln is all zero: this checkpoint's loop core was never trained")
    spec = tower.loop_spec
    layers = tower.vision_tower.vision_model.encoder.layers
    adopted = all(layers[i].adaln is core.adaln[str(i)] for i in range(spec["loop_start"], spec["loop_end"]))
    if not adopted or tower.question_conditioner is not core.question_conditioner \
            or tower.loop_step_embedder is not core.loop_step_embedder:
        raise RuntimeError("The vision tower does not use the loop core's modules")
    logger.info("Loop core reloaded: %d tensors from %s, max|to_adaln|=%.3e", len(file_sd), src, mx)
    return len(file_sd)


def load_tokenizer(path: str, **kwargs):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, use_fast=False, **kwargs)


def load_for_eval(base_model: str, adapter: Optional[str] = None, vision_tower: Optional[str] = None, *,
                  dtype: torch.dtype = torch.float16, device: str = "cuda",
                  attn_implementation: Optional[str] = None, with_tokenizer: bool = True):
    """Load a model for evaluation; returns ``(tokenizer | None, model, image_processor)``.

    * ``adapter`` given (LoRA run): config from the adapter dir (with :func:`normalize_model_config`),
      base weights from ``base_model``, then ``non_lora_trainables.bin`` (projector + loop core,
      ``strict=False``; a portable checkpoint has ``non_lora_trainables.safetensors`` instead, read
      the same way when the ``.bin`` is absent), then the LoRA adapter merged into the weights.
    * ``adapter`` is ``None`` (zero-shot): the full model from ``base_model``, which must not
      configure a loop core (a loop core only ships in an adapter directory's sidecar).

    Then the CLIP tower is loaded (retargeted to the configured resolution) and cast to ``dtype``,
    and for loop models the core is reloaded in fp32 and verified (:func:`reload_loop_core`).
    ``vision_tower`` overrides the CLIP path recorded in the checkpoint config, in memory only.
    The projector keeps the dtype ``from_pretrained`` materialised it in (``dtype``); it computes
    in fp32 regardless.
    """
    src_dir = adapter if adapter is not None else base_model
    config = normalize_model_config(LlavaConfig.from_pretrained(src_dir))
    if adapter is None and config.loop_mode != "none":
        raise ValueError(f"{base_model} configures a loop core (loop mode {config.loop_mode}); "
                         f"pass the adapter directory that holds it")
    if vision_tower is not None:
        config.mm_vision_tower = vision_tower
    device_map = None if device == "cpu" else device
    kwargs: Dict[str, Any] = dict(config=config, torch_dtype=dtype, low_cpu_mem_usage=True)
    if device_map is not None:
        kwargs["device_map"] = device_map
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = LlavaLlamaForCausalLM.from_pretrained(base_model, **kwargs)

    raw: Optional[Dict[str, torch.Tensor]] = None
    src = ""
    if adapter is not None:
        path = os.path.join(adapter, NON_LORA_FILE)
        safetensors_path = os.path.join(adapter, NON_LORA_SAFETENSORS_FILE)
        if not os.path.exists(path) and os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            raw = load_file(safetensors_path)
            src = NON_LORA_SAFETENSORS_FILE
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{adapter} has no {NON_LORA_FILE}; not a complete LoRA output directory")
            raw = torch.load(path, map_location="cpu", weights_only=True)
            src = NON_LORA_FILE
        model.load_state_dict(normalize_artifact_keys(raw), strict=False)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    tower = model.get_vision_tower()
    if not tower.is_loaded:
        tower.load_model(device_map=device_map)
    tower.to(device=device, dtype=dtype)
    if getattr(tower, "loop_spec", None) is not None:
        reload_loop_core(model, raw, src)
    model.eval()
    proj = model.get_model().mm_projector
    logger.info("Loaded %s (loop mode %s); projector weight dtype %s", adapter or base_model,
                config.loop_mode, next(proj.parameters()).dtype)
    tokenizer = load_tokenizer(base_model) if with_tokenizer else None
    return tokenizer, model, tower.image_processor
