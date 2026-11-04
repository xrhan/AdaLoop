"""Training entry point shared by every setting (``variant``: sft | adaln | a | b | c).

Build order:

1. ``set_seed(train.seed)`` (before any module is constructed); the resume checkpoint, if any,
   is located.
2. Tokenizer from the base model (slow Llama tokenizer, ``pad = unk``, right padding).
3. Model (:func:`adaloop_llava.model.loading.build_model_for_training`): stage-2 LLaVA base in
   bf16 -> cast to bf16 (rounds the rotary ``inv_freq`` buffer, as in the paper recipe) ->
   LoRA on the language-model projections only -> vision modules for the loop mode (loop core +
   retargeted CLIP; projector trainable in fp32) -> CLIP cast to bf16 -> spec check ->
   non-reentrant gradient checkpointing. With ``train.bf16: false`` the language model stays
   fp32 and the CLIP tower is cast to fp16, as in the paper recipe.
4. Trainable-parameter check (LoRA only on the LM projections; total = LoRA + projector + core).
5. Data: the VisRO training rows of ``data.tasks`` followed by the replay rows, pooled with
   equal row weight (:func:`adaloop_common.data.load_training_set`).
6. ``LLaVATrainer`` (HF Trainer; plain DDP under torchrun), optional resume, final save.

Outputs in ``cfg.output_dir`` (``paths.output_dir``, else ``<paths.output_root>/<run_name>``):
``adapter_model.safetensors``, ``adapter_config.json``, ``non_lora_trainables.bin`` (projector +
loop core), ``config.json`` (including the ``loop_*`` keys), ``resolved_config.yaml``, and
``checkpoint-<step>/`` directories with the same files plus optimizer / scheduler / RNG state.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Dict, List, Optional

import torch

from ..config import Config, dump_config, rank_mean_factor
from adaloop_common.data import load_training_set

from ..data.dataset import TrainingCollator, TrainingDataset
from ..model.loading import NON_LORA_FILE, build_model_for_training, load_tokenizer, training_dtypes
from .trainer import RESOLVED_CONFIG_FILE, GradArrivalCallback, LLaVATrainer, LLaVATrainingArguments

logger = logging.getLogger(__name__)

#: Global batch of the paper runs (8 GPUs x 8 per device x 1 accumulation step).
PAPER_GLOBAL_BATCH = 64

#: Trainable parameters of the released runs (Vicuna-7B, CLIP ViT-L/14, LoRA r=128 on all LM
#: projections, band [2, 8)), checked at startup when the model has exactly these dimensions.
#: Variant C adds the draft feedback (4,201,985 parameters) to the loop core.
PAPER_TRAINABLE = {"sft": 340_795_392, "adaln": 350_151_936, "a": 350_447_872, "b": 350_447_872,
                   "c": 354_649_857}

_LORA_NAME = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.lora_[AB]\.")


# ----------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------

def build_train_dataset(cfg: Config, tokenizer, image_processor) -> TrainingDataset:
    """The VisRO training rows of ``data.tasks`` (``data.visro_root``) followed by the replay rows
    (``data.replay_jsonl``, optional).

    The pool order (VisRO tasks, shards and records in dataset order, then the replay rows in file
    order) fixes which sample the seeded sampler draws at each step.
    """
    if not cfg.data.visro_root:
        raise ValueError("data.visro_root is not set")
    training_set = load_training_set(cfg.data.visro_root, tasks=cfg.data.tasks,
                                     difficulty=cfg.data.difficulty, replay_jsonl=cfg.data.replay_jsonl,
                                     replay_image_root=cfg.data.replay_image_root)
    logger.info("[data] %s", training_set.describe())
    return TrainingDataset(training_set.examples, tokenizer, image_processor,
                           condition_max_length=cfg.data.condition_max_length,
                           sample_total=cfg.data.data_limit or None)


def make_training_arguments(cfg: Config, world_size: int) -> LLaVATrainingArguments:
    t, o = cfg.train, cfg.optim
    ddp_backend = None
    if world_size > 1:
        ddp_backend = "nccl" if torch.cuda.is_available() else "gloo"
    return LLaVATrainingArguments(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        max_steps=t.max_steps,
        num_train_epochs=1,
        per_device_train_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum,
        optim=o.name,
        learning_rate=o.lr,
        mm_projector_lr=o.mm_projector_lr,
        weight_decay=o.weight_decay,
        adam_beta1=o.betas[0],
        adam_beta2=o.betas[1],
        adam_epsilon=o.eps,
        max_grad_norm=o.max_grad_norm,
        lr_scheduler_type=o.lr_scheduler,
        warmup_ratio=o.warmup_ratio,
        bf16=t.bf16,
        tf32=t.tf32,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if t.gradient_checkpointing else None,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        save_safetensors=t.save_safetensors,
        save_only_model=False,                  # optimizer / scheduler / RNG state are needed to resume
        logging_strategy="steps",
        logging_steps=t.logging_steps,
        seed=t.seed,
        dataloader_num_workers=t.dataloader_num_workers,
        dataloader_drop_last=False,
        remove_unused_columns=False,            # keeps condition_input_ids / condition_attention_mask
        group_by_length=False,
        ignore_data_skip=False,                 # a resumed run skips the batches it already trained on
        average_tokens_across_devices=True,     # see train.loss_normalization
        ddp_backend=ddp_backend,
        ddp_find_unused_parameters=t.ddp_find_unused_parameters,
        use_cpu=not torch.cuda.is_available(),  # CPU-only runs (tests); also enables CPU bf16 autocast
        report_to="none",
    )


def check_trainable_parameters(model, cfg: Config) -> Dict[str, int]:
    """Verify the trainable set and return its parameter counts.

    * LoRA wraps exactly the configured projections of every LM decoder layer (never the vision
      tower, the loop core, the projector or ``lm_head``).
    * Every other trainable parameter belongs to the projector or the loop core.
    * With the paper's dimensions, the total equals the released count for the variant.
    """
    base = model.get_base_model()
    inner = base.get_model()
    core = getattr(inner, "loop_core", None)
    core_ids = {id(p) for p in core.parameters()} if core is not None else set()
    proj_ids = {id(p) for p in inner.mm_projector.parameters()}
    counts = {"lora": 0, "projector": 0, "loop_core": 0}
    wrapped = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            m = _LORA_NAME.search(name)
            if m is None or m.group(3) not in cfg.lora.target_modules:
                raise RuntimeError(f"LoRA parameter outside the language-model projections: {name}")
            wrapped.add((int(m.group(1)), m.group(3)))
            counts["lora"] += p.numel()
        elif id(p) in proj_ids:
            counts["projector"] += p.numel()
        elif id(p) in core_ids:
            counts["loop_core"] += p.numel()
        else:
            raise RuntimeError(f"Unexpected trainable parameter: {name}")
    n_layers = base.config.num_hidden_layers
    want = {(i, t) for i in range(n_layers) for t in cfg.lora.target_modules}
    if wrapped != want:
        raise RuntimeError(f"LoRA wraps {len(wrapped)} modules, expected {len(want)} "
                           f"({len(cfg.lora.target_modules)} projections x {n_layers} layers)")
    if core is not None and counts["loop_core"] != sum(p.numel() for p in core.parameters()):
        raise RuntimeError("Part of the loop core is frozen")
    counts["total"] = counts["lora"] + counts["projector"] + counts["loop_core"]
    c, lo, loop = base.config, cfg.lora, cfg.model.loop
    paper_dims = (c.hidden_size == 4096 and c.intermediate_size == 11008 and c.num_hidden_layers == 32
                  and getattr(c, "mm_hidden_size", None) == 1024 and lo.r == 128
                  and sorted(lo.target_modules) == sorted(["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "gate_proj", "up_proj", "down_proj"])
                  and (loop.mode == "none" or (list(loop.band) == [2, 8] and loop.condition_dim == 512
                                               and loop.adaln_hidden_dim == 128
                                               and loop.step_embed_max_steps == 16)))
    if paper_dims and counts["total"] != PAPER_TRAINABLE[cfg.variant]:
        raise RuntimeError(f"Trainable parameters {counts['total']:,} differ from the released "
                           f"{PAPER_TRAINABLE[cfg.variant]:,} for variant {cfg.variant!r}: {counts}")
    return counts


def list_checkpoints(output_dir: str) -> List[str]:
    """``checkpoint-<step>`` directories of ``output_dir``, oldest first."""
    found = []
    for p in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        m = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(p))
        if m and os.path.isdir(p):
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def find_resume_checkpoint(cfg: Config) -> Optional[str]:
    """``train.resume``: ``auto`` -> the latest checkpoint of the output dir (or none);
    ``never`` -> start fresh (refused if the output dir already has checkpoints); otherwise the
    given checkpoint directory."""
    mode = cfg.train.resume
    ckpts = list_checkpoints(cfg.output_dir)
    if mode == "auto":
        return ckpts[-1] if ckpts else None
    if mode == "never":
        if ckpts:
            raise ValueError(f"{cfg.output_dir} already has checkpoints ({os.path.basename(ckpts[-1])}); "
                             "use train.resume=auto to continue, or choose another run_name / output_root")
        return None
    if not os.path.isdir(mode):
        raise FileNotFoundError(f"train.resume={mode!r} is neither 'auto', 'never' nor a checkpoint directory")
    return mode


def check_resume_supported() -> None:
    """Resuming restores the optimizer / scheduler / RNG state with ``torch.load``, which
    transformers 4.57 refuses on torch < 2.6 (CVE-2025-32434). Fail before the model is built."""
    from transformers.utils import import_utils
    try:
        import_utils.check_torch_load_is_safe()
    except ValueError as e:
        raise RuntimeError(f"Resuming needs torch >= 2.6 with this transformers version "
                           f"(found torch {torch.__version__}); upgrade torch or start a new run") from e


def reload_non_lora_trainables(model, checkpoint_dir: str) -> int:
    """Load ``non_lora_trainables.bin`` (projector + loop core) of a checkpoint into the model.

    The HF resume path restores only the LoRA adapter of a PEFT model. The key set must equal
    the model's trainable non-LoRA parameters exactly, so a resumed run never continues with a
    freshly initialised loop core or projector.
    """
    path = os.path.join(checkpoint_dir, NON_LORA_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{checkpoint_dir} has no {NON_LORA_FILE}; cannot resume the projector / loop core")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    res = model.load_state_dict(sd, strict=False)
    expect = {k for k, v in model.named_parameters() if v.requires_grad and "lora_" not in k}
    if res.unexpected_keys or set(sd) != expect:
        raise RuntimeError(f"{path} does not match the model: unexpected={res.unexpected_keys[:3]} "
                           f"missing={sorted(expect - set(sd))[:3]} extra={sorted(set(sd) - expect)[:3]}")
    return len(sd)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------

def main(cfg: Config) -> LLaVATrainer:
    """Train one setting; returns the trainer (its ``model`` is the PEFT-wrapped model)."""
    from transformers import set_seed

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    set_seed(cfg.train.seed)
    t = cfg.train
    global_batch = world_size * t.per_device_batch_size * t.grad_accum
    if global_batch != PAPER_GLOBAL_BATCH:
        logger.warning("Global batch is %d x %d x %d = %d; the paper runs used %d",
                       world_size, t.per_device_batch_size, t.grad_accum, global_batch, PAPER_GLOBAL_BATCH)
    if cfg.model.loop.mode != "none" and t.loss_normalization == "rank_mean":
        logger.info("Loss normalisation 'rank_mean': the loop loss is multiplied by %.6g "
                    "(= rank_mean_scale %g / (world %d x grad_accum %d)); logged loss and grad norm "
                    "are %g x a per-process token mean", rank_mean_factor(cfg, world_size),
                    t.rank_mean_scale, world_size, t.grad_accum, t.rank_mean_scale)

    checkpoint = find_resume_checkpoint(cfg)
    if checkpoint is not None:
        check_resume_supported()

    dtype, vision_dtype = training_dtypes(t.bf16)
    tokenizer = load_tokenizer(cfg.paths.base_model, model_max_length=int(cfg.model.model_max_length),
                               padding_side="right")
    tokenizer.pad_token = tokenizer.unk_token
    model = build_model_for_training(cfg, world_size=world_size, torch_dtype=dtype, vision_dtype=vision_dtype)
    counts = check_trainable_parameters(model, cfg)
    logger.info("Trainable parameters (mode %s): LoRA %s, projector %s, loop core %s, total %s",
                cfg.model.loop.mode, *(f"{counts[k]:,}" for k in ("lora", "projector", "loop_core", "total")))

    tower = model.get_base_model().get_vision_tower()
    dataset = build_train_dataset(cfg, tokenizer, tower.image_processor)
    logger.info("Training rows: %d", len(dataset))

    args = make_training_arguments(cfg, world_size)
    if args.world_size != world_size:
        raise RuntimeError(f"WORLD_SIZE={world_size} but the trainer sees {args.world_size} processes")
    trainer = LLaVATrainer(model=model, args=args, train_dataset=dataset,
                           data_collator=TrainingCollator(tokenizer=tokenizer),
                           processing_class=tokenizer, recipe=cfg)
    if not trainer.model_accepts_loss_kwargs:
        raise RuntimeError("The Trainer does not pass num_items_in_batch to the model; "
                           "the loss normalisation assumes transformers 4.57.x")
    if t.debug_grad_arrival:
        trainer.add_callback(GradArrivalCallback(model))
    if trainer.is_world_process_zero():
        dump_config(cfg, os.path.join(cfg.output_dir, RESOLVED_CONFIG_FILE))

    if checkpoint is not None:
        n = reload_non_lora_trainables(model, checkpoint)
        logger.info("Resuming from %s (%d non-LoRA tensors reloaded)", checkpoint, n)
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()
    trainer.save_state()
    # The same files as every checkpoint (LLaVATrainer._save).
    model.config.use_cache = True
    trainer.save_model(cfg.output_dir)
    trainer.accelerator.wait_for_everyone()
    return trainer
