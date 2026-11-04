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
# llava/train/llava_trainer.py: separate learning rate for the projector, every checkpoint carries
# the non-LoRA trainables (projector + loop core) and the model config, per-iteration loop losses
# are logged, and the length-grouped samplers were removed (the default random sampler is used).

"""HF ``Trainer`` subclass used for every setting."""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names

from ..config import Config, dump_config
from ..model.loading import NON_LORA_FILE, non_lora_trainable_state_dict

logger = logging.getLogger(__name__)

RESOLVED_CONFIG_FILE = "resolved_config.yaml"


@dataclass
class LLaVATrainingArguments(TrainingArguments):
    """``TrainingArguments`` plus the projector learning rate.

    ``remove_unused_columns`` defaults to False: the collator's ``condition_input_ids`` /
    ``condition_attention_mask`` are not named in the model's signature and would otherwise be
    dropped before they reach the loop tower.
    """
    remove_unused_columns: bool = field(default=False)
    mm_projector_lr: Optional[float] = field(
        default=None, metadata={"help": "Learning rate of the mm_projector parameters (others use learning_rate)."})


def _unwrap_peft(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


class LLaVATrainer(Trainer):
    """``Trainer`` with

    * two learning rates (``mm_projector_lr`` for the projector, ``learning_rate`` for LoRA and
      the loop core), see :meth:`create_optimizer`;
    * checkpoints that are complete for resuming and evaluation: next to the LoRA adapter, every
      save writes ``non_lora_trainables.bin`` (projector + loop core), ``config.json`` and, when
      a ``recipe`` is given, ``resolved_config.yaml``;
    * logging of the unweighted per-iteration losses of the loop models (``loop_loss_<k>``).
    """

    def __init__(self, *args, recipe: Optional[Config] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recipe = recipe
        self._loop_losses: List[torch.Tensor] = []
        self._sanitize_generation_config()

    # -- generation config ------------------------------------------------------------------

    def _sanitize_generation_config(self) -> None:
        """Reset sampling-only fields of a greedy generation config.

        The Vicuna-7B-v1.5 ``generation_config.json`` sets ``temperature=0.9`` and ``top_p=0.6``
        with ``do_sample=False``; transformers 4.57 refuses to save such a config. With
        ``do_sample=False`` these fields are ignored, so resetting them to their defaults does not
        change generation.
        """
        gc = getattr(self.model, "generation_config", None)
        if gc is None or getattr(gc, "do_sample", False):
            return
        fixed = []
        for name, default in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 50), ("typical_p", 1.0)):
            cur = getattr(gc, name, None)
            if cur is not None and cur != default:
                setattr(gc, name, default)
                fixed.append(f"{name} {cur}->{default}")
        if fixed:
            logger.info("generation_config has do_sample=False; reset sampling-only fields: %s", ", ".join(fixed))

    # -- optimiser --------------------------------------------------------------------------

    def create_optimizer(self):
        """AdamW over the trainable parameters, in up to four groups: (decay, no decay) x
        (projector, everything else). Projector parameters (names containing ``mm_projector``)
        use ``mm_projector_lr`` when it is set; LoRA and the loop core use ``learning_rate``.
        LayerNorm weights and biases get no weight decay."""
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = {name for name in decay_parameters if "bias" not in name}
            named = list(opt_model.named_parameters())
            if self.args.mm_projector_lr is not None:
                projector_parameters = {name for name, _ in named if "mm_projector" in name}
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in named
                                   if (n in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in named
                                   if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in named
                                   if (n in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in named
                                   if (n not in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in named if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in named if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    # -- saving -----------------------------------------------------------------------------

    # Note: _save_checkpoint is deliberately not overridden. Everything extra a LoRA checkpoint
    # needs is written in _save, so the parent's checkpoint logic (optimizer, scheduler, RNG
    # states, trainer_state.json, rotation) runs unchanged and resuming works.
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        ret = super()._save(output_dir, state_dict)
        out = output_dir if output_dir is not None else self.args.output_dir
        # The parent only saves the adapter of a PEFT model. Without the projector and the loop
        # core, resuming or evaluating this checkpoint would silently use their initial values.
        sd = non_lora_trainable_state_dict(self.model)
        torch.save(sd, os.path.join(out, NON_LORA_FILE))
        self.model.config.save_pretrained(out)
        if self.recipe is not None:
            dump_config(self.recipe, os.path.join(out, RESOLVED_CONFIG_FILE))
        logger.info("%s: %d tensors -> %s", NON_LORA_FILE, len(sd), out)
        return ret

    # -- per-iteration loss logging -----------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                   num_items_in_batch=num_items_in_batch)
        base = _unwrap_peft(self.model)
        steps = getattr(base, "last_loop_step_losses", None)
        if steps is not None:
            if base.training:
                self._loop_losses.append(torch.stack([s.float() for s in steps]))
            base.last_loop_step_losses = None
        return out

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Adds ``loop_loss_<k>`` (k = 1..K) to training logs: the unweighted loss of iteration k
        on this process, averaged over the micro-batches since the last log, as the model computed
        it (``rank_mean``: mean over the local supervised tokens; ``token_mean``: local sum divided
        by the global token count)."""
        if "loss" in logs and self._loop_losses:
            shapes = {t.shape for t in self._loop_losses}
            if len(shapes) == 1:
                mean = torch.stack(self._loop_losses).mean(dim=0).tolist()
                for k, v in enumerate(mean, start=1):
                    logs[f"loop_loss_{k}"] = round(float(v), 4)
            self._loop_losses = []
        super().log(logs, start_time)


class GradArrivalCallback(TrainerCallback):
    """Debug check (``train.debug_grad_arrival``): in the first two optimizer steps, every group
    of trainable parameters (LoRA, projector, loop core) must receive at least one gradient on
    some process; otherwise training stops. Catches a loop core or projector that is silently cut
    out of the graph."""

    def __init__(self, model):
        self.model = model

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step >= 2:
            return
        core = getattr(_unwrap_peft(self.model).get_model(), "loop_core", None)
        groups = {
            "lora": [p for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad],
            "mm_projector": [p for n, p in self.model.named_parameters() if "mm_projector" in n and p.requires_grad],
            # Counted by object: the tower borrows the core's modules under other names.
            "loop_core": [p for p in core.parameters() if p.requires_grad] if core is not None else [],
        }
        rows, missing = [], []
        for tag, ps in groups.items():
            if not ps:
                continue
            got = sum(1 for p in ps if p.grad is not None and p.grad.numel() > 0)
            zero = sum(1 for p in ps if p.grad is not None and p.grad.numel() > 0 and float(p.grad.abs().max()) == 0.0)
            got_t = torch.tensor([got], device=args.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(got_t, op=torch.distributed.ReduceOp.MAX)
            rows.append(f"{tag}: {len(ps)} tensors, {got} with grad, {zero} zero")
            if int(got_t.item()) == 0:
                missing.append(tag)
        logger.info("[grad-arrival] step %d: %s", state.global_step, " | ".join(rows))
        if missing and state.global_step >= 1:
            raise RuntimeError(f"Parameter groups {missing} received no gradient at step {state.global_step}")
