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
# llava/model/language_model/llava_llama.py: question-conditioned looped vision tower,
# deep-supervised training path (serial and fused readouts), the Variant B (image prefix) and
# Variant C (answer trajectory) paths, per-depth generation, and compatibility with
# transformers 4.57 (cache_position and loss keyword arguments).

"""LLaVA-1.5 on LLaMA: plain path, deep-supervised loop path (Variant A), Variants B and C, and
per-depth generation."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import LOSS_NORMALIZATIONS
from ..constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .llava_arch import LlavaMetaForCausalLM, LlavaMetaModel, check_images
from .loop_utils import TRAJ_DRAFT_TOKENS, TRAJ_SEPARATOR_TOKEN_ID, deep_supervision_weights, loop_config_value


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Loss normalisation of the deep-supervised (loop) path; see set_loss_normalization.
        self.loss_normalization = "rank_mean"
        self.rank_mean_factor = 1.0
        # Unweighted per-iteration losses of the last deep-supervised forward (detached).
        self.last_loop_step_losses: Optional[Tuple[torch.Tensor, ...]] = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def set_loss_normalization(self, mode: str, rank_mean_factor: float = 1.0) -> None:
        """Choose how the loop (deep-supervision) path normalises its loss.

        ``rank_mean`` (the paper recipe): each iteration's loss is the mean over the *local*
        supervised tokens and ``num_items_in_batch`` is ignored; the total is multiplied by
        ``rank_mean_factor`` (``rank_mean_scale / (world_size * grad_accum)``, exactly 1.0 for the
        paper's 8 GPUs x 1 accumulation step). The plain (SFT) path always forwards
        ``num_items_in_batch`` to the LM loss, as in the paper runs.

        ``token_mean``: each iteration's loss is ``sum of local CE / num_items_in_batch``, i.e. the
        same normalisation as the plain path (falls back to the local mean when
        ``num_items_in_batch`` is not provided).
        """
        if mode not in LOSS_NORMALIZATIONS:
            raise ValueError(f"loss normalization must be one of {LOSS_NORMALIZATIONS}, got {mode!r}")
        if not rank_mean_factor > 0:
            raise ValueError(f"rank_mean_factor must be > 0, got {rank_mean_factor}")
        self.loss_normalization = mode
        self.rank_mean_factor = float(rank_mean_factor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        # Question condition and depth for the loop tower (produced by the collator / eval code).
        condition_input_ids: Optional[torch.LongTensor] = None,
        condition_attention_mask: Optional[torch.Tensor] = None,
        loop_steps: Optional[int] = None,
        # transformers >= 4.57 generate() passes cache_position; forwarded unchanged.
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        check_images(images)
        tower = self.get_vision_tower()
        if getattr(tower, "loop_spec", None) is not None and inputs_embeds is None and images is not None:
            return self._forward_looped(
                input_ids, attention_mask, position_ids, past_key_values, labels,
                images, condition_input_ids, condition_attention_mask,
                loop_steps, use_cache, output_attentions, output_hidden_states, return_dict,
                cache_position=cache_position, **kwargs)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        # The plain path forwards **kwargs (including the Trainer's num_items_in_batch).
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

    def _set_loop_condition(self, tower, condition_input_ids, condition_attention_mask, loop_steps):
        """Set the question condition (:meth:`_question_condition`) and the depth on the tower.

        Called at the start of every looped forward and every generate, so no state leaks across
        batches.
        """
        tower.set_active_condition(self._question_condition(condition_input_ids, condition_attention_mask))
        tower.set_active_depth(loop_steps or None)

    def _answer_losses_fused(self, feats, input_ids, position_ids, attention_mask,
                             past_key_values, labels, images, num_items_in_batch=None):
        """Per-iteration answer losses of K readouts computed with a single LM forward.

        The K spliced inputs share text, masks and positions and differ only in the image
        features, so they are concatenated along the batch dimension (iteration-major) and run
        through the LM once. The LM head is applied only at supervised positions. Each
        iteration's loss is the mean over its supervised tokens (or their sum divided by
        ``num_items_in_batch`` when given), i.e. the same token set as the serial path; results
        differ from it only by floating-point associativity.
        """
        K = len(feats)
        embeds, labs, masks, pids = [], [], [], []
        for f in feats:
            (_, pid, am, _, ie, lb) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels,
                images, image_features=f)
            embeds.append(ie); labs.append(lb); masks.append(am); pids.append(pid)
        ie = torch.cat(embeds, dim=0)
        lb = torch.cat(labs, dim=0)
        am = None if masks[0] is None else torch.cat(masks, dim=0)
        pid = None if pids[0] is None else torch.cat(pids, dim=0)
        out = self.get_model()(inputs_embeds=ie, attention_mask=am, position_ids=pid,
                               use_cache=False, return_dict=True)
        h = out.last_hidden_state
        shifted = lb[:, 1:]
        sup = shifted.ne(IGNORE_INDEX)
        if torch.any(sup.sum(dim=1) == 0):
            raise ValueError("Deep supervision needs at least one supervised token in every row")
        tl = F.cross_entropy(self.lm_head(h[:, :-1][sup]).float(),
                             shifted[sup].to(h.device), reduction="none")
        B = input_ids.shape[0]
        counts = sup.sum(dim=1)
        bounds = torch.cumsum(counts, dim=0)
        losses = []
        for i in range(K):
            lo = 0 if i == 0 else int(bounds[i * B - 1].item())
            hi = int(bounds[(i + 1) * B - 1].item())
            if num_items_in_batch is None:
                losses.append(tl[lo:hi].mean())
            else:
                losses.append(tl[lo:hi].sum() / num_items_in_batch)
        return losses

    def _forward_looped(self, input_ids, attention_mask, position_ids, past_key_values, labels,
                        images, condition_input_ids, condition_attention_mask,
                        loop_steps, use_cache, output_attentions, output_hidden_states, return_dict,
                        cache_position=None, **kwargs):
        """Forward of a loop model.

        Training with labels (deep supervision): one tower pass yields the readouts of all K
        iterations; each goes through the projector and its own LM loss, and the losses are
        combined with the deep-supervision weights (``geometric:0.6`` gives ``(1.0,)`` for K=1).
        Otherwise (inference): a single LM forward on the readout of the last iteration.
        Variants B and C (``config.loop_readout_mode``) take their own paths.
        """
        readout = self._loop_readout()
        if readout != "last":
            fwd = self._forward_image_prefix if readout == "image_prefix" else self._forward_trajectory
            return fwd(input_ids, attention_mask, position_ids, past_key_values, labels, images,
                       condition_input_ids, condition_attention_mask, loop_steps, use_cache, output_attentions,
                       output_hidden_states, return_dict, cache_position=cache_position, **kwargs)
        tower = self.get_vision_tower()
        self._set_loop_condition(tower, condition_input_ids, condition_attention_mask, loop_steps)
        if not (self.training and labels is not None):
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
             ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images)
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=return_dict,
                cache_position=cache_position, **kwargs)

        dsw = getattr(self.config, "loop_deep_sup_weight", None)
        if not dsw:
            raise ValueError("Training a loop model requires config.loop_deep_sup_weight (e.g. 'geometric:0.6')")
        nib = kwargs.get("num_items_in_batch") if self.loss_normalization == "token_mean" else None
        if torch.is_tensor(nib):
            nib = nib.to(input_ids.device)
        tower.set_collect_loop_step_features(True)
        try:
            tower(images)                                       # fills the K readouts
            raws = tower.take_loop_step_features()
            proj = self.get_model().mm_projector
            feats = [proj(r) for r in raws]                     # one projector pass per readout
            weights = deep_supervision_weights(dsw, len(feats))
            # Iterations with weight 0 are skipped (the last one is always kept).
            used = [(k, w, f) for k, (w, f) in enumerate(zip(weights, feats))
                    if w != 0.0 or k == len(feats) - 1]
            fused = bool(getattr(self.config, "loop_fused_readout", False)) and len(used) > 1
            if fused:
                raw = self._answer_losses_fused(
                    [f for _, _, f in used], input_ids, position_ids, attention_mask,
                    past_key_values, labels, images, num_items_in_batch=nib)
                step_losses = [L * w for L, (_, w, _) in zip(raw, used)]
                last = CausalLMOutputWithPast(loss=None, logits=None)
            else:
                raw, step_losses, last = [], [], None
                for k, w, f in used:
                    (ii, pid, am, pkv, ie, lb) = self.prepare_inputs_labels_for_multimodal(
                        input_ids, position_ids, attention_mask, past_key_values, labels, images,
                        image_features=f)
                    extra = {} if nib is None else {"num_items_in_batch": nib}
                    # Unlike the plain path, **kwargs is not forwarded here (see set_loss_normalization).
                    out = super().forward(input_ids=ii, attention_mask=am, position_ids=pid,
                                          past_key_values=pkv, inputs_embeds=ie, labels=lb,
                                          use_cache=False, return_dict=True, **extra)
                    raw.append(out.loss)
                    step_losses.append(out.loss * w)
                    last = out
            self.last_loop_step_losses = tuple(x.detach() for x in raw)
            total = step_losses[0]
            for x in step_losses[1:]:
                total = total + x
            if self.loss_normalization == "rank_mean" and self.rank_mean_factor != 1.0:
                total = total * self.rank_mean_factor
            # Set through __setitem__: ModelOutput only exposes keys that were set as items, and the
            # Trainer reads outputs["loss"].
            last["loss"] = total
            return last
        finally:
            tower.set_collect_loop_step_features(False)

    # -- Variants B and C ----------------------------------------------------------------------

    def _loop_readout(self) -> str:
        """``config.loop_readout_mode`` of a loop model (``last`` for Variant A and single pass)."""
        return loop_config_value(self.config, "loop_readout_mode")

    def _pad_id(self) -> int:
        return self.config.pad_token_id if self.config.pad_token_id is not None else 0

    def _trajectory_settings(self) -> Tuple[int, int]:
        """Variant C's draft length and separator token, as recorded in ``config.json``."""
        return (int(getattr(self.config, "loop_draft_max_new_tokens", None) or TRAJ_DRAFT_TOKENS),
                int(getattr(self.config, "loop_traj_sep_token_id", None) or TRAJ_SEPARATOR_TOKEN_ID))

    def _question_condition(self, condition_input_ids, condition_attention_mask) -> torch.Tensor:
        """The pooled question condition ``[B, condition_dim]`` (fp32).

        The question tokens are embedded with the LM's input embeddings and detached (the loop does
        not train the word embeddings), then pooled by the loop core's question conditioner.
        """
        if condition_input_ids is None:
            raise ValueError("Loop models need condition_input_ids (the tokenized question)")
        emb = self.get_model().embed_tokens(condition_input_ids).detach()
        mask = condition_attention_mask if condition_attention_mask is not None \
            else torch.ones(condition_input_ids.shape, dtype=torch.bool, device=condition_input_ids.device)
        return self.get_vision_tower().question_conditioner(emb, mask)

    def _image_prefix_features(self, tower, images) -> torch.Tensor:
        """Variant B: the projected readouts of all rounds, concatenated in round order
        (``[B, K x N, D]``). The condition and the depth must be set on the tower."""
        tower.set_collect_loop_step_features(True)
        try:
            tower(images)
            raws = tower.take_loop_step_features()
        finally:
            tower.set_collect_loop_step_features(False)
        proj = self.get_model().mm_projector
        return torch.cat([proj(r) for r in raws], dim=1)

    def _forward_image_prefix(self, input_ids, attention_mask, position_ids, past_key_values, labels,
                              images, condition_input_ids, condition_attention_mask,
                              loop_steps, use_cache, output_attentions, output_hidden_states, return_dict,
                              cache_position=None, **kwargs):
        """Variant B: one language-model pass with the readouts of all K rounds as the image tokens.
        With labels this is the training loss, normalised like the plain path (``**kwargs``, which
        carries the Trainer's ``num_items_in_batch``, is forwarded)."""
        tower = self.get_vision_tower()
        self._set_loop_condition(tower, condition_input_ids, condition_attention_mask, loop_steps)
        feats = self._image_prefix_features(tower, images)
        (ii, pid, am, pkv, ie, lb) = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, past_key_values, labels, images, image_features=feats)
        return super().forward(input_ids=ii, attention_mask=am, position_ids=pid, past_key_values=pkv,
                               inputs_embeds=ie, labels=lb, use_cache=use_cache,
                               output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                               return_dict=return_dict, cache_position=cache_position, **kwargs)

    @torch.no_grad()
    def _rollout_draft(self, prompt_ids, prompt_mask, images, image_features, max_new_tokens):
        """Greedy draft of ``max_new_tokens`` tokens from a left-padded prompt batch whose
        ``<image>`` is filled with ``image_features`` (one round's detached readout).

        Every step is decoded (a hand-written KV-cache loop, not ``generate``); the returned mask
        keeps the tokens before the first EOS or pad. In training the model stays in training mode
        (LoRA dropout on): gradient checkpointing is switched off for the rollout, since it would
        disable the KV cache, and autocast runs without its weight cache, so that no weight copy
        made here under ``no_grad`` is reused by the training forward.
        """
        side = getattr(self.config, "tokenizer_padding_side", "right")
        gc_layers = [m for m in self.modules() if getattr(m, "gradient_checkpointing", False) is True]
        dev_type = images.device.type
        ac_dtype = torch.get_autocast_dtype(dev_type) if torch.is_autocast_enabled(dev_type) else None
        ac = (torch.autocast(device_type=dev_type, dtype=ac_dtype, enabled=True, cache_enabled=False)
              if ac_dtype is not None else torch.autocast(device_type=dev_type, enabled=False))
        eos = self.config.eos_token_id
        pad = self.config.pad_token_id if self.config.pad_token_id is not None else eos
        try:
            ac.__enter__()
            for m in gc_layers:
                m.gradient_checkpointing = False
            self.config.tokenizer_padding_side = "left"
            (_, pid, am, _, ie, _) = self.prepare_inputs_labels_for_multimodal(
                prompt_ids, None, prompt_mask, None, None, images, image_features=image_features)
            if pid is None:
                pid = am.long().cumsum(-1) - 1
                pid = pid.masked_fill(am == 0, 1)
            out = LlamaForCausalLM.forward(self, inputs_embeds=ie, attention_mask=am, position_ids=pid,
                                           use_cache=True, return_dict=True)
            pkv = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            toks = [nxt]
            cur_am, cur_pos = am, pid[:, -1:] + 1
            one = torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)
            for _ in range(1, int(max_new_tokens)):
                emb = self.get_model().embed_tokens(nxt[:, None])
                cur_am = torch.cat([cur_am, one], dim=1)
                out = LlamaForCausalLM.forward(self, inputs_embeds=emb, attention_mask=cur_am, position_ids=cur_pos,
                                               past_key_values=pkv, use_cache=True, return_dict=True)
                pkv = out.past_key_values
                nxt = out.logits[:, -1].argmax(-1)
                cur_pos = cur_pos + 1
                toks.append(nxt)
            ids = torch.stack(toks, dim=1)
        finally:
            ac.__exit__(None, None, None)
            torch.clear_autocast_cache()
            self.config.tokenizer_padding_side = side
            for m in gc_layers:
                m.gradient_checkpointing = True
        keep = (ids != eos) & (ids != pad)
        return ids, torch.cumprod(keep.long(), dim=1).bool()

    def _trajectory_features(self, tower, images, q, steps, prompt_ids, prompt_mask):
        """Variant C: run the K rounds; after every round but the last, draft an answer from that
        round's (detached) readout and fold the draft into the next round's condition.

        Returns the projected readouts of all rounds (with gradients) and the ``K - 1`` drafts as
        ``(ids, mask)`` pairs.
        """
        core = self.get_model().loop_core
        proj = self.get_model().mm_projector
        h = tower.prelude(images)
        cond = q
        feats, drafts = [], []
        for r in range(steps):
            h = tower.band_step(h, cond, r)
            feat = proj(tower.coda(h, images.dtype))
            feats.append(feat)
            if r == steps - 1:
                break
            draft_ids, draft_mask = self._rollout_draft(prompt_ids, prompt_mask, images, feat.detach(),
                                                        self._trajectory_settings()[0])
            drafts.append((draft_ids, draft_mask))
            cond = core.draft_feedback(q, self.get_model().embed_tokens(draft_ids).detach(), draft_mask)
        return feats, drafts

    @staticmethod
    def _prompt_batch(input_ids, labels, pad_id):
        """The prompt of every row of a right-padded training batch (the tokens before the first
        supervised label), as a left-padded batch with its mask."""
        B = input_ids.shape[0]
        has = labels != IGNORE_INDEX
        first = torch.where(has.any(dim=1), has.float().argmax(dim=1),
                            torch.full((B,), input_ids.shape[1], device=input_ids.device))
        L = int(first.max().item())
        prompt = torch.full((B, L), pad_id, dtype=input_ids.dtype, device=input_ids.device)
        mask = torch.zeros((B, L), dtype=torch.bool, device=input_ids.device)
        for i in range(B):
            n = int(first[i].item())
            prompt[i, L - n:] = input_ids[i, :n]
            mask[i, L - n:] = True
        return prompt, mask

    @staticmethod
    def _trajectory_rows(ids, drafts, i, sep_id):
        """One row of the trajectory layout: the prompt ``head <image> tail`` is split at its single
        ``<image>`` and becomes ``head + sum_r [<image> tail (draft_r sep)?]`` with K image tokens
        and the kept tokens of the K - 1 drafts."""
        img_at = torch.nonzero(ids == IMAGE_TOKEN_INDEX, as_tuple=False)
        if img_at.numel() != 1:
            raise ValueError(f"Row {i} of the prompt has {img_at.numel()} <image> tokens; the trajectory "
                             "layout needs exactly one")
        p = int(img_at[0])
        head, img, tail = ids[:p], ids[p:p + 1], ids[p + 1:]
        parts = [head]
        for r in range(len(drafts) + 1):
            parts += [img, tail]
            if r < len(drafts):
                d, dm = drafts[r]
                parts.append(d[i][dm[i].bool()])
                parts.append(torch.tensor([sep_id], dtype=ids.dtype, device=ids.device))
        return torch.cat(parts, dim=0)

    def _splice_trajectory(self, input_ids, labels, attention_mask, drafts):
        """Variant C training layout: every row becomes ``trajectory prompt + answer``, right-padded,
        with labels on the answer only. Returns ``(input_ids, labels, attention_mask)``."""
        pad_id, dev = self._pad_id(), input_ids.device
        rows, lrows = [], []
        for i in range(input_ids.shape[0]):
            keep = attention_mask[i].bool() if attention_mask is not None \
                else torch.ones_like(input_ids[i], dtype=torch.bool)
            ids, lab = input_ids[i][keep], labels[i][keep]
            ans_at = torch.nonzero(lab != IGNORE_INDEX, as_tuple=False)
            cut = int(ans_at[0]) if ans_at.numel() else ids.shape[0]
            row = torch.cat([self._trajectory_rows(ids[:cut], drafts, i, self._trajectory_settings()[1]),
                             ids[cut:]])
            ans_lab = lab[cut:]
            rows.append(row)
            lrows.append(torch.cat([torch.full((row.shape[0] - ans_lab.shape[0],), IGNORE_INDEX,
                                               dtype=lab.dtype, device=dev), ans_lab]))
        B, L = len(rows), max(r.shape[0] for r in rows)
        out = torch.full((B, L), pad_id, dtype=input_ids.dtype, device=dev)
        out_l = torch.full((B, L), IGNORE_INDEX, dtype=labels.dtype, device=dev)
        out_m = torch.zeros((B, L), dtype=torch.bool, device=dev)
        for i, (r, l) in enumerate(zip(rows, lrows)):
            out[i, :r.shape[0]], out_l[i, :l.shape[0]], out_m[i, :r.shape[0]] = r, l, True
        return out, out_l, out_m

    def _splice_trajectory_prompt(self, prompt_ids, prompt_mask, drafts):
        """Variant C generation layout: the trajectory prompt of every row, left-padded (no answer).
        Returns ``(input_ids, attention_mask)``."""
        pad_id, dev = self._pad_id(), prompt_ids.device
        rows = []
        for i in range(prompt_ids.shape[0]):
            keep = prompt_mask[i].bool() if prompt_mask is not None \
                else torch.ones_like(prompt_ids[i], dtype=torch.bool)
            rows.append(self._trajectory_rows(prompt_ids[i][keep], drafts, i, self._trajectory_settings()[1]))
        B, L = len(rows), max(r.shape[0] for r in rows)
        out = torch.full((B, L), pad_id, dtype=prompt_ids.dtype, device=dev)
        out_m = torch.zeros((B, L), dtype=torch.bool, device=dev)
        for i, r in enumerate(rows):
            out[i, L - r.shape[0]:], out_m[i, L - r.shape[0]:] = r, True
        return out, out_m

    def _forward_trajectory(self, input_ids, attention_mask, position_ids, past_key_values, labels,
                            images, condition_input_ids, condition_attention_mask,
                            loop_steps, use_cache, output_attentions, output_hidden_states, return_dict,
                            cache_position=None, **kwargs):
        """Variant C training forward: K rounds with K - 1 drafts, then one language-model pass over
        the trajectory layout (K image blocks interleaved with the drafts), labels on the answer only.
        The loss is normalised like the plain path (``**kwargs`` is forwarded)."""
        if labels is None:
            raise ValueError("The Variant C forward needs labels (they separate the prompt from the answer); "
                             "use generate() for inference")
        tower = self.get_vision_tower()
        steps = int(loop_steps or tower.loop_spec["loop_steps"])
        q = self._question_condition(condition_input_ids, condition_attention_mask)
        prompt_ids, prompt_mask = self._prompt_batch(input_ids, labels, self._pad_id())
        feats, drafts = self._trajectory_features(tower, images, q, steps, prompt_ids, prompt_mask)
        t_ids, t_lab, t_am = self._splice_trajectory(input_ids, labels, attention_mask, drafts)
        flat = [feats[k][b] for b in range(t_ids.shape[0]) for k in range(steps)]    # row-major
        (ii, pid, am, pkv, ie, lb) = self.prepare_inputs_labels_for_multimodal(
            t_ids, None, t_am, past_key_values, t_lab, images, image_features=flat)
        return super().forward(input_ids=ii, attention_mask=am, position_ids=pid, past_key_values=pkv,
                               inputs_embeds=ie, labels=lb, use_cache=use_cache,
                               output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                               return_dict=return_dict, cache_position=cache_position, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate from a prompt containing ``IMAGE_TOKEN_INDEX``.

        Loop models also take ``condition_input_ids`` / ``condition_attention_mask`` (the
        tokenized question) and ``loop_steps`` (evaluation depth; defaults to the trained K).
        At depth ``d``, Variant B reads the readouts of rounds 1..d as its image prefix, and
        Variant C drafts after rounds 1..d-1 and reads the trajectory of d image blocks and d - 1
        drafts.
        """
        check_images(images)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        tower = self.get_vision_tower()
        loop_keys = ("condition_input_ids", "condition_attention_mask", "loop_steps")
        readout = "last"
        if getattr(tower, "loop_spec", None) is not None:
            cond_ids, cond_mask = kwargs.pop("condition_input_ids", None), kwargs.pop("condition_attention_mask", None)
            steps = kwargs.pop("loop_steps", None)
            self._set_loop_condition(tower, cond_ids, cond_mask, steps)
            readout = self._loop_readout()
        elif any(kwargs.get(k) is not None for k in loop_keys):
            raise ValueError("condition_input_ids / loop_steps were given, but this model has no loop tower")
        else:
            for k in loop_keys:
                kwargs.pop(k, None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            feats = None
            if readout == "image_prefix":
                feats = self._image_prefix_features(tower, images)
            elif readout == "answer_trajectory":
                depth = tower.active_depth
                q = self._question_condition(cond_ids, cond_mask)
                rounds, drafts = self._trajectory_features(tower, images, q, depth, inputs, attention_mask)
                inputs, attention_mask = self._splice_trajectory_prompt(inputs, attention_mask, drafts)
                feats = [rounds[k][b] for b in range(inputs.shape[0]) for k in range(depth)]   # row-major
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_features=feats,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


AutoConfig.register("llava_llama", LlavaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM, exist_ok=True)
