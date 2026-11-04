"""Public PaliGemma2 AdaLoop model assembly.

The class in this module owns the stable top-level checkpoint scopes ``img``,
``llm``, and ``question_conditioner``.  It supports only the models of the paper:
plain SFT, single-pass AdaLN, and AdaLoop Variants A/B/C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

import flax.linen as nn
import jax.numpy as jnp

from ..constants import (
    GEMMA_VOCAB_SIZE,
    GEMMA_WIDTH,
    MULTI_IMAGE_VARIANTS,
    PAPER_ADALN_HIDDEN_DIM,
    PAPER_CONDITION_DIM,
    PAPER_CONDITION_HEADS,
    PAPER_DRAFT_MAX_TOKENS,
    PAPER_GEOMETRIC_RATIO,
    PAPER_LORA_ALPHA,
    PAPER_LORA_RANK,
    PAPER_LORA_TARGETS,
    PAPER_MAX_SEQUENCE_LENGTH,
    PAPER_QUESTION_MAX_TOKENS,
    PAPER_TRAIN_DEPTH,
    RECURRENT_VARIANTS,
    VARIANTS,
)
from .conditioning import QuestionConditioner
from .gemma_lora import Gemma2LoRA
from .losses import causal_cross_entropy_per_example, combine_deep_supervision
from .readouts import (
    make_attn_mask,
    pack_accumulated_images,
    pack_current_image,
    select_text_slots,
)
from .trajectory import (
    gather_last_slot,
    gather_text_slots,
    greedy_reprefill_drafts,
    pack_trajectory,
)
from .vision import LoopedSigLIPVision

if TYPE_CHECKING:
    from ..config import PaliGemmaAdaLoopConfig


@dataclass(frozen=True)
class PaliGemmaTokenIds:
    """Tokenizer-derived IDs required by the Variant-C rollout."""

    eos: int
    separator: tuple[int, ...]
    suppressed_draft: tuple[int, ...]

    def __post_init__(self):
        if self.eos < 0:
            raise ValueError("eos must be non-negative")
        if not self.separator:
            raise ValueError("newline separator must encode to at least one token")
        if self.eos in self.suppressed_draft:
            raise ValueError("EOS must remain available as a draft stop event")


class AdaLoopOutput(NamedTuple):
    loss: Any | None
    logits: Any
    depth_logits: tuple[Any, ...]
    depth_losses: tuple[Any, ...]
    drafts: Any | None
    draft_mask: Any | None
    auxiliary: Mapping[str, Any]


class GenerationPrefill(NamedTuple):
    next_token_logits: Any
    drafts: Any | None
    draft_mask: Any | None


class PaliGemmaAdaLoop(nn.Module):
    """PaliGemma2 3B Mix-448 with paper AdaLoop execution."""

    variant: str = "a"
    train_depth: int = PAPER_TRAIN_DEPTH
    deep_supervision: bool = True
    geometric_ratio: float = PAPER_GEOMETRIC_RATIO
    lora_rank: int = PAPER_LORA_RANK
    lora_alpha: float = PAPER_LORA_ALPHA
    lora_targets: tuple[str, ...] = PAPER_LORA_TARGETS
    vocab_size: int = GEMMA_VOCAB_SIZE
    condition_dim: int = PAPER_CONDITION_DIM
    condition_heads: int = PAPER_CONDITION_HEADS
    condition_max_tokens: int = PAPER_QUESTION_MAX_TOKENS
    adaln_hidden_dim: int = PAPER_ADALN_HIDDEN_DIM
    max_sequence_length: int = PAPER_MAX_SEQUENCE_LENGTH
    draft_max_tokens: int = PAPER_DRAFT_MAX_TOKENS
    eos_token_id: int = -1
    separator_token_ids: tuple[int, ...] = ()
    suppressed_draft_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0
    vision_remat_policy: str = "nothing_saveable"
    language_remat_policy: str = "nothing_saveable"

    def setup(self):
        variant = self.variant
        if variant not in VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(VARIANTS)}, got {variant!r}"
            )
        if self.train_depth < 1:
            raise ValueError("train_depth must be positive")
        if variant not in RECURRENT_VARIANTS and self.train_depth != 1:
            raise ValueError("SFT and single-pass AdaLN require train_depth=1")
        if variant in MULTI_IMAGE_VARIANTS and not self.deep_supervision:
            raise ValueError("paper Variants B/C use fixed-depth deep supervision")
        if variant == "c":
            PaliGemmaTokenIds(
                eos=self.eos_token_id,
                separator=self.separator_token_ids,
                suppressed_draft=self.suppressed_draft_token_ids,
            )
            if self.draft_max_tokens != PAPER_DRAFT_MAX_TOKENS:
                raise ValueError("the released Variant C uses draft_max_tokens=24")

        self._llm = Gemma2LoRA(
            vocab_size=self.vocab_size,
            remat_policy=self.language_remat_policy,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_targets=self.lora_targets,
            name="llm",
        )
        self._vision = LoopedSigLIPVision(
            num_classes=GEMMA_WIDTH,
            loop_steps=self.train_depth,
            use_adaln=variant != "sft",
            adaln_hidden_dim=self.adaln_hidden_dim,
            remat_policy=self.vision_remat_policy,
            name="img",
        )
        if variant != "sft":
            self._conditioner = QuestionConditioner(
                max_tokens=self.condition_max_tokens,
                cond_dim=self.condition_dim,
                num_heads=self.condition_heads,
                detach=True,
                name="question_conditioner",
            )

    def question_condition(
        self,
        condition_input_ids,
        condition_attention_mask,
        *,
        train: bool = False,
    ):
        """Pool the dedicated bare/no-BOS question sequence."""

        if self.variant == "sft":
            if condition_input_ids is not None or condition_attention_mask is not None:
                raise ValueError("plain SFT has no question-conditioning branch")
            return None, {}
        if condition_input_ids is None or condition_attention_mask is None:
            raise ValueError(
                "AdaLN/A/B/C require condition_input_ids and "
                "condition_attention_mask from the bare no-BOS tokenizer path"
            )
        return self._conditioner(
            self._llm,
            condition_input_ids,
            condition_attention_mask,
            train=train,
        )

    def encode_vision(
        self,
        image,
        condition,
        *,
        steps: int,
        train: bool = False,
        return_all_steps: bool = False,
    ):
        return self._vision(
            image,
            condition,
            train=train,
            n_steps=steps,
            return_all_steps=return_all_steps,
        )

    def _run_language(self, embeddings, input_mask, mask_ar, *, train: bool):
        """Return Gemma pre-logits (final-norm states) for a packed sequence."""

        attention = make_attn_mask(input_mask.astype(bool), mask_ar)
        return self._llm.compute_pre_logits(embeddings, mask=attention, train=train)

    def _vocabulary_logits(self, pre_logits, *, train: bool):
        # Like Big Vision PaliGemma and the paper's training runs, the objective
        # projects only text-slot pre-logits with the plain tied embedding.
        # Gemma's final-logit softcap belongs to its full forward, not the CE.
        return self._llm.compute_logits(pre_logits, train=train)

    def _separator_embeddings(self, batch_size: int, *, train: bool):
        ids = jnp.asarray(self.separator_token_ids, jnp.int32)[None]
        ids = jnp.broadcast_to(ids, (batch_size, ids.shape[1]))
        return self._llm.embed_tokens(ids, train=train)

    def _draft_embeddings(self, drafts, *, train: bool):
        # Embedder indexing accepts arbitrary leading token dimensions.
        return self._llm.embed_tokens(drafts, train=train)

    def _variant_c_drafts(
        self,
        image_steps,
        text_embeddings,
        input_mask,
        mask_ar,
        *,
        train: bool,
    ):
        separator = self._separator_embeddings(text_embeddings.shape[0], train=train)

        def embed(ids):
            return self._draft_embeddings(ids, train=train)

        def next_token_logits(trajectory):
            # Like prefill_cache, project only each row's last active slot:
            # logits for every packed slot would take 2.5 GiB per example and
            # draft token at the paper's training depth.
            pre_logits = self._run_language(
                trajectory.embeddings,
                trajectory.input_mask,
                trajectory.mask_ar,
                train=train,
            )
            return self._vocabulary_logits(
                gather_last_slot(pre_logits, trajectory), train=train
            )

        return greedy_reprefill_drafts(
            image_readouts=image_steps,
            text_embeddings=text_embeddings,
            text_input_mask=input_mask,
            text_mask_ar=mask_ar,
            separator_embeddings=separator,
            embed_tokens=embed,
            logits_for_trajectory=next_token_logits,
            eos_token_id=self.eos_token_id,
            suppressed_token_ids=self.suppressed_draft_token_ids,
            draft_max_tokens=self.draft_max_tokens,
            max_length=self.max_sequence_length,
        )

    def variant_c_sources(
        self,
        image,
        text,
        mask_ar,
        *,
        condition_input_ids,
        condition_attention_mask,
        steps: int,
    ):
        """Encode the question-only vision snapshots and static C sources."""

        if self.variant != "c":
            raise ValueError("variant_c_sources is available only for Variant C")
        condition, _ = self.question_condition(
            condition_input_ids, condition_attention_mask, train=False
        )
        _, auxiliary = self.encode_vision(
            image,
            condition,
            steps=int(steps),
            train=False,
            return_all_steps=True,
        )
        image_steps = tuple(auxiliary["zimg_steps"])
        return (
            image_steps,
            self._llm.embed_tokens(text, train=False),
            self._separator_embeddings(text.shape[0], train=False),
        )

    def variant_c_prompt(
        self,
        image_steps,
        text_embeddings,
        input_mask,
        mask_ar,
        drafts,
        draft_mask,
        separator_embeddings,
        *,
        depth: int,
    ):
        """Pack a round prompt ending at ``[I_depth | P]``."""

        return pack_trajectory(
            image_steps,
            text_embeddings,
            input_mask,
            mask_ar,
            self._draft_embeddings(drafts, train=False),
            draft_mask,
            separator_embeddings,
            depth=int(depth),
            terminal="draft",
            max_length=self.max_sequence_length,
        )

    def prefill_cache(self, embeddings, input_mask, mask_ar, *, cache_size: int):
        """Prefill Gemma's decode cache and return the next-token logits.

        Call through ``model.apply(..., mutable=("cache",))`` and continue with
        :meth:`extend_cache` on the returned cache collection.
        """

        return self._llm.prefill_cache(
            embeddings,
            input_mask,
            make_attn_mask(input_mask, mask_ar),
            cache_size=int(cache_size),
        )

    def extend_cache(self, token_ids):
        """Append one token per row to a cache created by :meth:`prefill_cache`."""

        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("token_ids must be [B] or [B,1]")
        return self._llm.extend_cache(
            self._llm.embed_tokens(token_ids, train=False)
        )

    def fixed_depth_logits(
        self,
        image,
        text,
        mask_ar,
        *,
        input_mask=None,
        condition_input_ids=None,
        condition_attention_mask=None,
        n_steps: int | None = None,
        drafts=None,
        draft_mask=None,
        return_all_depths: bool = True,
        train: bool = False,
    ):
        """Return logits at every requested recurrent depth.

        Variant C generates detached intermediate drafts when they are not
        supplied. Training uses the fixed-depth geometric deep-supervision
        objective from the paper.
        """

        steps = int(self.train_depth if n_steps is None else n_steps)
        if steps < 1:
            raise ValueError("n_steps must be positive")
        if input_mask is None:
            input_mask = text != self.pad_token_id
        if input_mask.shape != text.shape or mask_ar.shape != text.shape:
            raise ValueError("text, input_mask, and mask_ar must align")
        condition, condition_aux = self.question_condition(
            condition_input_ids,
            condition_attention_mask,
            train=train,
        )
        variant = self.variant
        needs_all = (
            variant in MULTI_IMAGE_VARIANTS
            or (variant == "a" and return_all_depths)
        )
        _, vision_aux = self.encode_vision(
            image,
            condition,
            steps=steps,
            train=train,
            return_all_steps=needs_all,
        )
        if needs_all:
            image_steps = tuple(vision_aux["zimg_steps"])
        else:
            image_steps = (vision_aux["logits"],)
        text_embeddings = self._llm.embed_tokens(text, train=train)

        depth_logits: list[Any] = []
        if variant not in RECURRENT_VARIANTS:
            packed = pack_current_image(
                image_steps[-1], text_embeddings, input_mask, mask_ar
            )
            pre_logits = self._run_language(
                packed.embeddings, packed.input_mask, packed.mask_ar, train=train
            )
            depth_logits.append(
                self._vocabulary_logits(
                    select_text_slots(pre_logits, packed, text.shape[1]), train=train
                )
            )
        elif variant == "a":
            for image_readout in image_steps:
                packed = pack_current_image(
                    image_readout, text_embeddings, input_mask, mask_ar
                )
                pre_logits = self._run_language(
                    packed.embeddings,
                    packed.input_mask,
                    packed.mask_ar,
                    train=train,
                )
                depth_logits.append(
                    self._vocabulary_logits(
                        select_text_slots(pre_logits, packed, text.shape[1]),
                        train=train,
                    )
                )
        elif variant == "b":
            for depth in range(1, steps + 1):
                packed = pack_accumulated_images(
                    image_steps,
                    text_embeddings,
                    input_mask,
                    mask_ar,
                    depth=depth,
                    max_length=self.max_sequence_length,
                )
                pre_logits = self._run_language(
                    packed.embeddings,
                    packed.input_mask,
                    packed.mask_ar,
                    train=train,
                )
                depth_logits.append(
                    self._vocabulary_logits(
                        select_text_slots(pre_logits, packed, text.shape[1]),
                        train=train,
                    )
                )
        else:
            if drafts is None or draft_mask is None:
                drafts, draft_mask = self._variant_c_drafts(
                    image_steps,
                    text_embeddings,
                    input_mask,
                    mask_ar,
                    train=train,
                )
            expected = (text.shape[0], steps - 1, self.draft_max_tokens)
            if drafts.shape != expected or draft_mask.shape != expected:
                raise ValueError(
                    f"Variant-C drafts/mask must have shape {expected}, got "
                    f"{drafts.shape}/{draft_mask.shape}"
                )
            embedded_drafts = self._draft_embeddings(drafts, train=train)
            separator = self._separator_embeddings(text.shape[0], train=train)
            for depth in range(1, steps + 1):
                trajectory = pack_trajectory(
                    image_steps,
                    text_embeddings,
                    input_mask,
                    mask_ar,
                    embedded_drafts,
                    draft_mask,
                    separator,
                    depth=depth,
                    terminal="answer",
                    max_length=self.max_sequence_length,
                )
                pre_logits = self._run_language(
                    trajectory.embeddings,
                    trajectory.input_mask,
                    trajectory.mask_ar,
                    train=train,
                )
                depth_logits.append(
                    self._vocabulary_logits(
                        gather_text_slots(pre_logits, trajectory), train=train
                    )
                )

        auxiliary = {
            "condition": condition,
            "condition_aux": condition_aux,
            "image_steps": image_steps,
            "drafts": drafts,
            "draft_mask": draft_mask,
        }
        return tuple(depth_logits), auxiliary

    def prefill_generation(
        self,
        image,
        prompt_ids,
        prompt_mask_ar,
        *,
        max_new_tokens: int,
        input_mask=None,
        condition_input_ids=None,
        condition_attention_mask=None,
        n_steps: int | None = None,
        drafts=None,
        draft_mask=None,
    ) -> GenerationPrefill:
        """Build the selected readout and initialize Gemma's decode cache.

        Call through ``model.apply(..., method=model.prefill_generation,
        mutable=("cache",))``.  Subsequent greedy tokens are passed to
        :meth:`extend_cache` with that returned cache collection.
        """

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        steps = int(self.train_depth if n_steps is None else n_steps)
        if input_mask is None:
            input_mask = prompt_ids != self.pad_token_id
        condition, _ = self.question_condition(
            condition_input_ids, condition_attention_mask, train=False
        )
        variant = self.variant
        needs_all = variant in MULTI_IMAGE_VARIANTS
        _, vision_aux = self.encode_vision(
            image,
            condition,
            steps=steps,
            train=False,
            return_all_steps=needs_all,
        )
        image_steps = (
            tuple(vision_aux["zimg_steps"])
            if needs_all
            else (vision_aux["logits"],)
        )
        text_embeddings = self._llm.embed_tokens(prompt_ids, train=False)
        if not needs_all:
            packed = pack_current_image(
                image_steps[-1], text_embeddings, input_mask, prompt_mask_ar
            )
            embeddings, valid, ar = (
                packed.embeddings,
                packed.input_mask,
                packed.mask_ar,
            )
        elif variant == "b":
            packed = pack_accumulated_images(
                image_steps,
                text_embeddings,
                input_mask,
                prompt_mask_ar,
                depth=steps,
                max_length=self.max_sequence_length,
            )
            # As in the paper runs, the capacity also covers the decoded answer.
            width = int(packed.embeddings.shape[1])
            if width + int(max_new_tokens) > self.max_sequence_length:
                raise ValueError(
                    "Variant-B context plus decoded answer exceeds "
                    f"max_length={self.max_sequence_length}: {width} prompt "
                    f"slots + {int(max_new_tokens)} new tokens"
                )
            embeddings, valid, ar = (
                packed.embeddings,
                packed.input_mask,
                packed.mask_ar,
            )
        else:
            if drafts is None or draft_mask is None:
                drafts, draft_mask = self._variant_c_drafts(
                    image_steps,
                    text_embeddings,
                    input_mask,
                    prompt_mask_ar,
                    train=False,
                )
            trajectory = pack_trajectory(
                image_steps,
                text_embeddings,
                input_mask,
                prompt_mask_ar,
                self._draft_embeddings(drafts, train=False),
                draft_mask,
                self._separator_embeddings(prompt_ids.shape[0], train=False),
                depth=steps,
                terminal="draft",
                max_length=self.max_sequence_length,
            )
            # As in the paper runs, the capacity also covers the decoded answer.
            width = int(trajectory.embeddings.shape[1])
            if width + int(max_new_tokens) > self.max_sequence_length:
                raise ValueError(
                    "Variant-C trajectory plus decoded answer exceeds "
                    f"max_length={self.max_sequence_length}: {width} prompt "
                    f"slots + {int(max_new_tokens)} new tokens"
                )
            embeddings, valid, ar = (
                trajectory.embeddings,
                trajectory.input_mask,
                trajectory.mask_ar,
            )
        next_logits = self.prefill_cache(
            embeddings,
            valid,
            ar,
            cache_size=embeddings.shape[1] + int(max_new_tokens),
        )
        return GenerationPrefill(next_logits, drafts, draft_mask)

    def __call__(
        self,
        image,
        text,
        mask_ar,
        loss_targets=None,
        loss_mask=None,
        *,
        input_mask=None,
        condition_input_ids=None,
        condition_attention_mask=None,
        n_steps: int | None = None,
        drafts=None,
        draft_mask=None,
        train: bool = False,
    ) -> AdaLoopOutput:
        depth_logits, auxiliary = self.fixed_depth_logits(
            image,
            text,
            mask_ar,
            input_mask=input_mask,
            condition_input_ids=condition_input_ids,
            condition_attention_mask=condition_attention_mask,
            n_steps=n_steps,
            drafts=drafts,
            draft_mask=draft_mask,
            return_all_depths=self.deep_supervision,
            train=train,
        )
        losses: tuple[Any, ...] = ()
        total_loss = None
        if (loss_targets is None) != (loss_mask is None):
            raise ValueError("loss_targets and loss_mask must be provided together")
        if loss_targets is not None:
            losses = tuple(
                causal_cross_entropy_per_example(logits, loss_targets, loss_mask)
                for logits in depth_logits
            )
            if self.deep_supervision and len(losses) > 1:
                total_loss, weights = combine_deep_supervision(
                    losses, self.geometric_ratio
                )
                auxiliary = {**auxiliary, "deep_supervision_weights": weights}
            else:
                total_loss = losses[-1]
        return AdaLoopOutput(
            loss=total_loss,
            logits=depth_logits[-1],
            depth_logits=depth_logits,
            depth_losses=losses,
            drafts=auxiliary.get("drafts"),
            draft_mask=auxiliary.get("draft_mask"),
            auxiliary=auxiliary,
        )


def build_model_from_config(
    config: PaliGemmaAdaLoopConfig | Mapping[str, Any],
    *,
    token_ids: PaliGemmaTokenIds | None = None,
) -> PaliGemmaAdaLoop:
    """Default construction hook used by checkpoint, trainer, and inference.

    ``config`` is a validated recipe or a mapping in its ``to_dict`` layout;
    fields a mapping omits take the paper values.  Loading the licensed base
    NPZ and applying a compact trainable overlay are checkpoint-layer
    responsibilities.  This hook deterministically rebuilds the Flax topology
    those parameter trees target.
    """

    raw = config if isinstance(config, Mapping) else config.to_dict()
    variant = raw["variant"]
    if variant == "c" and token_ids is None:
        raise ValueError(
            "Variant C construction requires tokenizer-derived PaliGemmaTokenIds"
        )
    lora = raw.get("lora", {})
    conditioner = raw.get("conditioner", {})
    readout = raw.get("readout", {})
    geometric_ratio = raw.get("training", {}).get("geometric_ratio")
    recurrent = variant in RECURRENT_VARIANTS
    return PaliGemmaAdaLoop(
        variant=variant,
        train_depth=int(raw.get("train_depth", PAPER_TRAIN_DEPTH if recurrent else 1)),
        deep_supervision=recurrent,
        geometric_ratio=(
            PAPER_GEOMETRIC_RATIO if geometric_ratio is None else float(geometric_ratio)
        ),
        lora_rank=int(lora.get("rank", PAPER_LORA_RANK)),
        lora_alpha=float(lora.get("alpha", PAPER_LORA_ALPHA)),
        lora_targets=tuple(lora.get("target_modules", PAPER_LORA_TARGETS)),
        vocab_size=int(raw.get("backbone", {}).get("vocab_size", GEMMA_VOCAB_SIZE)),
        condition_dim=int(conditioner.get("hidden_size", PAPER_CONDITION_DIM)),
        condition_heads=int(conditioner.get("attention_heads", PAPER_CONDITION_HEADS)),
        condition_max_tokens=int(
            conditioner.get("max_tokens", PAPER_QUESTION_MAX_TOKENS)
        ),
        adaln_hidden_dim=int(
            raw.get("adaln", {}).get("hidden_size", PAPER_ADALN_HIDDEN_DIM)
        ),
        max_sequence_length=int(readout.get("max_length", PAPER_MAX_SEQUENCE_LENGTH)),
        draft_max_tokens=int(readout.get("draft_max_tokens", PAPER_DRAFT_MAX_TOKENS)),
        eos_token_id=-1 if token_ids is None else token_ids.eos,
        separator_token_ids=() if token_ids is None else token_ids.separator,
        suppressed_draft_token_ids=(
            () if token_ids is None else token_ids.suppressed_draft
        ),
    )


__all__ = [
    "AdaLoopOutput",
    "GenerationPrefill",
    "PaliGemmaAdaLoop",
    "PaliGemmaTokenIds",
    "build_model_from_config",
]
