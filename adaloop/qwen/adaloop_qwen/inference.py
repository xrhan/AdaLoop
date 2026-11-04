"""High-level fixed-depth and answer-stability inference for Qwen AdaLoop."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from .checkpoint import LoadedQwenCheckpoint, QwenCheckpointLoader
from .model.generation import (
    REPETITION_PENALTY,
    P1Output,
    generate_p1_stable,
    generate_variant_c_reprefill,
    greedy_decode,
    suppressed_special_token_ids,
)
from .model.prompting import tokenize_question_conditions, tokenize_variant_c_fragments
from .model.readouts import (
    build_accumulated_image_readout,
    build_interleaved_draft_readout,
    prepare_interleaved_draft_readout,
)


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    depth: int
    token_ids: torch.LongTensor
    drafts: tuple[torch.LongTensor, ...] = ()


def _eos_token_id(tokenizer: Any) -> int:
    """Return the only stop token of every decode path (``<|im_end|>``).

    The base ``generation_config.json`` also lists ``<|endoftext|>``; the
    paper's evaluation and training stopped only at ``tokenizer.eos_token_id``.
    """

    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Qwen generation requires an integer tokenizer.eos_token_id")
    return int(value)


def _language_text(question: str, prompt: str | None) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if prompt is None:
        return question
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string when supplied")
    return prompt


def _active_tokens(ids: torch.Tensor, mask: torch.Tensor) -> torch.LongTensor:
    if ids.shape != mask.shape or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("single-example decoded IDs and mask must match [1, T]")
    return ids[0, mask[0].bool()].detach().cpu()


def _until_eos(ids: torch.Tensor, eos_token_id: int) -> torch.LongTensor:
    """Cut ``generate`` output at its EOS, which greedy_decode never returns."""

    stops = (ids == eos_token_id).nonzero()
    return ids if stops.numel() == 0 else ids[: int(stops[0, 0])]


class AdaLoopQwenPipeline:
    """One-image public inference interface for SFT, AdaLN, and A/B/C.

    ``question`` is the bare question: the AdaLN condition (checkpoints
    trained with a constant question condition on that constant instead; SFT
    and no-AdaLN checkpoints have no condition) and, by default, the
    language-model text.  The optional ``prompt`` replaces only the
    language-model text (chat turn and Variant-C fragments), e.g. the question
    followed by an answer-format instruction.  Image paths (``~`` expanded)
    are opened as the paper's Hugging Face processor did (EXIF orientation
    applied, then RGB); in-memory images go to the processor unchanged.
    Decoding follows the pinned paper policy documented in
    :mod:`adaloop_qwen.model.generation`.
    """

    def __init__(self, loaded: LoadedQwenCheckpoint) -> None:
        self.loaded = loaded
        self.model = loaded.model
        self.processor = loaded.processor
        self.config = loaded.config

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, **kwargs: Any) -> "AdaLoopQwenPipeline":
        return cls(QwenCheckpointLoader.from_pretrained(checkpoint, **kwargs))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def max_context_length(self) -> int:
        """Return the context limit signed into the checkpoint's runtime contract."""

        return int(self.loaded.runtime_contract.max_context_length)

    def _load_image(self, image: Any) -> Any:
        if isinstance(image, (str, Path)):
            from PIL import Image, ImageOps

            # The paper's runs passed image paths to the Hugging Face processor, whose
            # load_image applies the EXIF orientation before converting to RGB.
            with Image.open(Path(os.path.expanduser(image))) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
        return image

    def _prepare(self, image: Any, prompt: str) -> dict[str, torch.Tensor]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        image = self._load_image(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        if "image_grid_thw" not in encoded:
            raise ValueError("Qwen processor did not return image_grid_thw")
        if encoded["image_grid_thw"].shape != (1, 3):
            raise ValueError("The public inference path requires one image")
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }

    def _condition_tokens(self, question: str) -> dict[str, torch.Tensor]:
        conditioner = self.config.conditioner
        if conditioner.mode == "none":
            return {}
        text = (
            conditioner.constant_text
            if conditioner.mode == "constant_question"
            else question
        )
        batch = tokenize_question_conditions(
            self.processor.tokenizer,
            [text],
            max_tokens=int(conditioner.max_tokens),
            device=self.device,
        )
        return {
            "condition_input_ids": batch.input_ids,
            "condition_attention_mask": batch.attention_mask,
        }

    def _decode(self, ids: torch.Tensor) -> str:
        return self.processor.tokenizer.decode(
            ids.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _generation_limit(self, max_new_tokens: int | None) -> int:
        limit = (
            self.config.generation.max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        if limit < 1:
            raise ValueError("max_new_tokens must be >= 1")
        return limit

    def _check_generate_budget(self, prompt_length: int, max_new_tokens: int) -> None:
        """Refuse a Hugging Face ``generate`` call beyond the signed context.

        ``greedy_decode`` applies the same check to the cached readouts.
        """

        maximum = self.max_context_length
        requested = prompt_length + max_new_tokens
        if requested > maximum:
            raise ValueError(
                f"Decode budget exceeds max_context_length={maximum}: "
                f"{prompt_length} + {max_new_tokens} = {requested}"
            )

    def _pooled_condition(
        self, condition: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        if not condition:
            return None
        return self.model.pool_question_condition(
            condition["condition_input_ids"],
            condition["condition_attention_mask"],
        )

    def _decode_readout(
        self,
        runtime: Any,
        readout: Any,
        step_features: Sequence[torch.Tensor],
        *,
        max_new_tokens: int,
        suppressed_token_ids: Sequence[int] = (),
        repetition_penalty: float = 1.0,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        state = runtime.prefill(readout, step_features)
        decoded = greedy_decode(
            runtime,
            state,
            max_new_tokens=max_new_tokens,
            eos_token_ids=_eos_token_id(self.processor.tokenizer),
            pad_token_id=int(self.processor.tokenizer.pad_token_id),
            suppressed_token_ids=suppressed_token_ids,
            max_context_length=self.max_context_length,
            repetition_penalty=repetition_penalty,
            context_token_ids=readout.input_ids,
        )
        return decoded.input_ids, decoded.attention_mask

    def _hf_generation_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        """Pin the paper's Hugging Face ``generate`` policy for SFT/AdaLN/A."""

        tokenizer = self.processor.tokenizer
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": REPETITION_PENALTY,
            "eos_token_id": _eos_token_id(tokenizer),
            "pad_token_id": int(tokenizer.pad_token_id),
            "use_cache": True,
        }
        from transformers import GenerationConfig

        # Start from library defaults so that no value of the checkpoint's
        # generation_config.json (sampling, extra stop or suppressed tokens,
        # ...) is merged into the pinned policy above.
        kwargs["generation_config"] = GenerationConfig()
        kwargs["use_model_defaults"] = False
        return kwargs

    @torch.no_grad()
    def generate(
        self,
        image: Any,
        question: str,
        *,
        prompt: str | None = None,
        depth: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        depth = self.config.train_depth if depth is None else int(depth)
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if self.config.variant in {"sft", "adaln"} and depth != 1:
            raise ValueError(f"{self.config.variant} supports only depth 1")
        limit = self._generation_limit(max_new_tokens)
        prompt = _language_text(question, prompt)
        inputs = self._prepare(image, prompt)
        condition = self._condition_tokens(question)
        eos = _eos_token_id(self.processor.tokenizer)

        if self.config.variant in {"sft", "adaln", "a"}:
            original_width = int(inputs["input_ids"].shape[1])
            self._check_generate_budget(original_width, limit)
            generation_kwargs = {**inputs, **self._hf_generation_kwargs(limit)}
            # The plain SFT release deliberately uses the unmodified upstream
            # Qwen class.  Do not leak AdaLoop-only arguments into it.
            if self.config.variant != "sft":
                generation_kwargs.update(condition)
                generation_kwargs["loop_depth"] = depth
            generated = self.model.generate(**generation_kwargs)
            tokens = _until_eos(generated[0, original_width:].detach().cpu(), eos)
            return GenerationResult(
                answer=self._decode(tokens), depth=depth, token_ids=tokens
            )

        condition_vector = self._pooled_condition(condition)
        step_features = self.model.collect_vision_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            condition_vector,
            range(1, depth + 1),
        )
        if self.config.variant == "b":
            # Variant B uses the standard Qwen accumulated-image container.
            readout = build_accumulated_image_readout(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["image_grid_thw"],
                target_depth=depth,
                image_token_id=int(self.model.config.image_token_id),
                vision_start_token_id=int(self.model.config.vision_start_token_id),
                vision_end_token_id=int(self.model.config.vision_end_token_id),
                spatial_merge_size=int(self.model.adaptive_visual.spatial_merge_size),
                preserve_width=False,
                max_length=self.max_context_length,
            )
            runtime = self.model.make_decode_runtime()
            decoded_ids, decoded_mask = self._decode_readout(
                runtime,
                readout,
                step_features,
                max_new_tokens=limit,
                repetition_penalty=REPETITION_PENALTY,
            )
            tokens = _active_tokens(decoded_ids, decoded_mask)
            return GenerationResult(
                answer=self._decode(tokens), depth=depth, token_ids=tokens
            )

        fragments = tokenize_variant_c_fragments(
            self.processor.tokenizer, [prompt], device=self.device
        )
        prepared = prepare_interleaved_draft_readout(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["image_grid_thw"],
            fragments,
            image_token_id=int(self.model.config.image_token_id),
            vision_start_token_id=int(self.model.config.vision_start_token_id),
            vision_end_token_id=int(self.model.config.vision_end_token_id),
            spatial_merge_size=int(self.model.adaptive_visual.spatial_merge_size),
        )
        suppressed = suppressed_special_token_ids(
            getattr(self.processor.tokenizer, "all_special_ids", ()),
            eos_token_ids=eos,
        )
        output = generate_variant_c_reprefill(
            self.model.make_decode_runtime(),
            prepared,
            step_features,
            target_depth=depth,
            draft_max_tokens=int(self.config.readout.draft_max_tokens),
            max_new_tokens=limit,
            eos_token_ids=eos,
            pad_token_id=int(self.processor.tokenizer.pad_token_id),
            suppressed_token_ids=suppressed,
            max_context_length=self.max_context_length,
        )
        tokens = _active_tokens(output.sequences, output.sequence_attention_mask)
        return GenerationResult(
            answer=self._decode(tokens),
            depth=depth,
            token_ids=tokens,
            drafts=tuple(item.detach().cpu() for item in output.drafts),
        )

    @torch.no_grad()
    def generate_p1(
        self,
        image: Any,
        question: str,
        *,
        prompt: str | None = None,
        max_depth: int | None = None,
        fallback_depth: int | None = None,
        max_new_tokens: int | None = None,
    ) -> P1Output[GenerationResult]:
        if self.config.variant not in {"a", "b", "c"}:
            raise ValueError("p=1 stopping requires a recurrent A/B/C checkpoint")
        prompt = _language_text(question, prompt)
        if isinstance(max_depth, bool):
            raise ValueError("p=1 stability requires max_depth >= 2")
        maximum = self.config.train_depth if max_depth is None else int(max_depth)
        if maximum < 2:
            raise ValueError("p=1 stability requires max_depth >= 2")
        if (
            isinstance(fallback_depth, bool)
            or (
                fallback_depth is not None
                and not 1 <= int(fallback_depth) <= maximum
            )
        ):
            raise ValueError("fallback_depth must be between 1 and max_depth")
        limit = self._generation_limit(max_new_tokens)
        inputs = self._prepare(image, prompt)
        condition = self._condition_tokens(question)
        condition_vector = self._pooled_condition(condition)
        step_features = self.model.collect_vision_features(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            condition_vector,
            range(1, maximum + 1),
        )
        runtime = self.model.make_decode_runtime()
        eos = _eos_token_id(self.processor.tokenizer)
        pad_token_id = int(self.processor.tokenizer.pad_token_id)

        if self.config.variant == "a":
            # Variant A always has one language-visible image container.  The
            # selected recurrent state changes with depth, but the text layout
            # does not, so every terminal decode consumes one feature tensor.
            readout = SimpleNamespace(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=None,
                source_image_grid_thw=inputs["image_grid_thw"],
                image_grid_thw=inputs["image_grid_thw"],
                target_depth=1,
            )

            def generate_at_depth(depth: int) -> GenerationResult:
                decoded_ids, decoded_mask = self._decode_readout(
                    runtime,
                    readout,
                    (step_features[depth - 1],),
                    max_new_tokens=limit,
                    repetition_penalty=REPETITION_PENALTY,
                )
                tokens = _active_tokens(decoded_ids, decoded_mask)
                return GenerationResult(
                    answer=self._decode(tokens), depth=depth, token_ids=tokens
                )

        elif self.config.variant == "b":

            def generate_at_depth(depth: int) -> GenerationResult:
                readout = build_accumulated_image_readout(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["image_grid_thw"],
                    target_depth=depth,
                    image_token_id=int(self.model.config.image_token_id),
                    vision_start_token_id=int(
                        self.model.config.vision_start_token_id
                    ),
                    vision_end_token_id=int(self.model.config.vision_end_token_id),
                    spatial_merge_size=int(
                        self.model.adaptive_visual.spatial_merge_size
                    ),
                    preserve_width=False,
                    max_length=self.max_context_length,
                )
                decoded_ids, decoded_mask = self._decode_readout(
                    runtime,
                    readout,
                    step_features[:depth],
                    max_new_tokens=limit,
                    repetition_penalty=REPETITION_PENALTY,
                )
                tokens = _active_tokens(decoded_ids, decoded_mask)
                return GenerationResult(
                    answer=self._decode(tokens), depth=depth, token_ids=tokens
                )

        else:
            fragments = tokenize_variant_c_fragments(
                self.processor.tokenizer, [prompt], device=self.device
            )
            prepared = prepare_interleaved_draft_readout(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["image_grid_thw"],
                fragments,
                image_token_id=int(self.model.config.image_token_id),
                vision_start_token_id=int(self.model.config.vision_start_token_id),
                vision_end_token_id=int(self.model.config.vision_end_token_id),
                spatial_merge_size=int(
                    self.model.adaptive_visual.spatial_merge_size
                ),
            )
            suppressed = suppressed_special_token_ids(
                getattr(self.processor.tokenizer, "all_special_ids", ()),
                eos_token_ids=eos,
            )
            drafts: list[torch.LongTensor] = []
            draft_masks: list[torch.Tensor] = []

            def generate_at_depth(depth: int) -> GenerationResult:
                # Calls arrive in depth order from generate_p1_stable.  Extend
                # the one shared C trajectory with exactly the draft needed by
                # the next depth; never regenerate an earlier draft.
                while len(drafts) < depth - 1:
                    round_depth = len(drafts) + 1
                    draft_readout = build_interleaved_draft_readout(
                        prepared,
                        drafts,
                        draft_masks,
                        target_depth=round_depth,
                        pad_token_id=pad_token_id,
                        draft_padding_side="right",
                        padding_side="left",
                        max_length=self.max_context_length,
                        include_answer=False,
                    )
                    draft_ids, draft_mask = self._decode_readout(
                        runtime,
                        draft_readout,
                        step_features[:round_depth],
                        max_new_tokens=int(self.config.readout.draft_max_tokens),
                        suppressed_token_ids=suppressed,
                    )
                    drafts.append(draft_ids.detach())
                    draft_masks.append(draft_mask.detach())

                final_readout = build_interleaved_draft_readout(
                    prepared,
                    drafts,
                    draft_masks,
                    target_depth=depth,
                    pad_token_id=pad_token_id,
                    draft_padding_side="right",
                    padding_side="left",
                    max_length=self.max_context_length,
                    include_answer=False,
                )
                decoded_ids, decoded_mask = self._decode_readout(
                    runtime,
                    final_readout,
                    step_features[:depth],
                    max_new_tokens=limit,
                    suppressed_token_ids=suppressed,
                )
                tokens = _active_tokens(decoded_ids, decoded_mask)
                return GenerationResult(
                    answer=self._decode(tokens),
                    depth=depth,
                    token_ids=tokens,
                    drafts=tuple(item.detach().cpu() for item in drafts),
                )

        return generate_p1_stable(
            maximum,
            generate_at_depth,
            answer_from_output=lambda result: result.answer,
            fallback_depth=fallback_depth,
        )


__all__ = ["AdaLoopQwenPipeline", "GenerationResult"]
