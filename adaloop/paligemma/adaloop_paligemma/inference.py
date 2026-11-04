"""Deterministic fixed-depth and answer-stability PaliGemma inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from adaloop_common.halting import select_p1_answer

from .checkpoint import PaliGemmaCheckpointLoader
from .constants import MULTI_IMAGE_VARIANTS, RECURRENT_VARIANTS
from .data import build_decode_batch, decode_generated_tokens

if TYPE_CHECKING:
    from .runtime import PaliGemmaRuntime


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    depth: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class DepthGeneration:
    depth: int
    output: GenerationResult


@dataclass(frozen=True)
class P1GenerationResult:
    answer: str
    depth: int
    stabilized: bool
    evaluated: tuple[DepthGeneration, ...]


class AdaLoopPaliGemmaPipeline:
    """One-image inference interface for PaliGemma SFT, AdaLN, and A/B/C."""

    def __init__(self, runtime: "PaliGemmaRuntime", *, device_put: bool = True) -> None:
        try:
            import jax
        except ImportError as exc:  # pragma: no cover - optional runtime extra
            raise RuntimeError(
                "PaliGemma inference requires JAX; install "
                "adaloop-paligemma[runtime]"
            ) from exc
        self.runtime = runtime
        self.model = runtime.model
        self.recipe = runtime.recipe
        self.tokenizer = runtime.tokenizer
        self.params = jax.device_put(runtime.params) if device_put else runtime.params
        self._compiled_prefill: dict[tuple[int, int, int], Any] = {}
        self._compiled_extend: Any | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        base_checkpoint: str | Path,
        tokenizer: str | Path,
        verify_hashes: bool = True,
        device_put: bool = True,
    ) -> "AdaLoopPaliGemmaPipeline":
        from .runtime import PaliGemmaRuntime

        verified = PaliGemmaCheckpointLoader.from_pretrained(
            checkpoint,
            base_checkpoint=base_checkpoint,
            tokenizer=tokenizer,
            verify_hashes=verify_hashes,
        )
        if not isinstance(verified.reconstructed, PaliGemmaRuntime):
            raise RuntimeError("PaliGemma checkpoint did not reconstruct a runtime")
        return cls(verified.reconstructed, device_put=device_put)

    def _validate_depth(self, depth: int) -> int:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if self.recipe.variant not in RECURRENT_VARIANTS and depth != 1:
            raise ValueError(f"{self.recipe.variant} supports only depth 1")
        return depth

    def _condition_question(self, question: str) -> str:
        if self.recipe.conditioner.mode == "constant_question":
            assert self.recipe.conditioner.constant_text is not None
            return self.recipe.conditioner.constant_text
        return question

    def _prefill_function(
        self, *, depth: int, prompt_width: int, max_new_tokens: int
    ):
        key = (depth, prompt_width, max_new_tokens)
        if key in self._compiled_prefill:
            return self._compiled_prefill[key]
        import jax

        from .model.trajectory import rollout_reprefill_with_apply

        model = self.model
        conditioned = self.recipe.conditioner.mode != "none"
        variant_c = self.recipe.variant == "c"

        def prefill(params, batch):
            kwargs = {}
            if conditioned:
                kwargs = {
                    "condition_input_ids": batch["condition_input_ids"],
                    "condition_attention_mask": batch[
                        "condition_attention_mask"
                    ],
                }
            rollout_kwargs = {}
            if variant_c:
                drafts, draft_mask = rollout_reprefill_with_apply(
                    model,
                    params,
                    image=batch["image"],
                    text=batch["text"],
                    mask_ar=batch["mask_ar"],
                    input_mask=batch["mask_input"],
                    condition_input_ids=kwargs["condition_input_ids"],
                    condition_attention_mask=kwargs[
                        "condition_attention_mask"
                    ],
                    steps=depth,
                    draft_max_tokens=int(model.draft_max_tokens),
                    eos_token_id=int(model.eos_token_id),
                    suppressed_token_ids=tuple(model.suppressed_draft_token_ids),
                )
                rollout_kwargs = {"drafts": drafts, "draft_mask": draft_mask}
            result, mutable = model.apply(
                {"params": params},
                batch["image"],
                batch["text"],
                batch["mask_ar"],
                input_mask=batch["mask_input"],
                max_new_tokens=max_new_tokens,
                n_steps=depth,
                **kwargs,
                **rollout_kwargs,
                method=model.prefill_generation,
                mutable=("cache",),
            )
            return result.next_token_logits, mutable["cache"]

        compiled = jax.jit(prefill)
        self._compiled_prefill[key] = compiled
        return compiled

    def _extend_function(self):
        if self._compiled_extend is not None:
            return self._compiled_extend
        import jax

        model = self.model

        def extend(params, cache, token_ids):
            logits, mutable = model.apply(
                {"params": params, "cache": cache},
                token_ids,
                method=model.extend_cache,
                mutable=("cache",),
            )
            return logits, mutable["cache"]

        self._compiled_extend = jax.jit(extend)
        return self._compiled_extend

    @staticmethod
    def _validate_texts(question: str, prompt: str | None) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("prompt must be a non-empty string")

    def _new_token_limit(self, max_new_tokens: int | None) -> int:
        limit = (
            int(self.recipe.generation.max_new_tokens)
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        if limit < 1:
            raise ValueError("max_new_tokens must be positive")
        return limit

    def _decode_batch(
        self, image: Any, question: str, prompt: str | None
    ) -> tuple[dict[str, Any], int]:
        import jax

        prompt_capacity = int(self.recipe.preprocessing.eval_max_text_tokens)
        conditioned = self.recipe.conditioner.mode != "none"
        batch = build_decode_batch(
            tokenizer=self.tokenizer,
            images=[image],
            questions=[question],
            prompts=[prompt],
            max_text_len=prompt_capacity,
            image_size=int(self.recipe.preprocessing.image_size),
            prefix_template=self.recipe.preprocessing.eval_prompt_template,
            include_condition=conditioned,
        )
        prefix_length = int(batch["mask_input"][0].sum())
        if prefix_length > prompt_capacity:
            raise ValueError(
                f"Prompt needs {prefix_length} tokens but the signed checkpoint "
                f"limit is {prompt_capacity}"
            )
        if conditioned and self._condition_question(question) != question:
            # Rebuild only the separate condition; the language prompt remains
            # the user's prompt/question for the constant-question control.
            from .data import build_question_condition

            ids, mask = build_question_condition(
                tokenizer=self.tokenizer,
                question=self._condition_question(question),
            )
            batch["condition_input_ids"][0] = ids
            batch["condition_attention_mask"][0] = mask

        batch = {
            name: jax.device_put(value)
            for name, value in batch.items()
            if name != "row_mask"
        }
        return batch, prompt_capacity

    def generate(
        self,
        image: Any,
        question: str,
        *,
        prompt: str | None = None,
        depth: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Greedily decode one answer at a fixed recurrent depth.

        ``question`` is the bare question: the AdaLN condition (checkpoints
        trained with a constant question condition on that constant instead;
        SFT checkpoints have no condition) and, by default, the language
        prompt.  ``prompt`` optionally replaces only the language-visible
        text (including every Variant C round), e.g. the question plus the
        paper's answer-format instruction.
        """

        import jax

        self._validate_texts(question, prompt)
        requested_depth = self.recipe.train_depth if depth is None else depth
        requested_depth = self._validate_depth(requested_depth)
        limit = self._new_token_limit(max_new_tokens)
        batch, prompt_capacity = self._decode_batch(image, question, prompt)
        prefill = self._prefill_function(
            depth=requested_depth,
            prompt_width=prompt_capacity,
            max_new_tokens=limit,
        )
        logits, cache = prefill(self.params, batch)
        extend = self._extend_function()
        generated: list[int] = []
        eos = int(self.tokenizer.eos_id())
        for token_index in range(limit):
            row = logits[:, -1] if logits.ndim == 3 else logits
            token = int(np.asarray(jax.device_get(jax.numpy.argmax(row[0]))))
            if token == eos:
                break
            generated.append(token)
            if token_index + 1 < limit:
                token_ids = jax.numpy.asarray([[token]], dtype=jax.numpy.int32)
                logits, cache = extend(self.params, cache, token_ids)
        return GenerationResult(
            answer=decode_generated_tokens(self.tokenizer, generated),
            depth=requested_depth,
            token_ids=tuple(generated),
        )

    def generate_p1(
        self,
        image: Any,
        question: str,
        *,
        prompt: str | None = None,
        max_depth: int | None = None,
        fallback_depth: int | None = None,
        max_new_tokens: int | None = None,
    ) -> P1GenerationResult:
        """Run fixed depths and select the first repeated adjacent answer.

        ``question`` and ``prompt`` have the same meaning as in
        :meth:`generate` and are reused at every depth.
        """

        if self.recipe.variant not in RECURRENT_VARIANTS:
            raise ValueError("p=1 stopping requires a recurrent A/B/C checkpoint")
        maximum = self.recipe.train_depth if max_depth is None else int(max_depth)
        if maximum < 2:
            raise ValueError("p=1 stopping requires max_depth >= 2")
        fallback = maximum if fallback_depth is None else int(fallback_depth)
        if not 1 <= fallback <= maximum:
            raise ValueError("fallback_depth must lie in [1, max_depth]")
        if self.recipe.variant in MULTI_IMAGE_VARIANTS:
            # Trace (without running) the deepest prefill so a depth whose
            # packed prompt plus answer exceeds the B/C capacity fails before
            # any shallower depth is decoded.
            import jax

            self._validate_texts(question, prompt)
            limit = self._new_token_limit(max_new_tokens)
            batch, prompt_capacity = self._decode_batch(image, question, prompt)
            jax.eval_shape(
                self._prefill_function(
                    depth=maximum,
                    prompt_width=prompt_capacity,
                    max_new_tokens=limit,
                ),
                self.params,
                batch,
            )
        evaluated: list[DepthGeneration] = []
        answers: list[str] = []
        for depth in range(1, maximum + 1):
            output = self.generate(
                image,
                question,
                prompt=prompt,
                depth=depth,
                max_new_tokens=max_new_tokens,
            )
            evaluated.append(DepthGeneration(depth=depth, output=output))
            answers.append(output.answer)
            # Stop at the first repeat: deeper depths cannot change the choice.
            if depth > 1:
                selection = select_p1_answer(answers)
                if selection.stabilized:
                    break
        else:
            selection = select_p1_answer(answers, fallback_depth=fallback)
        return P1GenerationResult(
            answer=selection.answer,
            depth=selection.depth,
            stabilized=selection.stabilized,
            evaluated=tuple(evaluated),
        )


__all__ = [
    "AdaLoopPaliGemmaPipeline",
    "DepthGeneration",
    "GenerationResult",
    "P1GenerationResult",
]
