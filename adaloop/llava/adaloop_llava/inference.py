"""Single-example inference with a portable LLaVA-1.5 AdaLoop checkpoint.

:class:`AdaLoopLlavaPipeline` decodes with the generation step of the paper evaluation
(:func:`adaloop_llava.generation.generate_ids`, as in :func:`~adaloop_llava.eval.generate.run_depth`)
on a one-query batch. The prompt is the Vicuna-v1 conversation with the image placeholder and the
question; the question condition of loop models is BOS plus the question tokens (at most 128
tokens); decoding is greedy with at most 32 new tokens and no repetition penalty (the checkpoint's
runtime contract).

* :meth:`AdaLoopLlavaPipeline.generate` answers at one depth: 1 for SFT and single-pass AdaLN,
  the training depth (3) for Variants A, B and C by default. Every depth reruns the vision tower
  (prelude, ``depth`` band steps, coda; Variant C also the drafts of rounds ``1..depth-1``).
* :meth:`AdaLoopLlavaPipeline.generate_p1` (Variants A, B and C) evaluates depths ``1, 2, ...`` in
  order and stops at the first depth whose answer repeats the previous one after case and
  whitespace normalisation (:func:`adaloop_common.select_p1_answer`); without a repeat it returns
  the answer at ``fallback_depth`` (default: the last depth).

The paper's numbers come from batches of 16 left-padded records per depth with offline halting
over the stored answers. That gives the same answers as one record at a time up to floating-point
near-ties, which batch composition can flip.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from adaloop_common import select_p1_answer

from .checkpoint import LlavaCheckpointLoader, LoadedLlavaCheckpoint


@dataclass(frozen=True)
class GenerationResult:
    """The answer at one depth: the decoded text (special tokens skipped, stripped), the depth
    and the generated token ids (up to, not including, the first EOS)."""

    answer: str
    depth: int
    token_ids: Tuple[int, ...]


@dataclass(frozen=True)
class DepthGeneration:
    depth: int
    output: GenerationResult


@dataclass(frozen=True)
class P1Result:
    """The answer chosen by ``p = 1`` answer-stability halting and every depth evaluated for it."""

    answer: str
    depth: int
    stabilized: bool
    evaluated: Tuple[DepthGeneration, ...]


@contextlib.contextmanager
def _image_file(image: Any) -> Iterator[str]:
    """An image path for :func:`~adaloop_llava.generation.generate_ids`: the given file, or a
    lossless PNG of a PIL image converted to RGB (generation opens images and converts them to RGB)."""
    if isinstance(image, (str, os.PathLike)):
        path = Path(os.path.expanduser(os.fspath(image)))
        if not path.is_file():
            raise FileNotFoundError(f"image {image} does not exist")
        yield str(path.resolve())
        return
    from PIL import Image
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a path or a PIL image, got {type(image).__name__}")
    with tempfile.TemporaryDirectory(prefix="adaloop_llava_image_") as tmp:
        path = os.path.join(tmp, "image.png")
        image.convert("RGB").save(path, format="PNG")
        yield path


class AdaLoopLlavaPipeline:
    """One-image inference for LLaVA-1.5 SFT, single-pass AdaLN and Variant A, B and C checkpoints."""

    def __init__(self, loaded: LoadedLlavaCheckpoint) -> None:
        self.loaded = loaded
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.image_processor = loaded.image_processor
        self.recipe = loaded.recipe
        self.runtime = loaded.runtime
        self.manifest = loaded.manifest
        if self.tokenizer is None:
            raise ValueError("the pipeline needs the tokenizer (load with with_tokenizer=True)")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token          # as in training and evaluation
        self.variant = self.recipe.variant
        self.train_depth = self.manifest.train_depth
        condition = self.runtime.get("condition") or {}
        self.condition_max_length = int(condition.get("max_tokens", self.recipe.data.condition_max_length))
        self.default_max_new_tokens = int(self.runtime["decode"]["max_new_tokens"])

    @classmethod
    def from_pretrained(cls, checkpoint: Any, *, base_model: Any, vision_tower: Any, device: str = "cuda",
                        dtype: Any = "float16", attn_implementation: Optional[str] = "sdpa",
                        verify_hashes: bool = True, verify_base_hashes: bool = False) -> "AdaLoopLlavaPipeline":
        """Verify the checkpoint's manifest and the supplied base and CLIP directories, then load
        the model (:meth:`adaloop_llava.checkpoint.LlavaCheckpointLoader.from_pretrained`). The
        language model and CLIP run in ``dtype`` (the paper evaluation used float16); the loop
        core stays in float32."""
        return cls(LlavaCheckpointLoader.from_pretrained(
            checkpoint, base_model=base_model, vision_tower=vision_tower, device=device, dtype=dtype,
            attn_implementation=attn_implementation, verify_hashes=verify_hashes,
            verify_base_hashes=verify_base_hashes))

    @property
    def is_loop(self) -> bool:
        return getattr(self.model.get_vision_tower(), "loop_spec", None) is not None

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _depth(self, depth: Optional[int]) -> int:
        depth = self.train_depth if depth is None else depth
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError(f"depth must be a positive integer, got {depth!r}")
        if not self.is_loop and depth != 1:
            raise ValueError(f"this {self.variant} checkpoint has no loop tower; only depth 1 exists, got {depth}")
        return depth

    def _max_new_tokens(self, value: Optional[int]) -> int:
        limit = self.default_max_new_tokens if value is None else value
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"max_new_tokens must be a positive integer, got {value!r}")
        return limit

    @staticmethod
    def _check_texts(question: str, prompt: Optional[str]) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("prompt must be a non-empty string")

    def generate(self, image: Any, question: str, *, prompt: Optional[str] = None, depth: Optional[int] = None,
                 max_new_tokens: Optional[int] = None) -> GenerationResult:
        """Greedy answer at one depth.

        ``question`` is the bare question: the question condition of loop models and, unless
        ``prompt`` is given, the user turn of the language-model prompt. ``prompt`` replaces only
        the language-model text and never reaches the condition. ``image`` is a path or a PIL
        image."""
        from .generation import generate_ids

        self._check_texts(question, prompt)
        depth = self._depth(depth)
        limit = self._max_new_tokens(max_new_tokens)
        with _image_file(image) as path:
            query = {"question": question, "image": path}
            if prompt is not None:
                query["prompt"] = prompt
            out = generate_ids(self.model, self.tokenizer, self.image_processor, [query], depth,
                               is_loop=self.is_loop, device=self.device, max_new_tokens=limit,
                               condition_max_length=self.condition_max_length)
        # The decoding of run_depth, for a batch of one.
        text = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        ids = [int(t) for t in out[0].tolist()]
        eos = self.tokenizer.eos_token_id
        if eos is not None and eos in ids:
            ids = ids[:ids.index(eos)]
        return GenerationResult(answer=text.strip(), depth=depth, token_ids=tuple(ids))

    def generate_p1(self, image: Any, question: str, *, prompt: Optional[str] = None,
                    max_depth: Optional[int] = None, fallback_depth: Optional[int] = None,
                    max_new_tokens: Optional[int] = None) -> P1Result:
        """``p = 1`` answer-stability halting (Variants A, B and C): depths ``1..max_depth`` (default: the
        training depth) in order, stopping at the first depth whose answer equals the previous one
        after case and whitespace normalisation; without a repeat, the answer at
        ``fallback_depth`` (default ``max_depth``). ``question`` and ``prompt`` are as in
        :meth:`generate`."""
        if self.variant not in ("a", "b", "c"):
            raise ValueError(f"p=1 halting needs a Variant A, B or C checkpoint; this one is {self.variant!r}")
        maximum = self.train_depth if max_depth is None else max_depth
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 2:
            raise ValueError(f"p=1 halting needs max_depth >= 2, got {max_depth!r}")
        fallback = maximum if fallback_depth is None else fallback_depth
        if isinstance(fallback, bool) or not isinstance(fallback, int) or not 1 <= fallback <= maximum:
            raise ValueError(f"fallback_depth must lie in [1, {maximum}], got {fallback_depth!r}")
        self._check_texts(question, prompt)
        self._max_new_tokens(max_new_tokens)
        evaluated = []
        for depth in range(1, maximum + 1):
            evaluated.append(DepthGeneration(depth, self.generate(image, question, prompt=prompt, depth=depth,
                                                                  max_new_tokens=max_new_tokens)))
            answers = [item.output.answer for item in evaluated]
            if depth == maximum:
                selection = select_p1_answer(answers, fallback_depth=fallback)
            elif depth >= 2:
                selection = select_p1_answer(answers)
                if not selection.stabilized:
                    continue
            else:
                continue
            return P1Result(answer=selection.answer, depth=selection.depth, stabilized=selection.stabilized,
                            evaluated=tuple(evaluated))
        raise AssertionError("unreachable")


__all__ = ["AdaLoopLlavaPipeline", "DepthGeneration", "GenerationResult", "P1Result"]
