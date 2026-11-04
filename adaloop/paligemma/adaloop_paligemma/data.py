"""Image/question/answer input pipeline for PaliGemma AdaLoop.

Training examples come from :mod:`adaloop_common.data`: the VisRO training
split of the chosen tasks, optionally followed by a replay JSONL file.  Images
are read lazily, each checked against its record's ``image_sha256`` when it
has one.

This module reproduces the PaliGemma training convention used by the paper:

* RGB images are resized to 448 x 448 and normalized to ``[-1, 1]``;
* the language prefix is ``<bos>answer en {question}\n``;
* only answer and EOS tokens are supervised;
* the independent AdaLN condition is the raw question, with no BOS/EOS/chat
  tokens, right-padded after at most 128 semantic tokens.

A recipe with the release-extension ``constant_question`` conditioner mode
conditions on its ``constant_text`` instead, tokenized the same way, while
the language prefix keeps the real question.

JAX is intentionally not imported here.  The returned NumPy arrays can be
device-placed by the training runtime and the preprocessing helpers remain
unit-testable on CPU.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import numpy as np

from adaloop_common.data import ImageQAExample

from .constants import (
    PAPER_EVAL_PROMPT_TEMPLATE,
    PAPER_IMAGE_SIZE,
    PAPER_QUESTION_MAX_TOKENS,
    PAPER_TRAIN_PROMPT_TEMPLATE,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def _normalize_pil_image(image: PILImage, image_size: int) -> np.ndarray:
    from PIL import Image

    rgb = image.convert("RGB")
    resized = rgb.resize((image_size, image_size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    return values * np.float32(2.0) - np.float32(1.0)


def _uint8_pixels(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(
            "Image arrays must be uint8 pixels or preprocessed floating-point "
            f"pixels; got dtype {image.dtype}"
        )
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.size == 0 or not (
        image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in (3, 4))
    ):
        raise ValueError(
            "uint8 image arrays must have shape [H, W], [H, W, 1], "
            f"[H, W, 3] (RGB), or [H, W, 4] (RGBA); got {image.shape}"
        )
    return np.ascontiguousarray(image)


def _preprocessed_pixels(image: np.ndarray, image_size: int) -> np.ndarray:
    expected = (image_size, image_size, 3)
    guidance = "pass raw pixels as uint8, a PIL image, a path, or encoded bytes"
    values = image.astype(np.float32)
    if (
        values.shape != expected
        or not np.isfinite(values).all()
        or values.min() < -1.0
        or values.max() > 1.0
    ):
        raise ValueError(
            f"Floating-point image arrays must be preprocessed {expected} "
            f"pixels in [-1, 1] (got shape {image.shape}); {guidance}"
        )
    if values.min() >= 0.0 and values.max() > values.min():
        # Normalization maps every channel value below 128 to a negative
        # number, so a varying array without one holds raw [0, 1] pixels.
        raise ValueError(
            "Floating-point image arrays must be preprocessed pixels in "
            "[-1, 1], but this non-negative, non-constant array looks like "
            f"raw [0, 1] pixels; {guidance}"
        )
    return values


def decode_and_resize(
    image: bytes | bytearray | memoryview | str | Path | np.ndarray | PILImage,
    *,
    image_size: int = PAPER_IMAGE_SIZE,
) -> np.ndarray:
    """Decode one image to float32 ``[H, W, 3]`` in ``[-1, 1]``.

    Encoded files (paths or bytes), ``PIL.Image`` objects, and raw ``uint8``
    pixel arrays shaped ``[H, W]``, ``[H, W, 1]``, ``[H, W, 3]`` (RGB), or
    ``[H, W, 4]`` (RGBA) all take the same RGB conversion, bicubic resize, and
    normalization.  A floating-point array is accepted only as an already
    preprocessed ``[image_size, image_size, 3]`` image in ``[-1, 1]`` (this
    function's output) and is returned unchanged.  Normalizing a real image
    maps every channel value below 128 to a negative number, so a
    non-negative float array that is not constant is rejected as raw
    ``[0, 1]`` pixels; constant arrays such as all-zero placeholders are
    accepted.  Raw pixels must be ``uint8`` (or a PIL image, path, or bytes).
    """

    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    if isinstance(image, np.ndarray) and np.issubdtype(image.dtype, np.floating):
        return _preprocessed_pixels(image, image_size)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a runtime dependency
        raise RuntimeError("Image preprocessing requires Pillow") from exc

    try:
        if isinstance(image, Image.Image):
            # Never close a caller-owned image.
            return _normalize_pil_image(image, image_size)
        if isinstance(image, np.ndarray):
            return _normalize_pil_image(
                Image.fromarray(_uint8_pixels(image)), image_size
            )
        if isinstance(image, (str, Path)):
            opened = Image.open(Path(image).expanduser())
        else:
            opened = Image.open(io.BytesIO(bytes(image)))
        with opened:
            return _normalize_pil_image(opened, image_size)
    except OSError as exc:
        raise ValueError(f"Could not decode image {image!r}: {exc}") from exc


def _encode(tokenizer: Any, text: str, *, add_bos: bool) -> list[int]:
    """Use the pinned Big Vision tokenizer API and validate its result."""

    if not hasattr(tokenizer, "Encode"):
        raise TypeError("PaliGemma tokenizer must provide Encode(text, add_bos=...)")
    values = [int(item) for item in tokenizer.Encode(text, add_bos=add_bos)]
    if any(item < 0 for item in values):
        raise ValueError("Tokenizer returned a negative token ID")
    return values


def _eos_id(tokenizer: Any) -> int:
    if not hasattr(tokenizer, "eos_id"):
        raise TypeError("PaliGemma tokenizer must provide eos_id()")
    value = int(tokenizer.eos_id())
    if value < 0:
        raise ValueError("Tokenizer returned an invalid EOS token ID")
    return value


def _bos_id(tokenizer: Any) -> int | None:
    getter = getattr(tokenizer, "bos_id", None)
    if getter is None:
        return None
    value = int(getter() if callable(getter) else getter)
    return value if value >= 0 else None


def build_text_and_masks(
    *,
    tokenizer: Any,
    question: str,
    answer: str,
    max_text_len: int,
    prefix_template: str = PAPER_TRAIN_PROMPT_TEMPLATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build PaliGemma teacher-forcing tokens and masks.

    Arrays have length ``max_text_len + 1`` because the training step predicts
    ``text[:, 1:]`` from ``text[:, :-1]``.  Over-length rows are truncated
    exactly as by the data loader that trained the paper checkpoints:
    the prefix is right-truncated so that the whole answer still fits, and
    only an answer that alone exceeds the budget is truncated, always keeping
    EOS as its supervised stopping target.
    """

    if isinstance(max_text_len, bool) or not isinstance(max_text_len, int):
        raise ValueError("max_text_len must be an integer")
    if max_text_len < 1:
        raise ValueError("max_text_len must be positive")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    if not isinstance(prefix_template, str) or "{question}" not in prefix_template:
        raise ValueError("prefix_template must contain {question}")

    prefix = prefix_template.format(question=question).rstrip("\n")
    prefix_ids = _encode(tokenizer, prefix, add_bos=True)
    separator_ids = _encode(tokenizer, "\n", add_bos=False)
    eos_id = _eos_id(tokenizer)
    answer_ids = _encode(tokenizer, answer, add_bos=False) + [eos_id]
    prefix_part = prefix_ids + separator_ids
    if not prefix_part:
        raise ValueError("Prompt tokenized to an empty prefix")
    length = max_text_len + 1
    if len(prefix_part) + len(answer_ids) > length:
        # Truncation as in the paper runs: keep at least one prefix token (BOS) and the
        # complete answer when possible; the separator may be cut.
        prefix_part = prefix_part[: max(1, length - len(answer_ids))]
        answer_budget = length - len(prefix_part)
        if len(answer_ids) > answer_budget:
            answer_ids = answer_ids[: max(0, answer_budget - 1)] + [eos_id]

    text = np.zeros(length, dtype=np.int32)
    mask_ar = np.ones(length, dtype=np.int32)
    mask_loss = np.zeros(length, dtype=np.int32)
    prefix_length = len(prefix_part)
    answer_length = len(answer_ids)
    text[:prefix_length] = prefix_part
    text[prefix_length : prefix_length + answer_length] = answer_ids
    mask_ar[:prefix_length] = 0
    mask_loss[prefix_length : prefix_length + answer_length] = 1
    return text, mask_ar, mask_loss


def build_question_condition(
    *,
    tokenizer: Any,
    question: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize the role-free AdaLN condition without BOS or EOS.

    The question is truncated to the recipe's 128 condition tokens.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    ids = _encode(tokenizer, question, add_bos=False)[:PAPER_QUESTION_MAX_TOKENS]
    if not ids:
        raise ValueError("Question condition tokenized to an empty sequence")
    forbidden = {_eos_id(tokenizer)}
    bos = _bos_id(tokenizer)
    if bos is not None:
        forbidden.add(bos)
    present = forbidden.intersection(ids)
    if present:
        raise ValueError(
            "Bare question condition unexpectedly contains BOS/EOS: "
            f"{sorted(present)}"
        )

    values = np.zeros(PAPER_QUESTION_MAX_TOKENS, dtype=np.int32)
    mask = np.zeros(PAPER_QUESTION_MAX_TOKENS, dtype=np.bool_)
    values[: len(ids)] = ids
    mask[: len(ids)] = True
    return values, mask


def materialize_example(
    example: ImageQAExample,
    *,
    tokenizer: Any,
    max_text_len: int,
    image_size: int = PAPER_IMAGE_SIZE,
    prefix_template: str = PAPER_TRAIN_PROMPT_TEMPLATE,
    include_condition: bool = True,
    condition_question: str | None = None,
) -> dict[str, np.ndarray]:
    """Decode and tokenize one example into the model's public batch contract."""

    text, mask_ar, mask_loss = build_text_and_masks(
        tokenizer=tokenizer,
        question=example.question,
        answer=example.answer,
        max_text_len=max_text_len,
        prefix_template=prefix_template,
    )
    result: dict[str, np.ndarray] = {
        "image": decode_and_resize(example.image.read_bytes(), image_size=image_size),
        "text": text,
        "mask_ar": mask_ar,
        "mask_loss": mask_loss,
    }
    if include_condition:
        condition_ids, condition_mask = build_question_condition(
            tokenizer=tokenizer,
            question=(
                example.question
                if condition_question is None
                else condition_question
            ),
        )
        result["condition_input_ids"] = condition_ids
        result["condition_attention_mask"] = condition_mask
    return result


def _stack(rows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    keys = tuple(rows[0])
    expected = set(keys)
    for index, row in enumerate(rows[1:], start=1):
        if set(row) != expected:
            raise ValueError(f"Batch row {index} has different tensor fields")
    return {key: np.stack([np.asarray(row[key]) for row in rows]) for key in keys}


def epoch_indices(sample_count: int, *, seed: int, epoch: int) -> np.ndarray:
    """Return the deterministic shuffled index sequence of one epoch."""

    for name, value, minimum in (
        ("sample_count", sample_count, 1),
        ("epoch", epoch, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    # Match the paper trainer's epoch stream exactly.  The large odd stride
    # prevents adjacent epochs from sharing a trivially related PRNG seed,
    # while the Python ``random`` algorithm is part of the run contract
    # used by the released checkpoints.
    shuffled_list = list(range(sample_count))
    random.Random(int(seed) + 1_000_003 * int(epoch)).shuffle(shuffled_list)
    return np.asarray(shuffled_list, dtype=np.int64)


def iter_epoch_batches(
    examples: Sequence[ImageQAExample],
    *,
    tokenizer: Any,
    batch_size: int,
    max_text_len: int,
    seed: int,
    epoch: int,
    process_index: int = 0,
    process_count: int = 1,
    drop_remainder: bool = True,
    image_size: int = PAPER_IMAGE_SIZE,
    prefix_template: str = PAPER_TRAIN_PROMPT_TEMPLATE,
    include_condition: bool = True,
    condition_question: str | None = None,
    skip_batches: int = 0,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield deterministic, lazily decoded batches for one epoch.

    ``skip_batches`` resumes mid-epoch: the first ``skip_batches`` per-host
    batches are skipped in the index stream, without decoding or tokenizing
    them, and the remaining batches are identical to an uninterrupted epoch.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(skip_batches, bool)
        or not isinstance(skip_batches, int)
        or skip_batches < 0
    ):
        raise ValueError("skip_batches must be a non-negative integer")
    # Truncate in *global-batch* units before host sharding.  Truncating each
    # host independently can leave different hosts with different numbers of
    # batches, which deadlocks multihost collectives near an epoch boundary.
    global_indices = epoch_indices(len(examples), seed=seed, epoch=epoch)
    if drop_remainder:
        global_batch = batch_size * process_count
        usable = (len(global_indices) // global_batch) * global_batch
        global_indices = global_indices[:usable]
    indices = global_indices[process_index::process_count]
    for start in range(skip_batches * batch_size, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        if len(selected) < batch_size and drop_remainder:
            break
        rows = [
            materialize_example(
                examples[int(index)],
                tokenizer=tokenizer,
                max_text_len=max_text_len,
                image_size=image_size,
                prefix_template=prefix_template,
                include_condition=include_condition,
                condition_question=condition_question,
            )
            for index in selected
        ]
        yield _stack(rows)


def build_decode_batch(
    *,
    tokenizer: Any,
    images: Sequence[
        bytes | bytearray | memoryview | str | Path | np.ndarray | PILImage
    ],
    questions: Sequence[str],
    prompts: Sequence[str | None] | None = None,
    max_text_len: int,
    image_size: int = PAPER_IMAGE_SIZE,
    prefix_template: str = PAPER_EVAL_PROMPT_TEMPLATE,
    include_condition: bool = True,
    pad_to_multiple: int = 1,
) -> dict[str, np.ndarray]:
    """Build a padded greedy-decode batch with stable shapes.

    Each entry of ``questions`` is the bare question.  It is the separate
    AdaLN condition, tokenized independently and never including the
    language trajectory's BOS/newline, and it is also the language prompt
    unless the matching ``prompts`` entry supplies the full language-visible
    text (for example the question plus an answer-format instruction).  The
    paper's evaluation filled ``prefix_template`` with that benchmark prompt
    while conditioning on the bare question.  ``images`` accepts every input
    form of :func:`decode_and_resize`.  ``row_mask`` marks the real rows (padding
    rows are false) and is not a model input.
    """

    if len(images) != len(questions) or not images:
        raise ValueError("images and questions must be non-empty and aligned")
    if prompts is None:
        prompts = [None] * len(questions)
    elif len(prompts) != len(questions):
        raise ValueError("prompts must align with questions")
    if (
        isinstance(pad_to_multiple, bool)
        or not isinstance(pad_to_multiple, int)
        or pad_to_multiple < 1
    ):
        raise ValueError("pad_to_multiple must be a positive integer")
    if isinstance(max_text_len, bool) or not isinstance(max_text_len, int):
        raise ValueError("max_text_len must be an integer")
    if max_text_len < 1:
        raise ValueError("max_text_len must be positive")
    if not isinstance(prefix_template, str) or "{question}" not in prefix_template:
        raise ValueError("prefix_template must contain {question}")

    padded_size = ((len(images) + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    image_batch = np.zeros(
        (padded_size, image_size, image_size, 3), dtype=np.float32
    )
    text = np.zeros((padded_size, max_text_len), dtype=np.int32)
    input_mask = np.zeros((padded_size, max_text_len), dtype=np.bool_)
    mask_ar = np.ones((padded_size, max_text_len), dtype=np.int32)
    row_mask = np.zeros(padded_size, dtype=np.bool_)
    condition_ids = np.zeros(
        (padded_size, PAPER_QUESTION_MAX_TOKENS), dtype=np.int32
    )
    condition_mask = np.zeros_like(condition_ids, dtype=np.bool_)
    separator = _encode(tokenizer, "\n", add_bos=False)

    for index, (image, question, prompt) in enumerate(
        zip(images, questions, prompts, strict=True)
    ):
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"questions[{index}] must be a non-empty string")
        if prompt is None:
            prompt = question
        elif not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompts[{index}] must be a non-empty string or None")
        decoded = decode_and_resize(image, image_size=image_size)
        prefix = prefix_template.format(question=prompt).rstrip("\n")
        prompt_ids = _encode(tokenizer, prefix, add_bos=True) + separator
        if len(prompt_ids) > max_text_len:
            raise ValueError(
                f"Decode prompt {index} needs {len(prompt_ids)} tokens but "
                f"max_text_len={max_text_len}"
            )
        image_batch[index] = decoded
        text[index, : len(prompt_ids)] = prompt_ids
        input_mask[index, : len(prompt_ids)] = True
        mask_ar[index, : len(prompt_ids)] = 0
        row_mask[index] = True
        if include_condition:
            ids, mask = build_question_condition(
                tokenizer=tokenizer,
                question=question,
            )
            condition_ids[index] = ids
            condition_mask[index] = mask

    batch = {
        "image": image_batch,
        "text": text,
        "mask_input": input_mask,
        "mask_ar": mask_ar,
        "row_mask": row_mask,
    }
    if include_condition:
        batch["condition_input_ids"] = condition_ids
        batch["condition_attention_mask"] = condition_mask
    return batch


def decode_generated_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode positive IDs up to (but excluding) the first EOS token."""

    eos = _eos_id(tokenizer)
    retained: list[int] = []
    for raw in token_ids:
        token = int(raw)
        if token == eos:
            break
        if token > 0:
            retained.append(token)
    if not hasattr(tokenizer, "Decode"):
        raise TypeError("PaliGemma tokenizer must provide Decode(ids)")
    return str(tokenizer.Decode(retained)).strip() if retained else ""
