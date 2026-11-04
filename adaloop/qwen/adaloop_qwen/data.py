"""Image-question-answer training pipeline for Qwen2.5-VL.

Training examples come from :mod:`adaloop_common.data`: the VisRO training split of the chosen
tasks, optionally followed by a replay JSONL file.  Images are read lazily, each checked against
its record's ``image_sha256`` when it has one.  Prompt formatting remains owned by the
checkpoint's ``AutoProcessor``; this module only verifies the assistant boundary and masks every
prompt token exactly.  The role-free AdaLN condition and Variant-C fragments are tokenized by
:mod:`adaloop_qwen.model.prompting`, the code inference uses.

Torch, Pillow, and Transformers are optional package dependencies.  They are
therefore imported only on the code paths that need them.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from adaloop_common.data import ImageQAExample, ImageSource

from .config import CONDITION_MAX_TOKENS, READOUT_FOR_VARIANT


IGNORE_INDEX = -100
_READOUT_MODES = frozenset(READOUT_FOR_VARIANT.values())


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "AdaLoop Qwen training requires PyTorch; install adaloop-qwen[gpu]"
        ) from exc
    return torch


def _one_dimensional_ids(value: Any, *, name: str) -> Any:
    """Normalize a token tensor to ``[T]`` without importing Torch eagerly."""

    torch = _torch()
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.long)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [T] or [1, T], got {tuple(value.shape)}")
    return value.long()


def build_answer_only_labels(
    full_input_ids: Any,
    generation_prompt_input_ids: Any,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Any:
    """Mask the exact chat-template prompt prefix of an assistant completion.

    ``generation_prompt_input_ids`` must be the same user conversation rendered
    with ``add_generation_prompt=True``.  Requiring an exact token prefix is
    more robust than searching for a model-specific assistant marker and fails
    loudly if a processor changes its chat-template semantics.
    """

    torch = _torch()
    full = _one_dimensional_ids(full_input_ids, name="full_input_ids")
    prefix = _one_dimensional_ids(
        generation_prompt_input_ids, name="generation_prompt_input_ids"
    )
    if prefix.numel() >= full.numel():
        raise ValueError(
            "The rendered assistant completion must contain at least one token "
            "after its generation prompt"
        )
    if not torch.equal(full[: prefix.numel()].cpu(), prefix.cpu()):
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(full[: prefix.numel()].tolist(), prefix.tolist(), strict=True)
                )
                if actual != expected
            ),
            None,
        )
        raise ValueError(
            "The assistant generation prompt is not an exact prefix of the "
            f"rendered training conversation (first mismatch at token {mismatch})"
        )
    labels = full.clone()
    labels[: prefix.numel()] = int(ignore_index)
    if not labels.ne(ignore_index).any():
        raise ValueError("The rendered assistant response has no supervised token")
    return labels


def _default_image_loader(source: ImageSource) -> Any:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "Opening training images requires Pillow; install adaloop-qwen[gpu]"
        ) from exc
    # The paper's training passed the image path to the Qwen processor, whose
    # transformers load_image applies the EXIF orientation before converting
    # to RGB.  Doing the same keeps rotated photos' pixels and grid identical.
    with Image.open(io.BytesIO(source.read_bytes())) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class QwenImageQADataset:
    """Lazy map-style dataset using a checkpoint's native chat processor.

    ``constant_condition`` replaces every row's AdaLN condition text while the
    chat turn keeps the row's own question; without it each row conditions
    on its real question.  Library callers training or evaluating a
    constant-question recipe must pass the recipe's
    ``conditioner.constant_text``, as ``adaloop-qwen-train`` does.
    """

    def __init__(
        self,
        examples: Sequence[ImageQAExample],
        processor: Any,
        *,
        include_condition: bool = True,
        condition_max_tokens: int = CONDITION_MAX_TOKENS,
        constant_condition: str | None = None,
        include_trajectory: bool = False,
        image_loader: Callable[[ImageSource], Any] | None = None,
    ) -> None:
        self.examples = tuple(examples)
        if not self.examples:
            raise ValueError("QwenImageQADataset needs at least one example")
        self.processor = processor
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise TypeError("The Qwen processor must expose .tokenizer")
        if not isinstance(include_condition, bool):
            raise TypeError("include_condition must be a bool")
        if not isinstance(include_trajectory, bool):
            raise TypeError("include_trajectory must be a bool")
        if (
            isinstance(condition_max_tokens, bool)
            or not isinstance(condition_max_tokens, int)
            or condition_max_tokens < 1
        ):
            raise ValueError("condition_max_tokens must be a positive integer")
        if image_loader is not None and not callable(image_loader):
            raise TypeError("image_loader must be callable or None")
        self.tokenizer = tokenizer
        self.include_condition = include_condition
        self.condition_max_tokens = condition_max_tokens
        self.constant_condition = constant_condition
        self.include_trajectory = include_trajectory
        self.image_loader = image_loader or _default_image_loader
        if self.constant_condition is not None and not self.include_condition:
            raise ValueError("constant_condition requires include_condition=True")
        if self.constant_condition is not None:
            if not isinstance(self.constant_condition, str):
                raise TypeError("constant_condition must be a string or None")
            if not self.constant_condition.strip():
                raise ValueError("constant_condition must contain non-whitespace text")

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _messages(
        example: ImageQAExample, image: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": example.question},
            ],
        }
        prompt = [user]
        full = [
            user,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example.answer}],
            },
        ]
        return prompt, full

    def __getitem__(self, index: int) -> dict[str, Any]:
        from .model.prompting import encode_question_condition, encode_variant_c_fragments

        example = self.examples[index]
        image = self.image_loader(example.image)
        prompt_messages, full_messages = self._messages(example, image)
        prompt_result = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_result = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        if "input_ids" not in prompt_result or "input_ids" not in full_result:
            raise ValueError("Qwen chat processor did not return input_ids")

        result = dict(full_result)
        result["input_ids"] = _one_dimensional_ids(
            full_result["input_ids"], name="input_ids"
        )
        result["labels"] = build_answer_only_labels(
            result["input_ids"], prompt_result["input_ids"]
        )
        # The collator reconstructs masks from true lengths.  Keeping a
        # processor-supplied singleton mask here would invite accidental
        # double batching.
        result.pop("attention_mask", None)

        fields: dict[str, list[int]] = {}
        if self.include_condition:
            fields["condition"] = encode_question_condition(
                self.tokenizer,
                self.constant_condition or example.question,
                max_tokens=self.condition_max_tokens,
            )
        if self.include_trajectory:
            fragments = encode_variant_c_fragments(
                self.tokenizer, example.question, example.answer
            )
            for name, ids in fragments.items():
                fields[f"trajectory_{name}"] = ids
        torch = _torch()
        for name, ids in fields.items():
            tensor = torch.tensor(ids, dtype=torch.long)
            result[f"{name}_input_ids"] = tensor
            result[f"{name}_attention_mask"] = torch.ones_like(tensor, dtype=torch.bool)
        result["sample_id"] = example.sample_id
        return result


def _pad_1d(
    sequences: Sequence[Any],
    *,
    padding_value: int | bool,
    padding_side: str,
) -> Any:
    normalized = [_one_dimensional_ids(item, name="sequence") for item in sequences]
    maximum = max(item.numel() for item in normalized)
    output = normalized[0].new_full((len(normalized), maximum), padding_value)
    for row, item in enumerate(normalized):
        if padding_side == "left":
            output[row, maximum - item.numel() :] = item
        else:
            output[row, : item.numel()] = item
    return output


@dataclass
class QwenDataCollator:
    """Pad text fields and concatenate Qwen's packed visual tensors.

    Length policy (as in the paper's collator): a row longer than
    ``max_length`` keeps its first ``max_length`` tokens and labels, and is
    rejected only when that leaves it no supervised answer token.  The image
    placeholders precede the answer in the chat layout, so a row that keeps
    an answer token keeps its whole image.  Variant B and C sequences are
    checked again after readout expansion by the training hooks, which
    raise where the paper's collator raised.  With ``readout_mode``
    set to the recipe's ``readout.mode``, Variant-B (``accumulated_images``)
    rows are never truncated: the paper's runs expanded them whole, and the Variant-B
    hook rejects a row whose terminal-depth readout exceeds ``max_length``,
    reporting that full length.
    """

    tokenizer: Any
    max_length: int | None = 4096
    include_condition: bool = True
    include_trajectory: bool = False
    readout_mode: str | None = None

    def __call__(self, instances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        torch = _torch()
        if not instances:
            raise ValueError("QwenDataCollator received an empty batch")
        max_length = self.max_length
        if max_length is not None and (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 1
        ):
            raise ValueError("max_length must be a positive integer or None")
        if self.readout_mode is not None and self.readout_mode not in _READOUT_MODES:
            raise ValueError(f"unsupported readout_mode: {self.readout_mode!r}")
        if self.readout_mode == "accumulated_images":
            # The paper's runs passed Variant-B rows untruncated to the image-prefix
            # expansion; the Variant-B hook checks their expanded length.
            max_length = None
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        if padding_side not in {"left", "right"}:
            raise ValueError("tokenizer.padding_side must be 'left' or 'right'")
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            raise ValueError("Qwen tokenizer must define pad_token_id")

        ids = [_one_dimensional_ids(item["input_ids"], name="input_ids") for item in instances]
        labels = [_one_dimensional_ids(item["labels"], name="labels") for item in instances]
        for row, (row_ids, row_labels) in enumerate(zip(ids, labels, strict=True)):
            if row_ids.shape != row_labels.shape:
                raise ValueError(f"Row {row} input_ids and labels must have equal length")
            if not row_labels.ne(IGNORE_INDEX).any():
                raise ValueError(f"Row {row} contains no supervised answer token")
        if max_length is not None:
            emptied = [
                f"{item.get('sample_id', row)}={row_ids.numel()}"
                for row, (item, row_ids, row_labels) in enumerate(
                    zip(instances, ids, labels, strict=True)
                )
                if not row_labels[:max_length].ne(IGNORE_INDEX).any()
            ]
            if emptied:
                raise ValueError(
                    f"max_length={max_length} truncation removes the entire "
                    f"supervised answer ({', '.join(emptied)})"
                )
            # The paper's collator sliced its right-padded batch with
            # [:, :max_length]; per row, that keeps the first max_length tokens.
            ids = [row_ids[:max_length] for row_ids in ids]
            labels = [row_labels[:max_length] for row_labels in labels]

        batch_ids = _pad_1d(ids, padding_value=int(pad_id), padding_side=padding_side)
        batch_labels = _pad_1d(
            labels, padding_value=IGNORE_INDEX, padding_side=padding_side
        )
        lengths = torch.tensor([item.numel() for item in ids], dtype=torch.long)
        positions = torch.arange(batch_ids.shape[1])[None, :]
        if padding_side == "left":
            attention_mask = positions >= batch_ids.shape[1] - lengths[:, None]
        else:
            attention_mask = positions < lengths[:, None]
        batch: dict[str, Any] = {
            "input_ids": batch_ids,
            "labels": batch_labels,
            "attention_mask": attention_mask,
            "sample_ids": [str(item.get("sample_id", row)) for row, item in enumerate(instances)],
        }

        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            present = [key in item for item in instances]
            if any(present) and not all(present):
                raise ValueError(f"Visual field {key!r} must be present on every row or none")
            if all(present):
                tensors = [item[key] for item in instances]
                try:
                    batch[key] = torch.cat(tensors, dim=0)
                except (RuntimeError, TypeError) as exc:
                    shapes = [getattr(tensor, "shape", None) for tensor in tensors]
                    raise ValueError(f"Could not concatenate {key} tensors {shapes}") from exc

        if self.include_condition:
            self._pad_component(
                batch,
                instances,
                "condition",
                pad_id=int(pad_id),
                padding_side=padding_side,
            )
        if self.include_trajectory:
            for component in ("question", "answer", "cue", "separator"):
                self._pad_component(
                    batch,
                    instances,
                    f"trajectory_{component}",
                    pad_id=int(pad_id),
                    padding_side=padding_side,
                )
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is None:
                raise ValueError("Variant C requires tokenizer.eos_token_id")
            suppressed: list[int] = []
            for token_id in getattr(self.tokenizer, "all_special_ids", ()):
                value = int(token_id)
                if value != int(eos_id) and value not in suppressed:
                    suppressed.append(value)
            batch["trajectory_suppressed_token_ids"] = torch.tensor(
                suppressed, dtype=torch.long
            )
        return batch

    @staticmethod
    def _pad_component(
        batch: dict[str, Any],
        instances: Sequence[Mapping[str, Any]],
        component: str,
        *,
        pad_id: int,
        padding_side: str,
    ) -> None:
        ids_key = f"{component}_input_ids"
        mask_key = f"{component}_attention_mask"
        try:
            ids = [item[ids_key] for item in instances]
            masks = [item[mask_key] for item in instances]
        except KeyError as exc:
            raise ValueError(f"Batch is missing required field {exc.args[0]!r}") from exc
        batch[ids_key] = _pad_1d(
            ids, padding_value=pad_id, padding_side=padding_side
        )
        batch[mask_key] = _pad_1d(
            masks, padding_value=False, padding_side=padding_side
        ).bool()
        if batch[ids_key].shape != batch[mask_key].shape:
            raise ValueError(f"{ids_key} and {mask_key} must have matching shapes")
        if (~batch[mask_key].any(dim=1)).any():
            raise ValueError(f"Every row of {component} must contain an active token")


__all__ = [
    "IGNORE_INDEX",
    "QwenDataCollator",
    "QwenImageQADataset",
    "build_answer_only_labels",
]
