"""Pure tensor language readouts for Qwen AdaLoop Variants A, B, and C.

This module intentionally has no Transformers dependency.  It constructs the
native Qwen multi-image token/grid layouts and delegates M-RoPE calculation to
a callback supplied by the Qwen model wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from ..data import IGNORE_INDEX
from .prompting import PaddingSide, TokenBatch, VariantCPromptFragments


class MRoPEPositionBuilder(Protocol):
    """Callback matching the useful subset of Qwen ``get_rope_index``."""

    def __call__(
        self,
        input_ids: torch.LongTensor,
        *,
        image_grid_thw: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]: ...


class PositionedReadout(Protocol):
    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    image_grid_thw: torch.LongTensor


@dataclass(frozen=True)
class MRoPEPositions:
    position_ids: torch.LongTensor
    rope_deltas: torch.LongTensor


@dataclass(frozen=True)
class AccumulatedImageReadout:
    """Variant-B ``I1 I2 ... It Q A`` language batch."""

    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    labels: torch.LongTensor | None
    source_image_grid_thw: torch.LongTensor
    image_grid_thw: torch.LongTensor
    source_depths: tuple[int, ...]
    target_depth: int
    image_token_counts: tuple[int, ...]
    active_lengths: tuple[int, ...]


@dataclass(frozen=True)
class PreparedVariantC:
    """Validated source image containers and static Variant-C fragments."""

    containers: tuple[torch.LongTensor, ...]
    questions: tuple[torch.LongTensor, ...]
    answers: tuple[torch.LongTensor, ...] | None
    cues: tuple[torch.LongTensor, ...]
    separators: tuple[torch.LongTensor, ...]
    source_image_grid_thw: torch.LongTensor
    image_token_counts: tuple[int, ...]
    batch_size: int
    input_dtype: torch.dtype
    attention_mask_dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class InterleavedDraftReadout:
    """Variant-C ``I1 Q cue D1 S ... It Q cue [A]`` language batch."""

    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    labels: torch.LongTensor | None
    source_image_grid_thw: torch.LongTensor
    image_grid_thw: torch.LongTensor
    target_depth: int
    image_token_counts: tuple[int, ...]
    active_lengths: tuple[int, ...]


def _validate_integer_matrix(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
        raise ValueError(f"{name} must have shape [batch, sequence], got {shape}")
    if value.dtype == torch.bool or value.is_floating_point():
        raise TypeError(f"{name} must use an integer dtype")


def _validate_ids_and_mask(
    name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> None:
    _validate_integer_matrix(f"{name}_input_ids", input_ids)
    if batch_size is not None and input_ids.shape[0] != batch_size:
        raise ValueError(
            f"{name}_input_ids batch size {input_ids.shape[0]} does not match "
            f"{batch_size}"
        )
    if device is not None and input_ids.device != device:
        raise ValueError(f"{name}_input_ids must be on {device}")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        shape = (
            None
            if not isinstance(attention_mask, torch.Tensor)
            else tuple(attention_mask.shape)
        )
        raise ValueError(
            f"{name}_attention_mask must match {tuple(input_ids.shape)}, got {shape}"
        )
    if attention_mask.device != input_ids.device:
        raise ValueError(f"{name} IDs and mask must share a device")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError(f"{name}_attention_mask must contain only zero and one")


def _validate_grid(image_grid_thw: torch.Tensor, *, rows: int | None = None) -> None:
    if not isinstance(image_grid_thw, torch.Tensor):
        raise TypeError("image_grid_thw must be a torch.Tensor")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(
            "image_grid_thw must have shape [num_images, 3], got "
            f"{tuple(image_grid_thw.shape)}"
        )
    if rows is not None and image_grid_thw.shape[0] != rows:
        raise ValueError(
            f"image_grid_thw must contain {rows} rows, got {image_grid_thw.shape[0]}"
        )
    if image_grid_thw.shape[0] < 1:
        raise ValueError("image_grid_thw must contain at least one image")
    if image_grid_thw.dtype == torch.bool or image_grid_thw.is_floating_point():
        raise TypeError("image_grid_thw must use an integer dtype")
    if torch.any(image_grid_thw <= 0):
        raise ValueError("Every temporal, height, and width grid dimension must be > 0")


def merged_image_token_counts(
    source_image_grid_thw: torch.Tensor, spatial_merge_size: int
) -> tuple[int, ...]:
    """Return Qwen's post-merger image-token count for every source image."""

    _validate_grid(source_image_grid_thw)
    if (
        isinstance(spatial_merge_size, bool)
        or not isinstance(spatial_merge_size, int)
        or spatial_merge_size < 1
    ):
        raise ValueError("spatial_merge_size must be a positive integer")
    counts: list[int] = []
    for index, values in enumerate(source_image_grid_thw.detach().cpu().tolist()):
        temporal, height, width = (int(value) for value in values)
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError(
                f"Grid row {index} {(temporal, height, width)} is not divisible by "
                f"spatial_merge_size={spatial_merge_size}"
            )
        counts.append(
            temporal
            * (height // spatial_merge_size)
            * (width // spatial_merge_size)
        )
    return tuple(counts)


def infer_padding_side(attention_mask: torch.Tensor) -> PaddingSide:
    """Infer one contiguous padding side for a whole batch."""

    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        shape = (
            None
            if not isinstance(attention_mask, torch.Tensor)
            else tuple(attention_mask.shape)
        )
        raise ValueError(f"attention_mask must have shape [B, T], got {shape}")
    saw_left = False
    saw_right = False
    for row in attention_mask.bool():
        active = torch.nonzero(row, as_tuple=False).flatten()
        if active.numel() == 0:
            raise ValueError("Every readout row must contain an active token")
        start, end = int(active[0]), int(active[-1])
        if not torch.all(row[start : end + 1]):
            raise ValueError("Readout attention masks must use contiguous padding")
        saw_left = saw_left or start > 0
        saw_right = saw_right or end + 1 < row.numel()
    if saw_left and saw_right:
        raise ValueError("A batch cannot mix left- and right-padded sequences")
    return "left" if saw_left else "right"


def _active_rows(
    name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    padding_side: PaddingSide,
    allow_empty: bool = False,
) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    width = input_ids.shape[1]
    for index, (ids, mask) in enumerate(
        zip(input_ids, attention_mask, strict=True)
    ):
        length = int(mask.sum().item())
        if length == 0:
            if allow_empty:
                rows.append(ids[:0])
                continue
            raise ValueError(f"{name} row {index} is empty")
        expected = torch.zeros_like(mask, dtype=torch.bool)
        if padding_side == "right":
            expected[:length] = True
            active = ids[:length]
        else:
            expected[width - length :] = True
            active = ids[width - length :]
        if not torch.equal(mask.bool(), expected):
            raise ValueError(
                f"{name}_attention_mask row {index} is not contiguous "
                f"{padding_side}-padded data"
            )
        rows.append(active)
    return rows


def _masked_active_rows(
    name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Extract contiguous active rows when the tokenizer side is ambiguous."""

    rows: list[torch.Tensor] = []
    for index, (ids, mask) in enumerate(
        zip(input_ids, attention_mask.bool(), strict=True)
    ):
        active = torch.nonzero(mask, as_tuple=False).flatten()
        if active.numel() == 0:
            raise ValueError(f"{name} row {index} is empty")
        start, end = int(active[0]), int(active[-1])
        if not torch.all(mask[start : end + 1]):
            raise ValueError(f"{name}_attention_mask row {index} is not contiguous")
        rows.append(ids[mask])
    return rows


def _find_image_containers(
    row: torch.Tensor,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    row_index: int,
) -> list[tuple[int, int, int]]:
    values = row.detach().cpu().tolist()
    containers: list[tuple[int, int, int]] = []
    cursor = 0
    while cursor < len(values):
        token_id = values[cursor]
        if token_id == vision_start_token_id:
            image_start = cursor + 1
            image_stop = image_start
            while image_stop < len(values) and values[image_stop] == image_token_id:
                image_stop += 1
            if image_stop == image_start:
                raise ValueError(
                    f"Sample {row_index} vision-start is not followed by image tokens"
                )
            if image_stop >= len(values) or values[image_stop] != vision_end_token_id:
                raise ValueError(f"Sample {row_index} has an incomplete image container")
            containers.append((cursor, image_stop + 1, image_stop - image_start))
            cursor = image_stop + 1
            continue
        if token_id == image_token_id:
            raise ValueError(f"Sample {row_index} has an image token outside a container")
        if token_id == vision_end_token_id:
            raise ValueError(f"Sample {row_index} has an unmatched vision-end token")
        cursor += 1
    if not containers:
        raise ValueError(f"Sample {row_index} contains no Qwen image container")
    for previous, current in zip(containers, containers[1:]):
        if previous[1] != current[0]:
            raise ValueError(
                f"Sample {row_index} image containers must be one adjacent block"
            )
    return containers


def _source_grids(
    image_grid_thw: torch.Tensor, depths: Sequence[int]
) -> torch.Tensor:
    batch_size = len(depths)
    if image_grid_thw.shape[0] == batch_size:
        return image_grid_thw.clone()
    if image_grid_thw.shape[0] != sum(depths):
        raise ValueError(
            "image_grid_thw must contain one source row per sample or one row per "
            f"existing container; got {image_grid_thw.shape[0]} rows"
        )
    source_rows: list[torch.Tensor] = []
    offset = 0
    for sample_index, depth in enumerate(depths):
        group = image_grid_thw[offset : offset + depth]
        offset += depth
        if not torch.all(group == group[0]):
            raise ValueError(
                f"Repeated grid rows for sample {sample_index} are inconsistent"
            )
        source_rows.append(group[0])
    return torch.stack(source_rows)


def _padded_output(
    rows: Sequence[torch.Tensor],
    *,
    mask_dtype: torch.dtype,
    pad_token_id: int,
    padding_side: PaddingSide,
    minimum_width: int = 0,
    label_rows: Sequence[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    width = max(minimum_width, *(int(row.numel()) for row in rows))
    reference = rows[0]
    output_ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=reference.dtype,
        device=reference.device,
    )
    output_mask = torch.zeros(
        (len(rows), width), dtype=mask_dtype, device=reference.device
    )
    label_reference = label_rows[0] if label_rows is not None else None
    output_labels = (
        torch.full(
            (len(rows), width),
            IGNORE_INDEX,
            dtype=label_reference.dtype,
            device=label_reference.device,
        )
        if label_reference is not None
        else None
    )
    for index, row in enumerate(rows):
        length = int(row.numel())
        destination = slice(0, length) if padding_side == "right" else slice(-length, None)
        output_ids[index, destination] = row
        output_mask[index, destination] = 1
        if output_labels is not None and label_rows is not None:
            output_labels[index, destination] = label_rows[index]
    return output_ids, output_mask, output_labels


def build_accumulated_image_readout(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.LongTensor,
    *,
    target_depth: int,
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    spatial_merge_size: int,
    labels: torch.LongTensor | None = None,
    pad_token_id: int = 0,
    padding_side: PaddingSide | None = None,
    max_length: int | None = None,
    preserve_width: bool = True,
) -> AccumulatedImageReadout:
    """Resize Qwen image containers to Variant-B depth.

    Existing containers are retained from the left.  Missing containers copy
    the first complete source container and get ignored labels.  Grid rows are
    emitted in sample-major order, matching :func:`interleave_step_features`.
    """

    _validate_ids_and_mask("source", input_ids, attention_mask)
    _validate_grid(image_grid_thw)
    if labels is not None:
        _validate_integer_matrix("labels", labels)
        if labels.shape != input_ids.shape or labels.device != input_ids.device:
            raise ValueError("labels must match input_ids shape and device")
    if isinstance(target_depth, bool) or not isinstance(target_depth, int) or target_depth < 1:
        raise ValueError("target_depth must be a positive integer")
    if max_length is not None and (
        isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1
    ):
        raise ValueError("max_length must be a positive integer or None")
    if not isinstance(preserve_width, bool):
        raise TypeError("preserve_width must be a bool")
    side = infer_padding_side(attention_mask) if padding_side is None else padding_side
    if side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")

    source_rows = _active_rows(
        "source", input_ids, attention_mask, padding_side=side
    )
    label_source_rows = (
        _active_rows("labels", labels, attention_mask, padding_side=side)
        if labels is not None
        else None
    )
    containers = [
        _find_image_containers(
            row,
            image_token_id=int(image_token_id),
            vision_start_token_id=int(vision_start_token_id),
            vision_end_token_id=int(vision_end_token_id),
            row_index=index,
        )
        for index, row in enumerate(source_rows)
    ]
    source_depths = tuple(len(value) for value in containers)
    source_grids = _source_grids(image_grid_thw, source_depths)
    counts = merged_image_token_counts(source_grids, spatial_merge_size)

    rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] | None = [] if labels is not None else None
    for sample, (source, spans, expected_count) in enumerate(
        zip(source_rows, containers, counts, strict=True)
    ):
        observed = tuple(span[2] for span in spans)
        if any(count != expected_count for count in observed):
            raise ValueError(
                f"Sample {sample} image token counts {observed} do not match grid "
                f"count {expected_count}"
            )
        block_start, block_stop = spans[0][0], spans[-1][1]
        kept = min(len(spans), target_depth)
        parts = [source[:block_start]]
        parts.extend(source[start:stop] for start, stop, _ in spans[:kept])
        if target_depth > len(spans):
            template = source[spans[0][0] : spans[0][1]]
            parts.extend(template for _ in range(target_depth - len(spans)))
        parts.append(source[block_stop:])
        rows.append(torch.cat(parts))

        if label_rows is not None and label_source_rows is not None:
            source_labels = label_source_rows[sample]
            pieces = [source_labels[:block_start]]
            pieces.extend(source_labels[start:stop] for start, stop, _ in spans[:kept])
            if target_depth > len(spans):
                inserted_length = spans[0][1] - spans[0][0]
                pieces.extend(
                    torch.full(
                        (inserted_length,),
                        IGNORE_INDEX,
                        dtype=source_labels.dtype,
                        device=source_labels.device,
                    )
                    for _ in range(target_depth - len(spans))
                )
            pieces.append(source_labels[block_stop:])
            label_rows.append(torch.cat(pieces))

    active_lengths = tuple(int(row.numel()) for row in rows)
    minimum_width = int(input_ids.shape[1]) if preserve_width else 0
    output_width = max(minimum_width, *active_lengths)
    if max_length is not None and output_width > max_length:
        offending = [
            f"sample {index}: {length}"
            for index, length in enumerate(active_lengths)
            if length > max_length
        ]
        detail = ", ".join(offending) if offending else f"padded width: {output_width}"
        raise ValueError(
            f"Accumulated image readout exceeds max_length={max_length} ({detail})"
        )
    output_ids, output_mask, output_labels = _padded_output(
        rows,
        mask_dtype=attention_mask.dtype,
        pad_token_id=pad_token_id,
        padding_side=side,
        minimum_width=minimum_width,
        label_rows=label_rows,
    )
    return AccumulatedImageReadout(
        input_ids=output_ids,
        attention_mask=output_mask,
        labels=output_labels,
        source_image_grid_thw=source_grids,
        image_grid_thw=source_grids.repeat_interleave(target_depth, dim=0),
        source_depths=source_depths,
        target_depth=target_depth,
        image_token_counts=counts,
        active_lengths=active_lengths,
    )


def interleave_step_features(
    step_features: Sequence[torch.Tensor],
    source_image_grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Convert step-major connector features to sample-major language order."""

    if isinstance(step_features, torch.Tensor) or not isinstance(step_features, Sequence):
        raise TypeError("step_features must be a non-empty sequence of tensors")
    if not step_features:
        raise ValueError("step_features must contain a recurrent readout")
    counts = merged_image_token_counts(source_image_grid_thw, spatial_merge_size)
    total_tokens = sum(counts)
    reference = step_features[0]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
        raise ValueError("Every step feature tensor must have shape [tokens, hidden]")
    for index, features in enumerate(step_features):
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError(f"Step {index} features must have shape [tokens, hidden]")
        if features.shape != (total_tokens, reference.shape[1]):
            raise ValueError(
                f"Step {index} shape {tuple(features.shape)} does not match "
                f"({total_tokens}, {reference.shape[1]})"
            )
        if features.device != reference.device or features.dtype != reference.dtype:
            raise ValueError("All step features must share device and dtype")
    chunks = [features.split(counts, dim=0) for features in step_features]
    return torch.cat(
        [
            chunks[step][sample]
            for sample in range(len(counts))
            for step in range(len(step_features))
        ]
    )


def current_step_feature(
    step_features: Sequence[torch.Tensor], *, depth: int
) -> torch.Tensor:
    """Select Variant A's one current readout using a 1-based depth."""

    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if depth > len(step_features):
        raise ValueError(f"depth {depth} exceeds {len(step_features)} available readouts")
    return step_features[depth - 1]


def prepare_interleaved_draft_readout(
    source_input_ids: torch.LongTensor,
    source_attention_mask: torch.Tensor,
    source_image_grid_thw: torch.LongTensor,
    fragments: VariantCPromptFragments,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    spatial_merge_size: int,
    source_padding_side: PaddingSide | None = None,
    fragment_padding_side: PaddingSide | None = None,
) -> PreparedVariantC:
    """Validate and cache the static parts of the public Variant-C layout."""

    _validate_ids_and_mask("source", source_input_ids, source_attention_mask)
    batch_size = int(source_input_ids.shape[0])
    if batch_size < 1:
        raise ValueError("source_input_ids must contain at least one sample")
    _validate_grid(source_image_grid_thw, rows=batch_size)
    if source_image_grid_thw.device != source_input_ids.device:
        raise ValueError("source image grids and IDs must share a device")
    if not isinstance(fragments, VariantCPromptFragments):
        raise TypeError("fragments must be VariantCPromptFragments")

    named: tuple[tuple[str, TokenBatch | None], ...] = (
        ("question", fragments.question),
        ("cue", fragments.cue),
        ("separator", fragments.separator),
        ("answer", fragments.answer),
    )
    for name, batch in named:
        if batch is None:
            continue
        _validate_ids_and_mask(
            name,
            batch.input_ids,
            batch.attention_mask,
            batch_size=batch_size,
            device=source_input_ids.device,
        )

    source_side = (
        infer_padding_side(source_attention_mask)
        if source_padding_side is None
        else source_padding_side
    )
    if source_side not in {"left", "right"} or (
        fragment_padding_side is not None
        and fragment_padding_side not in {"left", "right"}
    ):
        raise ValueError("padding sides must be 'left' or 'right'")
    source_rows = _active_rows(
        "source", source_input_ids, source_attention_mask, padding_side=source_side
    )
    all_spans = [
        _find_image_containers(
            row,
            image_token_id=int(image_token_id),
            vision_start_token_id=int(vision_start_token_id),
            vision_end_token_id=int(vision_end_token_id),
            row_index=index,
        )
        for index, row in enumerate(source_rows)
    ]
    if any(len(spans) != 1 for spans in all_spans):
        observed = tuple(len(spans) for spans in all_spans)
        raise ValueError(
            "Variant-C source rows must contain exactly one image container; "
            f"got {observed}"
        )
    containers = tuple(
        row[spans[0][0] : spans[0][1]]
        for row, spans in zip(source_rows, all_spans, strict=True)
    )
    counts = merged_image_token_counts(source_image_grid_thw, spatial_merge_size)
    for sample, (container, count) in enumerate(zip(containers, counts, strict=True)):
        observed = int((container == int(image_token_id)).sum().item())
        if observed != count:
            raise ValueError(
                f"Sample {sample} image container has {observed} placeholders; "
                f"its grid requires {count}"
            )

    def fragment_rows(name: str, batch: TokenBatch) -> tuple[torch.Tensor, ...]:
        if fragment_padding_side is None:
            return tuple(
                _masked_active_rows(name, batch.input_ids, batch.attention_mask)
            )
        return tuple(
            _active_rows(
                name,
                batch.input_ids,
                batch.attention_mask,
                padding_side=fragment_padding_side,
            )
        )

    return PreparedVariantC(
        containers=containers,
        questions=fragment_rows("question", fragments.question),
        answers=(
            None
            if fragments.answer is None
            else fragment_rows("answer", fragments.answer)
        ),
        cues=fragment_rows("cue", fragments.cue),
        separators=fragment_rows("separator", fragments.separator),
        source_image_grid_thw=source_image_grid_thw,
        image_token_counts=counts,
        batch_size=batch_size,
        input_dtype=source_input_ids.dtype,
        attention_mask_dtype=source_attention_mask.dtype,
        device=source_input_ids.device,
    )


def build_interleaved_draft_readout(
    prepared: PreparedVariantC,
    draft_input_ids: Sequence[torch.LongTensor],
    draft_attention_masks: Sequence[torch.Tensor],
    *,
    target_depth: int,
    pad_token_id: int = 0,
    draft_padding_side: PaddingSide = "right",
    padding_side: PaddingSide = "right",
    max_length: int | None = None,
    include_answer: bool = True,
) -> InterleavedDraftReadout:
    """Assemble the only released Variant-C layout (``each_round``)."""

    if not isinstance(prepared, PreparedVariantC):
        raise TypeError("prepared must be PreparedVariantC")
    if isinstance(target_depth, bool) or not isinstance(target_depth, int) or target_depth < 1:
        raise ValueError("target_depth must be a positive integer")
    if draft_padding_side not in {"left", "right"} or padding_side not in {
        "left",
        "right",
    }:
        raise ValueError("padding sides must be 'left' or 'right'")
    if max_length is not None and (
        isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1
    ):
        raise ValueError("max_length must be a positive integer or None")
    if len(draft_input_ids) != target_depth - 1:
        raise ValueError(
            f"target_depth={target_depth} requires {target_depth - 1} drafts, "
            f"got {len(draft_input_ids)}"
        )
    if len(draft_attention_masks) != len(draft_input_ids):
        raise ValueError("draft_attention_masks must provide one mask per draft")
    if include_answer and prepared.answers is None:
        raise ValueError("include_answer=True requires answer fragments")

    draft_rows: list[list[torch.Tensor]] = []
    for index, (ids, mask) in enumerate(
        zip(draft_input_ids, draft_attention_masks, strict=True), start=1
    ):
        _validate_ids_and_mask(
            f"draft_{index}",
            ids,
            mask,
            batch_size=prepared.batch_size,
            device=prepared.device,
        )
        draft_rows.append(
            _active_rows(
                f"draft_{index}",
                ids,
                mask,
                padding_side=draft_padding_side,
                allow_empty=True,
            )
        )

    rows: list[torch.Tensor] = []
    labels: list[torch.Tensor] | None = [] if include_answer else None
    for sample in range(prepared.batch_size):
        parts: list[torch.Tensor] = []
        for round_index in range(target_depth):
            parts.extend(
                (
                    prepared.containers[sample],
                    prepared.questions[sample],
                    prepared.cues[sample],
                )
            )
            if round_index < target_depth - 1:
                parts.extend(
                    (
                        draft_rows[round_index][sample],
                        prepared.separators[sample],
                    )
                )
        prompt = torch.cat(parts)
        if not include_answer:
            rows.append(prompt)
            continue
        if prepared.answers is None:  # narrowed above
            raise RuntimeError("answer fragments were not initialized")
        answer = prepared.answers[sample]
        rows.append(torch.cat((prompt, answer)))
        if labels is None:  # narrowed above
            raise RuntimeError("labels were not initialized")
        labels.append(
            torch.cat(
                (
                    torch.full(
                        (prompt.numel(),),
                        IGNORE_INDEX,
                        dtype=prepared.input_dtype,
                        device=prepared.device,
                    ),
                    answer.to(dtype=prepared.input_dtype),
                )
            )
        )

    active_lengths = tuple(int(row.numel()) for row in rows)
    if max_length is not None:
        offending = [
            f"sample {index}: {length}"
            for index, length in enumerate(active_lengths)
            if length > max_length
        ]
        if offending:
            raise ValueError(
                f"Variant-C trajectory exceeds max_length={max_length} "
                f"({', '.join(offending)})"
            )
    output_ids, output_mask, output_labels = _padded_output(
        rows,
        mask_dtype=prepared.attention_mask_dtype,
        pad_token_id=pad_token_id,
        padding_side=padding_side,
        label_rows=labels,
    )
    return InterleavedDraftReadout(
        input_ids=output_ids,
        attention_mask=output_mask,
        labels=output_labels,
        source_image_grid_thw=prepared.source_image_grid_thw.clone(),
        image_grid_thw=prepared.source_image_grid_thw.repeat_interleave(
            target_depth, dim=0
        ),
        target_depth=target_depth,
        image_token_counts=prepared.image_token_counts,
        active_lengths=active_lengths,
    )


def build_mrope_positions(
    readout: PositionedReadout,
    position_builder: MRoPEPositionBuilder,
) -> MRoPEPositions:
    """Construct and validate Qwen multimodal positions for a readout batch."""

    if not callable(position_builder):
        raise TypeError("position_builder must be callable")
    position_ids, rope_deltas = position_builder(
        readout.input_ids,
        image_grid_thw=readout.image_grid_thw,
        attention_mask=readout.attention_mask,
    )
    expected = (3, *readout.input_ids.shape)
    if not isinstance(position_ids, torch.Tensor) or tuple(position_ids.shape) != expected:
        shape = None if not isinstance(position_ids, torch.Tensor) else tuple(position_ids.shape)
        raise ValueError(f"M-RoPE position_ids must have shape {expected}, got {shape}")
    if position_ids.dtype == torch.bool or position_ids.is_floating_point():
        raise TypeError("M-RoPE position_ids must use an integer dtype")
    if position_ids.device != readout.input_ids.device:
        raise ValueError("M-RoPE position_ids must share the readout device")
    if not isinstance(rope_deltas, torch.Tensor):
        raise TypeError("M-RoPE builder must return rope_deltas as a tensor")
    if rope_deltas.ndim != 2 or rope_deltas.shape[0] != readout.input_ids.shape[0]:
        raise ValueError(
            "M-RoPE rope_deltas must have shape [batch, n], got "
            f"{tuple(rope_deltas.shape)}"
        )
    if rope_deltas.device != readout.input_ids.device:
        raise ValueError("M-RoPE rope_deltas must share the readout device")
    return MRoPEPositions(position_ids=position_ids, rope_deltas=rope_deltas)


__all__ = [
    "AccumulatedImageReadout",
    "InterleavedDraftReadout",
    "MRoPEPositionBuilder",
    "MRoPEPositions",
    "PreparedVariantC",
    "build_accumulated_image_readout",
    "build_interleaved_draft_readout",
    "build_mrope_positions",
    "current_step_feature",
    "infer_padding_side",
    "interleave_step_features",
    "merged_image_token_counts",
    "prepare_interleaved_draft_readout",
]
