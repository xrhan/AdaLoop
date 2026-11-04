"""Training and evaluation examples: VisRO, read with the ``visro`` package, plus a replay set.

A training set is

1. the ``train`` split of the VisRO ``tasks`` (default ``core6``, the paper's six tasks),
   optionally restricted to some difficulty levels, in dataset order: tasks in ``visro`` task
   order, shards in name order, records in shard order; followed by
2. optionally, the rows of one replay JSONL file, in file order.

Every row has the same weight; each backend's seeded sampler shuffles the pooled list once per
epoch. The paper mix is the six tasks' 600,000 training rows plus 100,000 replay rows.

The dataset root is a downloaded VisRO directory (it holds ``tasks/<category>/<task>/<split>/``).
Its shards are read in place, either the published tar files or extracted shard directories, as
``visro.dataset`` finds them. A shard holds, per example, its record ``<key>.json`` followed by
its image ``<key>.png``; an image's bytes are checked against its record's ``image_sha256``
whenever the image is read.

A replay row is a JSON object with exactly ``image``, ``question`` and ``answer`` and an optional
unique ``id``. A relative ``image`` resolves against ``image_root`` (default: the directory of
the JSONL file). Lines end only at ``\\n``, ``\\r\\n`` or ``\\r``; blank lines are skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from visro.dataset import find_shards, read_shard, resolve_tasks

_REPLAY_FIELDS = frozenset({"id", "image", "question", "answer"})


@dataclass(frozen=True)
class ImageSource:
    """Where one image's bytes are: a whole file, or ``size`` bytes at ``offset`` of a tar shard.

    ``sha256``, when set, is the expected digest of the bytes; :meth:`read_bytes` checks it.
    """

    path: Path
    member: str | None = None
    offset: int | None = None
    size: int | None = None
    sha256: str | None = None

    def __str__(self) -> str:
        return f"{self.path}:{self.member}" if self.member else str(self.path)

    def read_bytes(self) -> bytes:
        try:
            with open(self.path, "rb") as handle:
                if self.offset is None:
                    data = handle.read()
                else:
                    handle.seek(self.offset)
                    data = handle.read(self.size)
        except OSError as exc:
            raise ValueError(f"Could not read image {self}: {exc}") from exc
        if self.size is not None and len(data) != self.size:
            raise ValueError(f"Image {self} is truncated: {len(data)} of {self.size} bytes")
        if self.sha256 is not None and hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"Image {self} does not match its image_sha256")
        return data


@dataclass(frozen=True)
class ImageQAExample:
    """One training or evaluation example.

    ``task`` and ``difficulty`` are the VisRO labels (``None`` for replay rows). ``record`` is the
    full VisRO record when it was kept (evaluation, which scores against it).
    """

    sample_id: str
    question: str
    answer: str
    image: ImageSource
    task: str | None = None
    difficulty: str | None = None
    record: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)


def iter_jsonl_lines(jsonl_path: str | Path) -> Iterator[str]:
    """Yield every physical line of a UTF-8 JSONL file, blank lines as ``""``.

    Lines end only at ``\\n``, ``\\r\\n``, or a lone ``\\r``.  ``str.splitlines()`` would also
    split at U+2028, U+2029, or NEL, which JSON strings may contain.  A line holding only
    (Unicode) whitespace is blank.
    """

    with Path(jsonl_path).open(encoding="utf-8") as handle:
        for line in handle:
            yield line.removesuffix("\n") if line.strip() else ""


def _tar_members(shard: Path) -> dict[str, tuple[int, int]]:
    """``{member name: (data offset, size)}`` of the regular files of a tar shard."""

    with tarfile.open(shard, "r:") as archive:
        return {m.name: (m.offset_data, m.size) for m in archive if m.isfile()}


def _record_text(record: Mapping[str, Any], name: str, where: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {name} must be a non-empty string")
    return value


def _difficulty_levels(difficulty: str | Iterable[str] | None) -> Iterable[str] | None:
    """The levels of a comma-separated ``difficulty`` string; other values unchanged."""

    if isinstance(difficulty, str):
        return [level for level in difficulty.split(",") if level.strip()]
    return difficulty


def load_visro_examples(
    root: str | Path,
    tasks: str | Iterable[str] | None = "core6",
    split: str = "train",
    difficulty: str | Iterable[str] | None = None,
    *,
    per_cell: int = 0,
    keep_records: bool = False,
) -> tuple[ImageQAExample, ...]:
    """Examples of ``tasks`` x ``split`` (x ``difficulty``) under the dataset ``root``.

    ``tasks`` takes task names, ``core6`` or ``all`` (a list or a comma-separated string).
    ``per_cell`` keeps at most that many examples per (task, difficulty) cell, the first ones in
    dataset order (0 keeps all). ``keep_records`` attaches each full record (for scoring).
    """

    if isinstance(per_cell, bool) or not isinstance(per_cell, int) or per_cell < 0:
        raise ValueError("per_cell must be a non-negative integer")
    root = Path(root).expanduser()
    difficulty = _difficulty_levels(difficulty)
    examples: list[ImageQAExample] = []
    cells: Counter = Counter()
    for shard in find_shards(root, tasks, split):
        members = _tar_members(shard) if shard.is_file() else None
        for item in read_shard(shard, images=False, difficulty=difficulty):
            record = item.record
            where = f"{shard}: record {record.get('id')!r}"
            cell = (record.get("task"), record.get("difficulty"))
            if per_cell and cells[cell] >= per_cell:
                continue
            cells[cell] += 1
            name = _record_text(record, "image", where)
            if members is None:
                image = ImageSource(shard / name, sha256=record.get("image_sha256"))
            else:
                if name not in members:
                    raise ValueError(f"{where}: the shard has no member {name}")
                offset, size = members[name]
                image = ImageSource(shard, name, offset, size, record.get("image_sha256"))
            examples.append(
                ImageQAExample(
                    sample_id=_record_text(record, "id", where),
                    question=_record_text(record, "question", where),
                    answer=_record_text(record, "answer", where),
                    image=image,
                    task=record.get("task"),
                    difficulty=record.get("difficulty"),
                    record=record if keep_records else None,
                )
            )
    if not examples:
        raise ValueError(f"No VisRO {split} examples for tasks {tasks!r} under {root}")
    _require_unique(examples)
    return tuple(examples)


def load_replay_examples(
    jsonl_path: str | Path,
    *,
    image_root: str | Path | None = None,
) -> tuple[ImageQAExample, ...]:
    """Read and strictly validate a replay JSONL file (see the module docstring)."""

    source = Path(jsonl_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve() if image_root is not None else source.parent
    try:
        lines = tuple(iter_jsonl_lines(source))
    except OSError as exc:
        raise ValueError(f"Could not read replay JSONL {source}: {exc}") from exc

    examples: list[ImageQAExample] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        context = f"{source}:{line_number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context} is not valid JSON: {exc.msg}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context} must contain a JSON object")
        unknown = set(raw) - _REPLAY_FIELDS
        missing = {"image", "question", "answer"} - set(raw)
        if unknown:
            raise ValueError(f"{context} contains unknown field(s): {', '.join(sorted(unknown))}")
        if missing:
            raise ValueError(f"{context} is missing field(s): {', '.join(sorted(missing))}")
        sample_id = raw.get("id", f"line-{line_number}")
        if not isinstance(sample_id, (str, int)) or isinstance(sample_id, bool):
            raise ValueError(f"{context}.id must be a non-empty string or integer")
        sample_id = str(sample_id).strip()
        if not sample_id:
            raise ValueError(f"{context}.id must be non-empty")
        image_path = Path(os.path.expanduser(_record_text(raw, "image", context)))
        if not image_path.is_absolute():
            image_path = root / image_path
        examples.append(
            ImageQAExample(
                sample_id=sample_id,
                question=_record_text(raw, "question", context),
                answer=_record_text(raw, "answer", context),
                image=ImageSource(image_path),
            )
        )
    if not examples:
        raise ValueError(f"Replay JSONL {source} contains no examples")
    _require_unique(examples)
    return tuple(examples)


def _require_unique(examples: Sequence[ImageQAExample]) -> None:
    seen: set[str] = set()
    for example in examples:
        if example.sample_id in seen:
            raise ValueError(f"Duplicate sample id {example.sample_id!r}")
        seen.add(example.sample_id)


@dataclass(frozen=True)
class TrainingSet:
    """The pooled training examples and what they were selected from (no local paths)."""

    examples: tuple[ImageQAExample, ...]
    tasks: tuple[str, ...]
    difficulty: tuple[str, ...] | None
    visro_count: int
    replay_count: int

    def __len__(self) -> int:
        return len(self.examples)

    def describe(self) -> dict[str, Any]:
        """The selection and the per-task counts, for logs and run contracts."""

        per_task = Counter(example.task for example in self.examples[: self.visro_count])
        return {
            "visro": {
                "split": "train",
                "tasks": list(self.tasks),
                "difficulty": None if self.difficulty is None else list(self.difficulty),
                "sample_count": self.visro_count,
                "per_task": {task: per_task[task] for task in self.tasks},
            },
            "replay_sample_count": self.replay_count,
            "sample_count": len(self.examples),
        }


def load_training_set(
    visro_root: str | Path,
    *,
    tasks: str | Iterable[str] | None = "core6",
    difficulty: str | Iterable[str] | None = None,
    replay_jsonl: str | Path | None = None,
    replay_image_root: str | Path | None = None,
) -> TrainingSet:
    """VisRO training examples followed by the optional replay rows (see the module docstring)."""

    if replay_jsonl is None and replay_image_root is not None:
        raise ValueError("replay_image_root requires replay_jsonl")
    difficulty = _difficulty_levels(difficulty)
    visro = load_visro_examples(visro_root, tasks, "train", difficulty)
    replay = (
        load_replay_examples(replay_jsonl, image_root=replay_image_root)
        if replay_jsonl is not None
        else ()
    )
    examples = visro + replay
    _require_unique(examples)
    return TrainingSet(
        examples=examples,
        tasks=resolve_tasks(tasks),
        difficulty=None if difficulty is None else tuple(difficulty),
        visro_count=len(visro),
        replay_count=len(replay),
    )


def add_training_data_arguments(parser: Any) -> None:
    """The training-data options of the Qwen and PaliGemma training CLIs."""

    parser.add_argument(
        "--visro-root", required=True, help="Downloaded VisRO dataset (holds tasks/)"
    )
    parser.add_argument(
        "--tasks",
        default="core6",
        help="Comma-separated VisRO tasks, core6 or all (default: core6, the paper's six)",
    )
    parser.add_argument(
        "--difficulty",
        help="Comma-separated difficulty levels to train on (default: all)",
    )
    parser.add_argument(
        "--replay-jsonl",
        help="Optional replay JSONL ({id, image, question, answer}) pooled after VisRO",
    )
    parser.add_argument(
        "--replay-image-root",
        help="Base directory of relative replay image paths (default: the JSONL's directory)",
    )


def training_set_from_args(args: Any) -> TrainingSet:
    """:func:`load_training_set` for the options of :func:`add_training_data_arguments`."""

    try:
        return load_training_set(
            args.visro_root,
            tasks=args.tasks,
            difficulty=args.difficulty,
            replay_jsonl=args.replay_jsonl,
            replay_image_root=args.replay_image_root,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not load the training data: {exc}") from exc


# Threads that hash images without a declared digest (replay rows).  Reading many small files
# from network storage is bound by per-file round trips rather than bandwidth, so 32 threads hash
# a large replay set about 25 times faster than one.
_FINGERPRINT_WORKERS = 32
# Images submitted at a time, which bounds the queued work.
_FINGERPRINT_BATCH = 4096


def _file_digest(source: ImageSource) -> bytes | ValueError:
    try:
        return hashlib.sha256(source.read_bytes()).digest()
    except ValueError as exc:
        return exc


def image_content_fingerprint(examples: Sequence[ImageQAExample]) -> str:
    """A path-independent digest of every sample-to-image association.

    Signs the ordered ``sample_id -> SHA-256(image bytes)`` associations. A VisRO image
    contributes its record's ``image_sha256`` (checked against the bytes whenever the image is
    read); any other image is hashed once, with concurrent reads. A missing or unreadable image
    raises ``ValueError``.
    """

    if not examples:
        raise ValueError("Cannot fingerprint an empty dataset")
    digests: dict[ImageSource, bytes] = {}
    pending: list[ImageSource] = []
    for example in examples:
        source = example.image
        if source in digests:
            continue
        if source.sha256 is not None:
            digests[source] = bytes.fromhex(source.sha256)
        else:
            digests[source] = b""
            pending.append(source)
    with ThreadPoolExecutor(max_workers=_FINGERPRINT_WORKERS) as pool:
        for start in range(0, len(pending), _FINGERPRINT_BATCH):
            batch = pending[start : start + _FINGERPRINT_BATCH]
            # map() yields in submission order, so the error raised is the first failing image
            # in sample order, as a serial pass reports.
            for source, outcome in zip(batch, pool.map(_file_digest, batch)):
                if isinstance(outcome, ValueError):
                    raise outcome
                digests[source] = outcome
    result = hashlib.sha256()
    for example in examples:
        sample_bytes = example.sample_id.encode("utf-8")
        result.update(len(sample_bytes).to_bytes(8, "big"))
        result.update(sample_bytes)
        result.update(digests[example.image])
    return result.hexdigest()


def text_content_fingerprint(examples: Sequence[ImageQAExample]) -> str:
    """A digest of the ordered ``(sample_id, question, answer)`` triples."""

    result = hashlib.sha256()
    for example in examples:
        for text in (example.sample_id, example.question, example.answer):
            encoded = text.encode("utf-8")
            result.update(len(encoded).to_bytes(8, "big"))
            result.update(encoded)
    return result.hexdigest()


__all__ = [
    "ImageQAExample",
    "ImageSource",
    "TrainingSet",
    "add_training_data_arguments",
    "image_content_fingerprint",
    "iter_jsonl_lines",
    "load_replay_examples",
    "load_training_set",
    "load_visro_examples",
    "text_content_fingerprint",
    "training_set_from_args",
]
