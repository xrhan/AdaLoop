"""VisRO evaluation shared by the backends.

An evaluation answers every item of the ``eval`` and/or ``ood`` split of the chosen tasks at the
fixed depths ``1..D``, scores each answer with ``visro.scoring.score_record`` and applies
answer-stability stopping with ``p = 1`` to the fixed-depth answers: an item stops at the first
depth ``t`` whose answer equals the answer at ``t - 1`` after case and whitespace normalization,
else at ``T``, and keeps that depth's answer. The depth-``t`` answer is the answer an online loop
gives after ``t`` steps, so this equals stopping online. Before anything is generated, every
item's reference answer must score as correct (:func:`check_gold`).

Prompts. The condition of a question-conditioned model is always the bare question. The
language-model text depends on ``prompt_style``:

* ``instruction`` (the paper's Qwen and PaliGemma evaluation): the question followed by an
  answer-format instruction chosen by task and answer type (:func:`answer_format_instruction`);
* ``question`` (the paper's LLaVA evaluation): the bare question.

Outputs of :func:`write_split` and :func:`write_report` under the output directory::

    <split>/depth<d>.jsonl   one row per item: id, task, difficulty, depth, prediction,
                             correct, parsed, gold, rule
    <split>/p1_T<T>.jsonl    the p=1 row of every item (depth = the stopping depth)
    summary.json             per split and setting: accuracy per task and difficulty, macro and
                             micro means, mean depth
    results.md               per split: one row per setting, one column per task, the macro
                             mean and the mean depth
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from visro.dataset import resolve_tasks
from visro.scoring import maze_kind, score_record
from visro.scoring import summarize as visro_summarize

from .data import ImageQAExample, load_visro_examples
from .halting import select_p1_answer

PROMPT_STYLES = ("instruction", "question")
EVAL_SPLITS = ("eval", "ood")

#: Table column of each of the paper's six tasks; other tasks are labelled by their name.
TASK_LABELS = {
    "count_clusters": "Count",
    "flow_free": "Flow",
    "overlapping_glyphs": "Glyph",
    "metro_map": "Metro",
    "maze": "Maze",
    "path_tracing": "Path",
}


def answer_format_instruction(record: Mapping[str, Any]) -> str:
    """The paper's answer-format instruction for a record (it starts with a newline).

    maze: ``Answer with just the letter.`` (choice), ``Answer with only True or False.``
    (true/false) or ``Answer with just the number.`` (counting); path_tracing: none; any other
    choice question: ``Answer with just the option label.``; any other open question: none.
    """

    if record["task"] == "maze":
        return {
            "choice": "\nAnswer with just the letter.",
            "boolean": "\nAnswer with only True or False.",
            "number": "\nAnswer with just the number.",
        }[maze_kind(record)]
    if record["task"] != "path_tracing" and record["answer_type"] == "choice":
        return "\nAnswer with just the option label."
    return ""


def build_prompt(record: Mapping[str, Any], style: str) -> str:
    """The language-model text of a record for ``style`` (see the module docstring)."""

    if style == "question":
        return record["question"]
    if style == "instruction":
        return record["question"] + answer_format_instruction(record)
    raise ValueError(f"prompt style must be one of {PROMPT_STYLES}, got {style!r}")


def load_eval_examples(
    root: str | Path,
    tasks: str | Iterable[str] | None,
    split: str,
    *,
    per_cell: int = 0,
) -> tuple[ImageQAExample, ...]:
    """The examples of an evaluation split, with their records, checked by :func:`check_gold`."""

    if split not in EVAL_SPLITS:
        raise ValueError(f"evaluation split must be one of {EVAL_SPLITS}, got {split!r}")
    examples = load_visro_examples(root, tasks, split, per_cell=per_cell, keep_records=True)
    check_gold(examples)
    return examples


def check_gold(examples: Sequence[ImageQAExample]) -> None:
    """Refuse a split whose reference answers do not all score as correct."""

    bad = [
        example.sample_id
        for example in examples
        if not score_record(example.record, example.record["answer"])["correct"]
    ]
    if bad:
        raise ValueError(
            f"{len(bad)} of {len(examples)} reference answers do not score as correct "
            f"(e.g. {', '.join(bad[:5])}): the scorer and the data disagree"
        )


def score_answers(
    examples: Sequence[ImageQAExample], answers: Sequence[str], *, depth: int
) -> list[dict[str, Any]]:
    """One scored row per example (see the module docstring)."""

    if len(examples) != len(answers):
        raise ValueError(f"{len(answers)} answers for {len(examples)} examples")
    rows = []
    for example, answer in zip(examples, answers):
        result = score_record(example.record, answer)
        rows.append(
            {
                "id": example.sample_id,
                "task": example.task,
                "difficulty": example.difficulty,
                "depth": int(depth),
                "prediction": answer,
                **result,
            }
        )
    return rows


def p1_rows(
    rows_by_depth: Mapping[int, Sequence[Mapping[str, Any]]], max_depth: int
) -> list[dict[str, Any]]:
    """The p=1 row of every item over depths ``1..max_depth`` (fallback: ``max_depth``)."""

    if max_depth < 2:
        raise ValueError("p=1 stopping needs at least two depths")
    missing = [d for d in range(1, max_depth + 1) if d not in rows_by_depth]
    if missing:
        raise ValueError(f"p=1 with T={max_depth} needs depths 1..{max_depth}; missing {missing}")
    first = rows_by_depth[1]
    for depth in range(2, max_depth + 1):
        if [row["id"] for row in rows_by_depth[depth]] != [row["id"] for row in first]:
            raise ValueError(f"depth {depth} does not hold the items of depth 1 in the same order")
    out = []
    for index in range(len(first)):
        answers = [str(rows_by_depth[d][index]["prediction"]) for d in range(1, max_depth + 1)]
        selection = select_p1_answer(answers)
        out.append(
            {**rows_by_depth[selection.depth][index], "stabilized": selection.stabilized}
        )
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``visro.scoring.summarize`` plus the mean depth of the rows."""

    summary = visro_summarize(rows)
    summary["mean_depth"] = sum(int(row["depth"]) for row in rows) / len(rows) if rows else 0.0
    return summary


def generate_answers(
    examples: Sequence[ImageQAExample],
    answer: Callable[[ImageQAExample, Sequence[int]], Sequence[str]],
    depths: Sequence[int],
    *,
    log: Callable[[str], None] = print,
    every: int = 100,
) -> dict[int, list[str]]:
    """``{depth: answers}`` in example order; ``answer(example, depths)`` returns one example's
    answers at ``depths``."""

    depths = [int(depth) for depth in depths]
    out: dict[int, list[str]] = {depth: [] for depth in depths}
    start = time.time()
    for index, example in enumerate(examples, start=1):
        answers = list(answer(example, depths))
        if len(answers) != len(depths):
            raise ValueError(f"{len(answers)} answers for {len(depths)} depths")
        for depth, text in zip(depths, answers):
            out[depth].append(text)
        if index % every == 0 or index == len(examples):
            rate = (time.time() - start) / index
            log(f"[eval] {index}/{len(examples)} items, {rate:.2f}s per item")
    return out


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_split(
    output_dir: str | Path,
    split: str,
    rows_by_depth: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    p1_max_depth: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Write a split's per-depth (and p=1) rows; returns ``{setting: summary}``.

    Settings are ``depth<d>`` for every depth and ``p1_T<T>`` when ``p1_max_depth`` is set.
    """

    split_dir = Path(output_dir) / split
    summaries: dict[str, dict[str, Any]] = {}
    for depth in sorted(rows_by_depth):
        rows = rows_by_depth[depth]
        _write_jsonl(split_dir / f"depth{depth}.jsonl", rows)
        summaries[f"depth{depth}"] = summarize_rows(rows)
    if p1_max_depth:
        rows = p1_rows(rows_by_depth, p1_max_depth)
        _write_jsonl(split_dir / f"p1_T{p1_max_depth}.jsonl", rows)
        summaries[f"p1_T{p1_max_depth}"] = summarize_rows(rows)
    return summaries


def _setting_label(setting: str) -> str:
    if setting.startswith("depth"):
        return f"depth {setting[5:]}"
    return f"p=1 (T={setting.split('_T')[1]})"


def results_table(summaries: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    """Markdown tables, one per split: a row per setting, accuracy (%) per task, the macro mean
    over the tasks and the mean depth."""

    blocks = []
    for split, settings in summaries.items():
        tasks = [t for t in resolve_tasks("all") if any(t in s["tasks"] for s in settings.values())]
        tasks.sort(key=lambda t: list(TASK_LABELS).index(t) if t in TASK_LABELS else len(TASK_LABELS))
        header = [split] + [TASK_LABELS.get(t, t) for t in tasks] + ["Avg", "Depth"]
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for setting, summary in settings.items():
            cells = [_setting_label(setting)]
            for task in tasks:
                value = summary["tasks"].get(task)
                cells.append(f"{100 * value['accuracy']:.2f}" if value else "")
            cells.append(f"{100 * summary['macro_accuracy']:.2f}")
            cells.append(f"{summary['mean_depth']:.2f}")
            lines.append("| " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def write_report(
    output_dir: str | Path,
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metadata: Mapping[str, Any],
) -> str:
    """Write ``summary.json`` (``metadata`` plus the summaries) and ``results.md``; returns the
    tables."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({**metadata, "splits": summaries}, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    table = results_table(summaries)
    (output_dir / "results.md").write_text(table, encoding="utf-8")
    return table


def resolve_depths(
    max_depth: int | None, p1_max_depth: int | None, *, train_depth: int, recurrent: bool
) -> tuple[list[int], int | None]:
    """The fixed depths ``1..D`` and the p=1 budget ``T`` (``None``: no p=1).

    Defaults: ``D`` is the training depth of a recurrent checkpoint and 1 otherwise; ``T`` is the
    training depth (at most ``D``) when that is at least 2. A checkpoint without a loop accepts
    only depth 1 and no p=1.
    """

    depth = max_depth if max_depth is not None else (train_depth if recurrent else 1)
    if depth < 1:
        raise ValueError("the maximum depth must be >= 1")
    if not recurrent and depth != 1:
        raise ValueError("a checkpoint without a loop answers at depth 1 only")
    if p1_max_depth is None:
        budget = min(train_depth, depth) if recurrent else 0
        p1_max_depth = budget if budget >= 2 else 0
    if p1_max_depth:
        if not recurrent:
            raise ValueError("p=1 stopping needs a recurrent checkpoint")
        if not 2 <= p1_max_depth <= depth:
            raise ValueError(
                f"the p=1 budget must lie in [2, {depth}] (the fixed depths), got {p1_max_depth}"
            )
    return list(range(1, depth + 1)), p1_max_depth or None


def run_evaluation(
    answer: Callable[[ImageQAExample, Sequence[int]], Sequence[str]],
    args: Any,
    *,
    depths: Sequence[int],
    p1_max_depth: int | None,
    metadata: Mapping[str, Any],
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate every split of ``args`` (the options of :func:`add_eval_arguments`) one example
    at a time and write the report; returns ``{split: {setting: summary}}``."""

    summaries = {}
    for split in parse_splits(args.splits):
        examples = load_eval_examples(args.visro_root, args.tasks, split, per_cell=args.per_cell)
        log(f"[eval] {split}: {len(examples)} items; reference answers pass the scorer")
        answers = generate_answers(examples, answer, depths, log=log)
        rows = {d: score_answers(examples, answers[d], depth=d) for d in depths}
        summaries[split] = write_split(args.output_dir, split, rows, p1_max_depth=p1_max_depth)
    log(write_report(args.output_dir, summaries, metadata))
    return summaries


def add_eval_arguments(parser: Any) -> None:
    """The dataset and output options every backend's evaluation CLI shares."""

    parser.add_argument("--visro-root", required=True, help="Downloaded VisRO dataset (holds tasks/)")
    parser.add_argument(
        "--tasks", default="core6", help="Comma-separated tasks, core6 or all (default: core6)"
    )
    parser.add_argument(
        "--splits", default="eval,ood", help="Comma-separated splits: eval, ood (default: both)"
    )
    parser.add_argument(
        "--per-cell",
        type=int,
        default=0,
        help="At most this many items per (task, difficulty) cell, the first ones (0: all)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Answer at the fixed depths 1..N (default: the checkpoint's training depth)",
    )
    parser.add_argument(
        "--p1-max-depth",
        type=int,
        help=(
            "Budget T of p=1 stopping over the fixed depths (default: the training depth of a "
            "recurrent checkpoint; 0 disables it)"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output-dir", required=True)


def parse_splits(value: str) -> list[str]:
    splits = [s.strip() for s in value.split(",") if s.strip()]
    bad = [s for s in splits if s not in EVAL_SPLITS]
    if not splits or bad:
        raise ValueError(f"--splits takes {', '.join(EVAL_SPLITS)}; got {value!r}")
    return splits


__all__ = [
    "EVAL_SPLITS",
    "PROMPT_STYLES",
    "TASK_LABELS",
    "add_eval_arguments",
    "answer_format_instruction",
    "build_prompt",
    "check_gold",
    "generate_answers",
    "load_eval_examples",
    "p1_rows",
    "parse_splits",
    "resolve_depths",
    "results_table",
    "run_evaluation",
    "score_answers",
    "summarize_rows",
    "write_report",
    "write_split",
]
