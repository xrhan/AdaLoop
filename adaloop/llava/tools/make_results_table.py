#!/usr/bin/env python3
"""Build the paper's LLaVA table on the six core VisRO tasks from evaluation outputs and compare it
with the paper.

    python adaloop/llava/tools/make_results_table.py --eval-root outputs/eval [--tol 0.0] [--strict]

Reads ``<eval-root>/<name>/summary.json`` as written by ``adaloop-llava-eval``, for each row of
the table:

=====================  ==================  ==========================================
row                    name (option)       setting
=====================  ==================  ==========================================
Zero-shot              ``--zeroshot``      ``depth1``
SFT                    ``--sft``           ``depth1``
Single-pass AdaLN      ``--adaln``         ``depth1``
AdaLoop (p=1, T=3)     ``--adaloop``       ``p1_T3``
AdaLoop, fixed depth 3 ``--adaloop``       ``depth3``
=====================  ==================  ==========================================

Prints the main table (average over the six tasks on ood and eval, per-task eval accuracy, mean
depth), the per-task ood accuracy, and every cell next to the paper's value. Cells are compared
after rounding to two decimals; ``--tol`` (points) allows a difference, ``--strict`` makes any
difference beyond it an error (exit status 1). Missing summaries leave their cells empty.
"""

import argparse
import json
import os
import sys

from adaloop_common.evaluation import TASK_LABELS

TASKS = list(TASK_LABELS.values())

#: Paper values (accuracy in %, the six core VisRO tasks, LLaVA-1.5-7B at 448px; "id" is the eval
#: split). "t_bar" is the mean stopping depth.
PAPER = {
    "zeroshot": {
        "id": {"overall": 10.42, "Count": 6.00, "Flow": 0.17, "Glyph": 0.50, "Metro": 13.67, "Maze": 29.50, "Path": 12.67},
        "ood": {"overall": 8.56, "Count": 0.00, "Flow": 0.00, "Glyph": 0.00, "Metro": 12.67, "Maze": 26.17, "Path": 12.50},
    },
    "sft": {
        "id": {"overall": 50.44, "Count": 48.33, "Flow": 18.00, "Glyph": 62.83, "Metro": 51.67, "Maze": 59.50, "Path": 62.33},
        "ood": {"overall": 24.78, "Count": 2.33, "Flow": 1.17, "Glyph": 24.67, "Metro": 37.83, "Maze": 34.50, "Path": 48.17},
    },
    "adaln": {
        "id": {"overall": 57.69, "Count": 52.50, "Flow": 29.50, "Glyph": 82.00, "Metro": 53.00, "Maze": 63.50, "Path": 65.67},
        "ood": {"overall": 29.56, "Count": 2.33, "Flow": 4.17, "Glyph": 37.33, "Metro": 41.33, "Maze": 36.33, "Path": 55.83},
    },
    "adaloop_p1": {
        "id": {"overall": 61.78, "Count": 56.83, "Flow": 38.67, "Glyph": 84.67, "Metro": 59.00, "Maze": 64.17, "Path": 67.33,
               "t_bar": 2.19},
        "ood": {"overall": 31.11, "Count": 2.33, "Flow": 6.00, "Glyph": 42.33, "Metro": 43.33, "Maze": 33.33, "Path": 59.33,
                "t_bar": 2.29},
    },
    "adaloop_d3": {
        "id": {"overall": 62.11, "Count": 56.67, "Flow": 40.00, "Glyph": 85.00, "Metro": 57.67, "Maze": 65.67, "Path": 67.67},
        "ood": {"overall": 31.00, "Count": 2.17, "Flow": 5.50, "Glyph": 42.17, "Metro": 42.83, "Maze": 33.67, "Path": 59.67},
    },
}

ROWS = [("zeroshot", "Zero-shot"), ("sft", "SFT (LoRA)"), ("adaln", "Single-pass AdaLN"),
        ("adaloop_p1", "AdaLoop (p=1, T=3)"), ("adaloop_d3", "AdaLoop, fixed depth 3")]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def cells_from_summary(summary):
    """Percent accuracy per paper column plus overall (the average over the tasks) and the mean depth."""
    out = {"overall": 100 * summary["macro_accuracy"], "n": summary["n"], "t_bar": summary["mean_depth"]}
    for task, label in TASK_LABELS.items():
        if task in summary["tasks"]:
            out[label] = 100 * summary["tasks"][task]["accuracy"]
    return out


def collect(root, names):
    """``{row: {split: cells}}`` from the evaluation summaries (missing ones are skipped)."""
    settings = {"zeroshot": (names["zeroshot"], "depth1"), "sft": (names["sft"], "depth1"),
                "adaln": (names["adaln"], "depth1"), "adaloop_p1": (names["adaloop"], "p1_T3"),
                "adaloop_d3": (names["adaloop"], "depth3")}
    out, missing = {}, []
    for key, (name, setting) in settings.items():
        out[key] = {}
        path = os.path.join(root, name, "summary.json")
        doc = _load(path)
        for paper_split, split in (("id", "eval"), ("ood", "ood")):
            summary = (doc or {}).get("splits", {}).get(split, {}).get(setting)
            if summary is None:
                missing.append(f"{path}: {split}/{setting}")
                continue
            out[key][paper_split] = cells_from_summary(summary)
    return out, missing


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main_table(res):
    head = "| Method | OOD | ID | " + " | ".join(TASKS) + " | T\u0304 |"
    lines = [head, "|" + "---|" * (len(TASKS) + 4)]
    for key, label in ROWS:
        idc, oodc = res.get(key, {}).get("id", {}), res.get(key, {}).get("ood", {})
        if key == "adaloop_p1":
            tb = f"{fmt(idc.get('t_bar'))} (OOD {fmt(oodc.get('t_bar'))})"
        else:
            tb = fmt(idc.get("t_bar"), 0)
        lines.append(f"| {label} | {fmt(oodc.get('overall'))} | {fmt(idc.get('overall'))} | "
                     + " | ".join(fmt(idc.get(t)) for t in TASKS) + f" | {tb} |")
    return lines


def ood_table(res):
    lines = ["| Method (OOD) | " + " | ".join(TASKS) + " |", "|" + "---|" * (len(TASKS) + 1)]
    for key, label in ROWS:
        oodc = res.get(key, {}).get("ood", {})
        lines.append(f"| {label} | " + " | ".join(fmt(oodc.get(t)) for t in TASKS) + " |")
    return lines


def compare(res, tol):
    """``(lines, n_compared, n_mismatch)`` comparing every available cell with the paper."""
    lines, n, bad = [], 0, 0
    for key, label in ROWS:
        for split in ("id", "ood"):
            got = res.get(key, {}).get(split)
            if not got:
                continue
            for col, want in PAPER[key][split].items():
                if col not in got:
                    continue
                g = round(got[col], 2)
                diff = g - want
                ok = abs(diff) <= tol + 1e-9
                n += 1
                bad += 0 if ok else 1
                if not ok:
                    lines.append(f"  DIFF  {label:<24s} {split:<3s} {col:<8s} measured {g:6.2f}  paper {want:6.2f}  "
                                 f"({diff:+.2f})")
    return lines, n, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-root", default="outputs/eval")
    ap.add_argument("--zeroshot", default="zeroshot")
    ap.add_argument("--sft", default="sft")
    ap.add_argument("--adaln", default="adaln_single_pass")
    ap.add_argument("--adaloop", default="adaloop")
    ap.add_argument("--tol", type=float, default=0.0, help="allowed difference in points (default 0)")
    ap.add_argument("--strict", action="store_true", help="exit with status 1 on a difference beyond --tol")
    ap.add_argument("--json", default=None, help="also write the measured cells as JSON")
    a = ap.parse_args(argv)

    names = {"zeroshot": a.zeroshot, "sft": a.sft, "adaln": a.adaln, "adaloop": a.adaloop}
    res, missing = collect(a.eval_root, names)
    print("Six core VisRO tasks, accuracy (%). Per-task columns are eval (ID); overall = average over the six "
          "tasks.\n")
    print("\n".join(main_table(res)))
    print()
    print("\n".join(ood_table(res)))
    if missing:
        print("\nMissing (cells left empty):")
        for p in missing:
            print(f"  {p}")
    lines, n, bad = compare(res, a.tol)
    print(f"\nComparison with the paper (tolerance {a.tol:.2f} pt): {n - bad}/{n} cells match")
    for line in lines:
        print(line)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    if a.strict and bad:
        sys.exit(1)
    return res


if __name__ == "__main__":
    main()
