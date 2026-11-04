"""Evaluate a PaliGemma AdaLoop checkpoint on VisRO at fixed depths and with p=1 stopping."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from adaloop_common.evaluation import (
    PROMPT_STYLES,
    add_eval_arguments,
    build_prompt,
    resolve_depths,
    run_evaluation,
)

from ..constants import RECURRENT_VARIANTS
from . import add_matmul_precision_argument, apply_matmul_precision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Writes <output-dir>/<split>/depth<d>.jsonl and p1_T<T>.jsonl (one scored row per "
            "item), summary.json and results.md."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Public overlay directory")
    parser.add_argument(
        "--base-checkpoint", required=True, help="Pinned licensed PaliGemma Big Vision NPZ"
    )
    parser.add_argument("--tokenizer", required=True, help="Pinned PaliGemma SentencePiece model")
    add_eval_arguments(parser)
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default="instruction",
        help=(
            "Language prompt: the question plus the paper's answer-format "
            "instruction (default), or the bare question; the condition is always "
            "the bare question"
        ),
    )
    add_matmul_precision_argument(parser)
    parser.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help=(
            "Skip SHA-256 of the checkpoint's own files; the base checkpoint and "
            "tokenizer are always verified"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be a positive integer")
    if args.per_cell < 0:
        parser.error("--per-cell must be non-negative")
    for name in ("base_checkpoint", "tokenizer"):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_file():
            parser.error(f"--{name.replace('_', '-')} is not a file: {path}")

    import jax

    from ..inference import AdaLoopPaliGemmaPipeline

    matmul_precision = apply_matmul_precision(jax, args.matmul_precision)
    pipeline = AdaLoopPaliGemmaPipeline.from_pretrained(
        args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        tokenizer=args.tokenizer,
        verify_hashes=not args.no_verify_hashes,
    )
    recipe = pipeline.recipe
    try:
        depths, p1_max_depth = resolve_depths(
            args.max_depth,
            args.p1_max_depth,
            train_depth=int(recipe.train_depth),
            recurrent=recipe.variant in RECURRENT_VARIANTS,
        )
    except ValueError as exc:
        parser.error(str(exc))

    def answer(example: Any, depths: Sequence[int]) -> list[str]:
        image = example.image.read_bytes()
        prompt = build_prompt(example.record, args.prompt_style)
        return [
            pipeline.generate(
                image,
                example.question,
                prompt=prompt,
                depth=depth,
                max_new_tokens=args.max_new_tokens,
            ).answer
            for depth in depths
        ]

    run_evaluation(
        answer,
        args,
        depths=depths,
        p1_max_depth=p1_max_depth,
        metadata={
            "backend": recipe.backend,
            "checkpoint": args.checkpoint,
            "variant": recipe.variant,
            "tasks": args.tasks,
            "prompt_style": args.prompt_style,
            "matmul_precision": matmul_precision,
            "depths": depths,
            "p1_max_depth": p1_max_depth,
            "max_new_tokens": args.max_new_tokens,
        },
    )


if __name__ == "__main__":  # pragma: no cover
    main()
