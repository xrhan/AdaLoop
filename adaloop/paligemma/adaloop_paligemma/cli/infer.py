"""Run deterministic fixed-depth or p=1 PaliGemma AdaLoop inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import add_matmul_precision_argument, apply_matmul_precision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Public overlay directory")
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help="Pinned licensed PaliGemma Big Vision NPZ",
    )
    parser.add_argument(
        "--tokenizer", required=True, help="Pinned PaliGemma SentencePiece model"
    )
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument(
        "--question",
        required=True,
        help=(
            "Bare question: the AdaLN conditioner text (checkpoints trained "
            "with a constant question use that constant instead; SFT "
            "checkpoints have no conditioner) and the language prompt unless "
            "--prompt is given"
        ),
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Full language prompt, e.g. the question plus an answer-format "
            "instruction (defaults to --question; never conditions vision)"
        ),
    )
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--depth", type=int, help="Fixed recurrent depth")
    policy.add_argument(
        "--p1",
        action="store_true",
        help="Stop at the first pair of consecutive normalized answers that agree",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum p=1 depth (defaults to the checkpoint training depth)",
    )
    parser.add_argument(
        "--fallback-depth",
        type=int,
        help="p=1 fallback when no pair agrees (defaults to --max-depth)",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help=(
            "Skip SHA-256 of the checkpoint's own files; the base checkpoint and "
            "tokenizer are always verified"
        ),
    )
    add_matmul_precision_argument(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable record rather than only the answer",
    )
    return parser


def _positive(parser: argparse.ArgumentParser, name: str, value: int | None) -> None:
    if value is not None and value < 1:
        parser.error(f"{name} must be a positive integer")


def _record(result: Any, *, p1: bool) -> dict[str, Any]:
    if p1:
        return {
            "answer": result.answer,
            "depth": result.depth,
            "stabilized": result.stabilized,
            "evaluated": [
                {"depth": item.depth, "answer": item.output.answer}
                for item in result.evaluated
            ],
        }
    return {"answer": result.answer, "depth": result.depth}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("depth", "max_depth", "fallback_depth", "max_new_tokens"):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    for name in ("image", "base_checkpoint", "tokenizer"):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_file():
            parser.error(f"--{name.replace('_', '-')} is not a file: {path}")
    if not args.question.strip():
        parser.error("--question must contain non-whitespace text")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must contain non-whitespace text")
    if not args.p1 and (args.max_depth is not None or args.fallback_depth is not None):
        parser.error("--max-depth and --fallback-depth require --p1")

    import jax

    from ..inference import AdaLoopPaliGemmaPipeline

    apply_matmul_precision(jax, args.matmul_precision)
    pipeline = AdaLoopPaliGemmaPipeline.from_pretrained(
        args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        tokenizer=args.tokenizer,
        verify_hashes=not args.no_verify_hashes,
    )
    if args.p1:
        result = pipeline.generate_p1(
            args.image,
            args.question,
            prompt=args.prompt,
            max_depth=args.max_depth,
            fallback_depth=args.fallback_depth,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        result = pipeline.generate(
            args.image,
            args.question,
            prompt=args.prompt,
            depth=args.depth,
            max_new_tokens=args.max_new_tokens,
        )
    payload = _record(result, p1=args.p1)
    print(json.dumps(payload, ensure_ascii=False) if args.json else payload["answer"])


if __name__ == "__main__":  # pragma: no cover
    main()
