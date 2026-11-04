"""Run deterministic fixed-depth or p=1 inference with a release checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local directory or Hub repo")
    parser.add_argument("--checkpoint-revision", help="Optional immutable Hub revision")
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument(
        "--question",
        required=True,
        help=(
            "Bare question: the AdaLN conditioner text (checkpoints trained "
            "with a constant question use that constant instead; SFT and "
            "no-AdaLN checkpoints have no conditioner) and the language-model "
            "text unless --prompt is given"
        ),
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Full language-model text, e.g. the question plus an answer-format "
            "instruction (defaults to --question)"
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="sdpa",
    )
    parser.add_argument(
        "--no-merge-lora",
        action="store_true",
        help=(
            "Keep the LoRA adapters unmerged instead of merging them into the "
            "base weights as the paper evaluation did; the two agree in "
            "float32, but BF16 rounding can rarely flip a near-tie"
        ),
    )
    parser.add_argument("--no-verify-hashes", action="store_true")
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
                {
                    "depth": item.depth,
                    "answer": item.output.answer,
                }
                for item in result.evaluated
            ],
        }
    return {
        "answer": result.answer,
        "depth": result.depth,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("depth", "max_depth", "fallback_depth", "max_new_tokens"):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    image = Path(os.path.expanduser(args.image))
    if not image.is_file():
        parser.error(f"--image is not a file: {args.image}")
    if not args.question.strip():
        parser.error("--question must contain non-whitespace text")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must contain non-whitespace text")
    if not args.p1 and (args.max_depth is not None or args.fallback_depth is not None):
        parser.error("--max-depth and --fallback-depth require --p1")
    if args.p1 and args.depth is not None:  # guarded by argparse, kept explicit
        parser.error("--p1 and --depth are mutually exclusive")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Inference requires the GPU dependencies; install adaloop-qwen[gpu]"
        ) from exc
    from ..inference import AdaLoopQwenPipeline

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    pipeline = AdaLoopQwenPipeline.from_pretrained(
        args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        merge_lora=not args.no_merge_lora,
        verify_hashes=not args.no_verify_hashes,
    )
    if args.p1:
        result = pipeline.generate_p1(
            image,
            args.question,
            prompt=args.prompt,
            max_depth=args.max_depth,
            fallback_depth=args.fallback_depth,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        result = pipeline.generate(
            image,
            args.question,
            prompt=args.prompt,
            depth=args.depth,
            max_new_tokens=args.max_new_tokens,
        )
    payload = _record(result, p1=args.p1)
    print(json.dumps(payload, ensure_ascii=False) if args.json else payload["answer"])


if __name__ == "__main__":
    main()
