"""Evaluate a Qwen AdaLoop checkpoint on VisRO at fixed depths and with p=1 stopping."""

from __future__ import annotations

import argparse
import io
from typing import Any, Sequence

from adaloop_common.evaluation import (
    PROMPT_STYLES,
    add_eval_arguments,
    build_prompt,
    resolve_depths,
    run_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Writes <output-dir>/<split>/depth<d>.jsonl and p1_T<T>.jsonl (one scored row per "
            "item), summary.json and results.md."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Local directory or Hub repo")
    parser.add_argument("--checkpoint-revision", help="Optional immutable Hub revision")
    add_eval_arguments(parser)
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default="instruction",
        help=(
            "Language-model text: the question plus the paper's answer-format "
            "instruction (default), or the bare question; the condition is always "
            "the bare question"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="sdpa",
    )
    parser.add_argument(
        "--no-merge-lora",
        action="store_true",
        help="Keep the LoRA adapters unmerged (the paper evaluation merged them)",
    )
    parser.add_argument("--no-verify-hashes", action="store_true")
    return parser


def _open_image(data: bytes) -> Any:
    from PIL import Image, ImageOps

    # As the pipeline opens an image path: EXIF orientation applied, then RGB.
    with Image.open(io.BytesIO(data)) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be a positive integer")
    if args.per_cell < 0:
        parser.error("--per-cell must be non-negative")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Evaluation requires the GPU dependencies; install adaloop-qwen[gpu]"
        ) from exc
    from ..inference import AdaLoopQwenPipeline

    pipeline = AdaLoopQwenPipeline.from_pretrained(
        args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        device=args.device,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        attn_implementation=args.attn_implementation,
        merge_lora=not args.no_merge_lora,
        verify_hashes=not args.no_verify_hashes,
    )
    config = pipeline.config
    try:
        depths, p1_max_depth = resolve_depths(
            args.max_depth,
            args.p1_max_depth,
            train_depth=int(config.train_depth),
            recurrent=config.variant in {"a", "b", "c"},
        )
    except ValueError as exc:
        parser.error(str(exc))

    def answer(example: Any, depths: Sequence[int]) -> list[str]:
        image = _open_image(example.image.read_bytes())
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
            "backend": config.backend,
            "checkpoint": args.checkpoint,
            "variant": config.variant,
            "tasks": args.tasks,
            "prompt_style": args.prompt_style,
            "depths": depths,
            "p1_max_depth": p1_max_depth,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
        },
    )


if __name__ == "__main__":
    main()
