"""``adaloop-llava-infer``: answer one question about one image with a portable checkpoint.

    adaloop-llava-infer --checkpoint outputs/adaloop/release \\
        --base-model checkpoints/llava-v1.5-7b-448-s2 \\
        --vision-tower checkpoints/clip-vit-large-patch14-336 \\
        --image example.png --question "How many red circles are there?" --p1 --json

``--checkpoint`` is a portable checkpoint directory (``adaloop-llava-train`` writes one to
``<output-dir>/release``; ``adaloop-llava-convert`` converts adapter directories). The
base and CLIP directories are supplied at load time and checked against the checkpoint's
manifest: every file's size and the SHA-256 of the small files by default, every byte with
``--verify-base-hashes``.

``--depth N`` answers at a fixed depth (default: 1 for SFT and single-pass AdaLN, 3 for
Variants A, B and C); ``--p1`` (Variants A, B and C) runs depths 1, 2, ... and stops at the first answer that
repeats the previous one (:func:`adaloop_common.select_p1_answer`). Decoding is greedy, at most
``--max-new-tokens`` (default 32) new tokens, in float16 unless ``--dtype`` says otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adaloop-llava-infer",
                                description="Greedy single-example inference with a portable LLaVA-1.5 "
                                            "AdaLoop checkpoint.")
    p.add_argument("--checkpoint", required=True, help="portable checkpoint directory (with adaloop_manifest.json)")
    p.add_argument("--base-model", required=True, help="stage-2 LLaVA-1.5 base directory the checkpoint was made on")
    p.add_argument("--vision-tower", required=True, help="CLIP ViT-L/14-336 directory the checkpoint was made with")
    p.add_argument("--image", required=True, help="path to one image")
    p.add_argument("--question", required=True,
                   help="bare question: the question condition of loop models and, unless --prompt is given, "
                        "the user turn of the language-model prompt")
    p.add_argument("--prompt", help="language-model text replacing --question in the prompt only "
                                    "(e.g. the question plus an answer-format instruction)")
    policy = p.add_mutually_exclusive_group()
    policy.add_argument("--depth", type=int, help="fixed depth (default: the checkpoint's training depth)")
    policy.add_argument("--p1", action="store_true",
                        help="p=1 answer-stability halting (Variants A, B, C): stop at the first answer that repeats "
                             "the previous depth's answer")
    p.add_argument("--max-depth", type=int, help="largest depth --p1 evaluates (default: the training depth, 3)")
    p.add_argument("--fallback-depth", type=int,
                   help="answer depth of --p1 when no two consecutive answers agree (default: --max-depth)")
    p.add_argument("--max-new-tokens", type=int, help="answer budget (default: 32, the checkpoint's runtime contract)")
    p.add_argument("--device", default="cuda", help="torch device (default cuda)")
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16",
                   help="language-model and CLIP precision (default float16, as in the paper evaluation; "
                        "the loop core always runs in float32)")
    p.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    p.add_argument("--no-verify-hashes", action="store_true",
                   help="check file sizes only (no SHA-256) for the checkpoint and the base / CLIP files")
    p.add_argument("--verify-base-hashes", action="store_true",
                   help="also hash every base and CLIP weight file (slow for the 7B base)")
    p.add_argument("--json", action="store_true", help="print a JSON record instead of only the answer")
    return p


def _record(result: Any, *, p1: bool) -> Dict[str, Any]:
    if p1:
        return {"answer": result.answer, "depth": result.depth, "stabilized": result.stabilized,
                "evaluated": [{"depth": item.depth, "answer": item.output.answer} for item in result.evaluated]}
    return {"answer": result.answer, "depth": result.depth, "token_ids": list(result.token_ids)}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("depth", "max_depth", "fallback_depth", "max_new_tokens"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    if not os.path.isfile(os.path.expanduser(args.image)):
        parser.error(f"--image is not a file: {args.image}")
    if not args.question.strip():
        parser.error("--question must contain non-whitespace text")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must contain non-whitespace text")
    if not args.p1 and (args.max_depth is not None or args.fallback_depth is not None):
        parser.error("--max-depth and --fallback-depth require --p1")

    from ..checkpoint import LlavaCheckpointLoader
    manifest = LlavaCheckpointLoader.inspect(args.checkpoint)
    if args.p1 and manifest.variant not in ("a", "b", "c"):
        parser.error(f"--p1 needs a Variant A, B or C checkpoint; {args.checkpoint} is {manifest.variant!r}")
    if args.p1 and args.max_depth is not None and args.max_depth < 2:
        parser.error("--max-depth must be at least 2 for --p1")
    if args.p1 and args.fallback_depth is not None and args.fallback_depth > (args.max_depth or manifest.train_depth):
        parser.error("--fallback-depth must not exceed --max-depth")
    if args.depth is not None and manifest.variant == "sft" and args.depth != 1:
        parser.error("an SFT checkpoint has no loop; only --depth 1 exists")

    from transformers.utils import logging as transformers_logging

    from ..inference import AdaLoopLlavaPipeline

    # As in adaloop-llava-eval: Transformers logs errors only. The base loads without the loop core,
    # whose weights come from the checkpoint right after, so its "newly initialized ... TRAIN this
    # model" notice would only mislead. The previous verbosity is restored for in-process callers.
    verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        pipeline = AdaLoopLlavaPipeline.from_pretrained(
            args.checkpoint, base_model=args.base_model, vision_tower=args.vision_tower, device=args.device,
            dtype=args.dtype, attn_implementation=args.attn_implementation,
            verify_hashes=not args.no_verify_hashes, verify_base_hashes=args.verify_base_hashes)
        if args.p1:
            result = pipeline.generate_p1(args.image, args.question, prompt=args.prompt, max_depth=args.max_depth,
                                          fallback_depth=args.fallback_depth, max_new_tokens=args.max_new_tokens)
        else:
            result = pipeline.generate(args.image, args.question, prompt=args.prompt, depth=args.depth,
                                       max_new_tokens=args.max_new_tokens)
    finally:
        transformers_logging.set_verbosity(verbosity)
    payload = _record(result, p1=args.p1)
    print(json.dumps(payload, ensure_ascii=False) if args.json else payload["answer"])


if __name__ == "__main__":
    main()
