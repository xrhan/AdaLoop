"""Inspect and strictly verify a portable PaliGemma AdaLoop checkpoint."""

from __future__ import annotations

import argparse
import json

from ..checkpoint import PaliGemmaCheckpointLoader, read_overlay_headers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--base-checkpoint",
        help="Also verify the external pinned Big Vision NPZ",
    )
    parser.add_argument(
        "--tokenizer",
        help="Also verify the external pinned SentencePiece model",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help=(
            "Debug-only: skip SHA-256 of the checkpoint's own files; a given base "
            "checkpoint and tokenizer are always verified"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if (args.base_checkpoint is None) != (args.tokenizer is None):
        raise SystemExit("--base-checkpoint and --tokenizer must be supplied together")
    verify_hashes = not args.skip_hashes
    if args.base_checkpoint is not None:
        verified = PaliGemmaCheckpointLoader.from_pretrained(
            args.checkpoint,
            base_checkpoint=args.base_checkpoint,
            tokenizer=args.tokenizer,
            verify_hashes=verify_hashes,
            reconstruct=False,
            restore_hook=read_overlay_headers,
        )
        manifest = verified.manifest
        assets_verified = True
    else:
        _, manifest, _, _, _ = PaliGemmaCheckpointLoader.verify_artifact(
            args.checkpoint,
            verify_hashes=verify_hashes,
            restore_hook=read_overlay_headers,
        )
        assets_verified = False
    output = manifest.to_dict()
    output["external_assets_verified"] = assets_verified
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
