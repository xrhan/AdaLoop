"""Inspect and optionally hash-verify an AdaLoop Qwen2.5-VL checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from ..checkpoint import QwenCheckpointLoader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--verify-hashes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = QwenCheckpointLoader.inspect(args.checkpoint)
    manifest.validate_files(args.checkpoint, verify_hashes=args.verify_hashes)
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
