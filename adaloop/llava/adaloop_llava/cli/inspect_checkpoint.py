"""``adaloop-llava-inspect``: validate a portable checkpoint and print its manifest.

    adaloop-llava-inspect outputs/adaloop/release --verify-hashes

Checks the manifest, the size of every checkpoint file (and its SHA-256 with ``--verify-hashes``),
the agreement of the manifest, ``adaloop_config.yaml``, ``adaloop_runtime.json``, ``config.json``
and ``adapter_config.json``, and the exact tensor inventories (read from the safetensors headers).
Prints the manifest as JSON on stdout and a one-line summary on stderr. It does not import torch
and does not need the base model; a failed check raises
:class:`~adaloop_llava.errors.CheckpointValidationError`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adaloop-llava-inspect",
                                description="Validate a portable LLaVA-1.5 AdaLoop checkpoint and print its manifest.")
    p.add_argument("checkpoint", help="portable checkpoint directory")
    p.add_argument("--verify-hashes", action="store_true", help="also check the SHA-256 of every checkpoint file")
    return p


def summary(verified) -> str:
    """One line: variant, loop mode, span, training depth and the language layout of the answer."""
    from adaloop_common import Variant, layout_notation
    t = verified.manifest.topology
    span = "no loop" if t["loop_span"] is None else f"span [{t['loop_span'][0]}, {t['loop_span'][1]})"
    layout = {"b": Variant.B, "c": Variant.C}.get(t["variant"], Variant.A)
    return (f"variant {t['variant']} (loop mode {t['loop_mode']}), {span}, training depth {t['train_depth']}, "
            f"language layout {layout_notation(layout, t['train_depth'])}, "
            f"{len(verified.manifest.sidecar)} sidecar tensors")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    from ..checkpoint import LlavaCheckpointLoader
    verified = LlavaCheckpointLoader.verify(args.checkpoint, verify_hashes=args.verify_hashes)
    print(json.dumps(verified.manifest.to_dict(), indent=2, sort_keys=True))
    hashed = "all file hashes verified" if args.verify_hashes else "file sizes verified (hashes not checked)"
    print(f"{summary(verified)}; {hashed}", file=sys.stderr)


if __name__ == "__main__":
    main()
