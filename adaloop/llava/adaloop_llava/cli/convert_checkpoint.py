"""``adaloop-llava-convert``: convert an adapter directory into a portable checkpoint.

    adaloop-llava-convert --adapter-dir outputs/adaloop \\
        --config adaloop/llava/configs/variant_a.yaml --output-dir outputs/adaloop/release \\
        --base-model checkpoints/llava-v1.5-7b-448-s2 \\
        --vision-tower checkpoints/clip-vit-large-patch14-336

The adapter directory holds ``adapter_config.json``, ``adapter_model.safetensors``,
``non_lora_trainables.bin`` and ``config.json``, as the training code writes them (the final
output directory or a ``checkpoint-<step>`` directory). Its ``config.json`` and
``adapter_config.json`` are compared field by field with the recipe, the base and CLIP
directories are fingerprinted, and the sidecar is rewritten to safetensors bit for bit (see
:func:`adaloop_llava.checkpoint.convert_adapter_checkpoint`). ``--base-fingerprint`` reuses the
base and CLIP hashes of an earlier conversion (its ``adaloop_manifest.json``): the directories are
then checked by size and small-file hashes only. Prints the manifest path.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adaloop-llava-convert",
                                description="Convert a LLaVA-1.5 adapter directory into a portable "
                                            "AdaLoop checkpoint.")
    p.add_argument("--adapter-dir", required=True,
                   help="adapter directory with adapter_config.json, adapter_model.safetensors, "
                        "non_lora_trainables.bin and config.json")
    p.add_argument("--config", required=True,
                   help="recipe the checkpoint was trained with, e.g. adaloop/llava/configs/variant_a.yaml")
    p.add_argument("--output-dir", required=True, help="destination of the portable checkpoint")
    p.add_argument("--base-model", required=True, help="stage-2 LLaVA-1.5 base directory (fingerprinted)")
    p.add_argument("--vision-tower", required=True, help="CLIP ViT-L/14-336 directory (fingerprinted)")
    p.add_argument("--base-fingerprint", metavar="JSON",
                   help="reuse the base_model / vision_tower fingerprint of this JSON file, e.g. the "
                        "adaloop_manifest.json of an earlier conversion against the same base")
    p.add_argument("--overwrite", action="store_true",
                   help="replace the checkpoint files of a non-empty output directory")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    from ..checkpoint import convert_adapter_checkpoint
    manifest = convert_adapter_checkpoint(args.adapter_dir, args.config, args.output_dir,
                                         base_model=args.base_model, vision_tower=args.vision_tower,
                                         base_fingerprint=args.base_fingerprint, overwrite=args.overwrite)
    print(manifest)


if __name__ == "__main__":
    main()
