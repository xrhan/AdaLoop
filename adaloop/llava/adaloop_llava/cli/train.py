"""``adaloop-llava-train``: train one LLaVA-1.5 recipe (SFT, single-pass AdaLN or Variant A, B or C).

    torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.train \\
        --config adaloop/llava/configs/variant_a.yaml --output-dir outputs/adaloop \\
        --base-model checkpoints/llava-v1.5-7b-448-s2 \\
        --vision-tower checkpoints/clip-vit-large-patch14-336 \\
        --visro-root data/VisRO --replay-jsonl data/llava_replay/replay.jsonl

One process, or one process per GPU under ``torchrun`` (plain DDP through the Hugging Face
``Trainer``). Every flag is a dotted override of the recipe (``--max-steps 20`` is
``train.max_steps=20``, see :data:`FLAG_OVERRIDES`); ``--set KEY=VALUE`` passes any other
override through, its value parsed as YAML. The resolved recipe is validated
(:func:`adaloop_llava.config.load_config`) and trained by
:func:`adaloop_llava.training.train.main`, which writes the adapter, ``non_lora_trainables.bin``,
``config.json``, ``resolved_config.yaml`` and ``checkpoint-<step>/`` directories to
``--output-dir``.

``--resume-from-checkpoint`` defaults to the recipe's ``train.resume: auto``: starting the same
command again continues from the latest ``checkpoint-<step>`` of the output directory.

At the end of a run, the main process exports the portable checkpoint to
``<output-dir>/release`` (:func:`adaloop_llava.checkpoint.export_release_checkpoint`: the recipe,
runtime contract, adapter, ``non_lora_trainables.safetensors`` and a manifest with the SHA-256 of
every checkpoint, base and CLIP file). ``--no-export`` skips this; ``adaloop-llava-convert`` exports
the same checkpoint later.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Sequence

from ..checkpoint import RELEASE_SUBDIR

#: ``(argparse destination, dotted recipe key)`` of every flag that maps onto the recipe.
FLAG_OVERRIDES = (
    ("output_dir", "paths.output_dir"),
    ("base_model", "paths.base_model"),
    ("vision_tower", "paths.vision_tower"),
    ("visro_root", "data.visro_root"),
    ("tasks", "data.tasks"),
    ("replay_jsonl", "data.replay_jsonl"),
    ("replay_image_root", "data.replay_image_root"),
    ("per_device_batch_size", "train.per_device_batch_size"),
    ("gradient_accumulation_steps", "train.grad_accum"),
    ("max_steps", "train.max_steps"),
    ("save_steps", "train.save_steps"),
    ("save_total_limit", "train.save_total_limit"),
    ("num_workers", "train.dataloader_num_workers"),
    ("attn_implementation", "model.attn_implementation"),
    ("seed", "train.seed"),
    ("resume_from_checkpoint", "train.resume"),
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adaloop-llava-train",
                                description="Train SFT / single-pass AdaLN / Variants A, B, C on LLaVA-1.5 from a recipe.")
    p.add_argument("--config", required=True, help="recipe, e.g. adaloop/llava/configs/variant_a.yaml")
    p.add_argument("--output-dir", required=True, help="output directory (paths.output_dir)")
    p.add_argument("--base-model", help="stage-2 LLaVA-1.5 base directory (paths.base_model)")
    p.add_argument("--vision-tower", help="CLIP ViT-L/14-336 directory (paths.vision_tower)")
    p.add_argument("--visro-root", help="downloaded VisRO dataset, holds tasks/ (data.visro_root)")
    p.add_argument("--tasks", help="comma-separated VisRO tasks, core6 or all (data.tasks)")
    p.add_argument("--replay-jsonl", help="replay JSONL pooled after the VisRO rows (data.replay_jsonl; "
                                          "--set data.replay_jsonl=null trains without replay)")
    p.add_argument("--replay-image-root", help="base of relative replay image paths "
                                               "(data.replay_image_root)")
    p.add_argument("--per-device-batch-size", type=int, help="train.per_device_batch_size")
    p.add_argument("--gradient-accumulation-steps", type=int, help="train.grad_accum")
    p.add_argument("--max-steps", type=int, help="train.max_steps")
    p.add_argument("--save-steps", type=int, help="train.save_steps")
    p.add_argument("--save-total-limit", type=int, help="train.save_total_limit")
    p.add_argument("--num-workers", type=int, help="dataloader workers (train.dataloader_num_workers)")
    p.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"),
                   help="language-model attention (model.attn_implementation)")
    p.add_argument("--seed", type=int, help="train.seed")
    p.add_argument("--resume-from-checkpoint", metavar="{auto,never,DIR}",
                   help="train.resume: auto (latest checkpoint of the output directory, the recipe default), "
                        "never, or a checkpoint directory")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="any other dotted recipe override, value parsed as YAML (repeatable), "
                        "e.g. train.bf16=false")
    p.add_argument("--no-export", action="store_true",
                   help=f"do not export the portable checkpoint to <output-dir>/{RELEASE_SUBDIR} at the end "
                        "(adaloop-llava-convert can export it later)")
    return p


def _yaml_value(value) -> str:
    """An override value that ``yaml.safe_load`` reads back unchanged (strings are quoted)."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def overrides_from_args(args: argparse.Namespace) -> List[str]:
    """The dotted overrides of the flags that were given, followed by the ``--set`` items."""
    out = [f"{key}={_yaml_value(getattr(args, dest))}" for dest, key in FLAG_OVERRIDES
           if getattr(args, dest) is not None]
    flagged = {item.split("=", 1)[0] for item in out}
    for item in args.overrides:
        key = item.split("=", 1)[0].strip()
        if key in flagged:
            raise SystemExit(f"--set {item!r} repeats a key that a command-line flag already sets")
        out.append(item)
    return out


def setup_logging() -> None:
    rank = int(os.environ.get("RANK", "0"))
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(level=level, stream=sys.stdout,
                        format=f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s")
    import transformers
    transformers.utils.logging.set_verbosity_warning()


def resolve_config(args: argparse.Namespace):
    """The validated recipe of a parsed command line."""
    from ..config import load_config
    return load_config(args.config, overrides_from_args(args))


def export_run(cfg) -> str:
    """Export the finished run in ``cfg.output_dir`` to ``<output_dir>/release`` (replacing the
    checkpoint files of an earlier export of the same run); returns the manifest path."""
    from ..checkpoint import export_release_checkpoint
    manifest = export_release_checkpoint(cfg.output_dir, os.path.join(cfg.output_dir, RELEASE_SUBDIR),
                                         recipe=cfg, overwrite=True)
    logging.getLogger("adaloop_llava.train").info("Exported the portable checkpoint: %s", manifest)
    return str(manifest)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    setup_logging()
    from ..training.train import main as train
    train(cfg)
    if not args.no_export and int(os.environ.get("RANK", "0")) == 0:
        export_run(cfg)


if __name__ == "__main__":
    main()
