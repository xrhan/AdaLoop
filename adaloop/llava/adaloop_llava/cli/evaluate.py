"""``adaloop-llava-eval``: evaluate zero-shot / SFT / single-pass AdaLN / Variants A, B, C on VisRO.

    adaloop-llava-eval --config adaloop/llava/configs/eval/variant_a.yaml [--set KEY=VALUE ...]
    torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.evaluate \\
        --config adaloop/llava/configs/eval/variant_a.yaml

Single process, or one process per GPU under ``torchrun``. ``--set`` takes dotted overrides of the
evaluation config, values parsed as YAML (``--set eval.splits=[ood]``). :func:`run` does:

1. Read the items of every split in ``eval.splits`` (``eval`` and/or ``ood``) of ``eval.tasks``
   under ``eval.visro_root``, at most ``per_cell`` per (task, difficulty) cell, and check that every
   reference answer scores as correct, before the model is loaded
   (:func:`adaloop_common.evaluation.load_eval_examples`).
2. Load the model (:func:`adaloop_llava.model.loading.load_for_eval`): the LoRA run in
   ``eval.adapter`` merged into the base model, or the base model itself when ``adapter`` is
   null (zero-shot). LM and CLIP in ``eval.dtype``; the loop core is kept in fp32.
   ``eval.adapter`` is an adapter directory (``non_lora_trainables.bin``) or a portable
   checkpoint (``non_lora_trainables.safetensors``); a portable checkpoint is first verified
   against its manifest, together with the base and CLIP directories of ``paths``
   (:func:`verify_portable_adapter`).
3. For every split and every depth ``1..eval.max_depth``: greedy answers (:func:`run_depth`;
   the language-model text follows ``eval.prompt_style``, the condition is the bare question).
   The loop mode and the meaning of the depth come from the checkpoint; a model without a loop
   accepts only depth 1.
4. Score the answers with ``visro.scoring``, apply ``p=1`` stopping over depths
   ``1..eval.p1_max_depth`` (when set) and write ``<output_dir>/<split>/depth<d>.jsonl``,
   ``p1_T<T>.jsonl``, ``summary.json`` and ``results.md``
   (:mod:`adaloop_common.evaluation`).

``eval.shard`` decides how several processes split the work: ``batch`` (process ``r`` of ``w``
takes batches ``r::w`` of every depth; same batch composition, hence the same outputs, as one
process) or ``depth`` (process ``r`` takes depths ``r::w``).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from typing import Any, Dict

logger = logging.getLogger("adaloop_llava.eval")

RESOLVED_EVAL_CONFIG = "resolved_eval_config.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adaloop-llava-eval",
                                description="Evaluate zero-shot / SFT / single-pass AdaLN / Variants A, B, C on VisRO.")
    p.add_argument("--config", required=True, help="evaluation config, e.g. adaloop/llava/configs/eval/variant_a.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, value parsed as YAML (repeatable), e.g. eval.splits=[ood]")
    return p


def setup_logging() -> None:
    rank = int(os.environ.get("RANK", "0"))
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(level=level, stream=sys.stdout,
                        format=f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s")
    import transformers
    transformers.utils.logging.set_verbosity_error()


def setup_distributed():
    """``(rank, world_size, device)``; initialises the process group under torchrun."""
    import torch
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if cuda else "gloo",
                                             timeout=datetime.timedelta(hours=6))
    return rank, world, device


def _barrier(world: int) -> None:
    import torch
    if world > 1:
        torch.distributed.barrier()


def load_split_examples(cfg) -> Dict[str, Any]:
    """The examples of every split in ``eval.splits``, checked by the gold self-check."""
    from adaloop_common.evaluation import load_eval_examples
    ev = cfg.eval
    out = {}
    for split in ev.splits:
        examples = load_eval_examples(ev.visro_root, ev.tasks, split, per_cell=ev.per_cell)
        logger.info("[eval] split %s: %d items of %s (per_cell=%s); every reference answer scores as correct",
                    split, len(examples), ev.tasks, ev.per_cell or "all")
        out[split] = examples
    return out


def check_depths(cfg, loop_mode: str) -> None:
    ev = cfg.eval
    if loop_mode == "none" and (ev.max_depth != 1 or ev.p1_max_depth):
        raise SystemExit("The checkpoint has no loop tower (loop mode 'none'): eval.max_depth must be 1 "
                         f"and eval.p1_max_depth 0, got {ev.max_depth} and {ev.p1_max_depth}")


def verify_portable_adapter(cfg) -> bool:
    """When ``eval.adapter`` is a portable checkpoint (it has ``adaloop_manifest.json``), verify its
    files, hashes and cross-checks and the base and CLIP directories of ``paths`` against its
    manifest (:class:`adaloop_llava.checkpoint.LlavaCheckpointLoader`). Returns whether it was one."""
    from ..checkpoint import MANIFEST_FILE, LlavaCheckpointLoader
    adapter = cfg.eval.adapter
    if not adapter or not os.path.isfile(os.path.join(adapter, MANIFEST_FILE)):
        return False
    verified = LlavaCheckpointLoader.verify(adapter, verify_hashes=True)
    LlavaCheckpointLoader.verify_external(verified, base_model=cfg.paths.base_model,
                                          vision_tower=cfg.paths.vision_tower)
    logger.info("[eval] portable checkpoint %s verified (variant %s)", adapter, verified.manifest.variant)
    return True


def run(cfg) -> Dict[str, Any]:
    """Run an evaluation config; returns ``{split: {setting: summary}}`` (complete on rank 0)."""
    import torch

    from adaloop_common.evaluation import build_prompt, score_answers, write_report, write_split

    from ..config import dump_config
    from ..eval.generate import run_depth
    from ..model.loading import load_for_eval
    from ..model.loop_utils import loop_config_value

    ev = cfg.eval
    rank, world, device = setup_distributed()
    split_examples = load_split_examples(cfg)
    verify_portable_adapter(cfg)

    tokenizer, model, image_processor = load_for_eval(
        cfg.paths.base_model, ev.adapter, cfg.paths.vision_tower, dtype=getattr(torch, ev.dtype),
        device=device, attn_implementation=ev.attn_implementation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token          # as in training
    loop_mode = model.config.loop_mode
    check_depths(cfg, loop_mode)
    condition_max_length = int(loop_config_value(model.config, "condition_max_length")) if loop_mode != "none" else None
    depths = list(range(1, ev.max_depth + 1))
    logger.info("[eval] %s: loop mode %s, depths %s, p=1 budget %s, %d process(es), shard=%s", ev.name, loop_mode,
                depths, ev.p1_max_depth or "-", world, ev.shard)
    if rank == 0:
        os.makedirs(ev.output_dir, exist_ok=True)
        dump_config(cfg, os.path.join(ev.output_dir, RESOLVED_EVAL_CONFIG))

    if ev.shard == "depth":
        my_depths, gen_rank, gen_world = depths[rank::world], 0, 1
    else:
        my_depths, gen_rank, gen_world = depths, rank, world

    summaries: Dict[str, Any] = {}
    for split, examples in split_examples.items():
        queries = [{"question": ex.question, "prompt": build_prompt(ex.record, ev.prompt_style), "image": ex.image}
                   for ex in examples]
        answers = {d: run_depth(model, tokenizer, image_processor, queries, d, ev, rank=gen_rank,
                                world_size=gen_world, device=device) for d in my_depths}
        if ev.shard == "depth" and world > 1:
            gathered: list = [None] * world
            torch.distributed.all_gather_object(gathered, answers)
            answers = {d: texts for part in gathered for d, texts in part.items()}
        if rank == 0:
            rows = {d: score_answers(examples, answers[d], depth=d) for d in depths}
            summaries[split] = write_split(ev.output_dir, split, rows, p1_max_depth=ev.p1_max_depth or None)
        _barrier(world)
    if rank == 0:
        table = write_report(ev.output_dir, summaries, {
            "backend": "llava-1.5", "model": ev.adapter or cfg.paths.base_model, "base_model": cfg.paths.base_model,
            "tasks": ev.tasks, "prompt_style": ev.prompt_style, "depths": depths,
            "p1_max_depth": ev.p1_max_depth or None,
            "generation": {"batch_size": ev.batch_size, "max_new_tokens": ev.max_new_tokens, "do_sample": False,
                           "dtype": ev.dtype, "attn_implementation": ev.attn_implementation,
                           "condition_max_length": condition_max_length, "shard": ev.shard,
                           "world_size": world}})
        logger.info("[eval] results (%s):\n%s", ev.output_dir, table)
    if world > 1:
        torch.distributed.destroy_process_group()
    return summaries


def resolve_config(args: argparse.Namespace):
    """The validated evaluation config of a parsed command line."""
    from ..config import load_eval_config
    return load_eval_config(args.config, args.overrides)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    setup_logging()
    run(cfg)


if __name__ == "__main__":
    main()
