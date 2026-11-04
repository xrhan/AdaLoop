"""Per-depth greedy generation for the VisRO evaluation.

:func:`run_depth` answers a list of queries at one depth in batches of ``eval.batch_size``, in
order; each batch goes through :func:`adaloop_llava.generation.generate_ids` (prompt, padding,
question condition, images and decoding settings are described there), and the answers are
decoded with ``skip_special_tokens=True`` and stripped.

With several processes (``eval.shard: batch``) process ``r`` of ``w`` generates batches
``r, r + w, ...`` of the same partition, and the answers are gathered back into query order; every
batch has the same composition as in a single process, so the outputs are the same.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..generation import generate_ids
from ..model.loop_utils import loop_config_value

logger = logging.getLogger(__name__)


def _dist_info(rank: Optional[int], world_size: Optional[int]) -> Tuple[int, int]:
    dist = torch.distributed
    initialized = dist.is_available() and dist.is_initialized()
    if world_size is None:
        world_size = dist.get_world_size() if initialized else 1
    if rank is None:
        rank = dist.get_rank() if initialized else 0
    if world_size > 1 and not initialized:
        raise RuntimeError("world_size > 1 needs an initialised torch.distributed process group")
    return int(rank), int(world_size)


def run_depth(model, tokenizer, image_processor, queries: Sequence[Dict[str, Any]], depth: int, cfg=None, *,
              rank: Optional[int] = None, world_size: Optional[int] = None, device=None,
              log: Callable[[str], None] = logger.info) -> List[str]:
    """Greedy answers (stripped) to ``queries`` at loop depth ``depth``, in query order.

    ``cfg`` is the ``eval`` section of an evaluation config (``batch_size``, ``max_new_tokens``;
    defaults if None). Loop models get the question condition at the checkpoint's
    ``condition_max_length``. ``rank`` / ``world_size`` (default: the torch.distributed group, if
    any) split the batches across processes; every process returns all answers.
    """
    from ..config import EvalRunConfig
    cfg = cfg if cfg is not None else EvalRunConfig()
    if getattr(cfg, "do_sample", False):
        raise ValueError("only greedy decoding is supported (eval.do_sample must be false)")
    rank, world_size = _dist_info(rank, world_size)
    model.eval()
    spec = getattr(model.get_vision_tower(), "loop_spec", None)
    is_loop = spec is not None
    depth = int(depth)
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if not is_loop:
        # A model without a loop tower has no depth: every depth would give the same answers.
        if depth != 1:
            raise ValueError(f"This model has no loop tower; only depth 1 is meaningful, got depth={depth}")
        log("[eval] model without a loop tower (plain CLIP); depth does not apply")
    else:
        log(f"[eval] depth={depth} loop_spec={spec}")
    condition_max_length = int(loop_config_value(model.config, "condition_max_length"))
    if device is None:
        device = next(model.parameters()).device
    bs = int(cfg.batch_size)
    batches = [list(queries[i:i + bs]) for i in range(0, len(queries), bs)]
    mine = list(range(rank, len(batches), world_size))
    log(f"[eval] {len(queries)} queries, {len(batches)} batches of {bs}; this process generates {len(mine)}"
        + (f" (rank {rank} of {world_size})" if world_size > 1 else ""))

    t0 = time.time()
    done: List[Tuple[int, List[str]]] = []
    n_done = 0
    for step, b in enumerate(mine):
        out = generate_ids(model, tokenizer, image_processor, batches[b], depth, is_loop=is_loop, device=device,
                           max_new_tokens=int(cfg.max_new_tokens), condition_max_length=condition_max_length)
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        done.append((b, [t.strip() for t in texts]))
        n_done += len(texts)
        if step % 20 == 0:
            log(f"[eval] depth={depth} {n_done}/{sum(len(batches[x]) for x in mine)} "
                f"{(time.time() - t0) / max(1, n_done):.2f}s/query")

    if world_size > 1:
        gathered: List[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered, done)
        done = [x for part in gathered for x in part]
    done.sort(key=lambda x: x[0])
    if [b for b, _ in done] != list(range(len(batches))):
        raise RuntimeError(f"gathered batches {[b for b, _ in done][:8]}... do not cover the {len(batches)} batches")
    answers = [t for _, part in done for t in part]
    assert len(answers) == len(queries)
    n_empty = sum(1 for t in answers if not t)
    log(f"[eval] depth={depth}: {len(answers)} answers ({n_empty} empty), wall {time.time() - t0:.0f}s")
    return answers
