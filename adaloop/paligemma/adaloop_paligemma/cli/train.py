"""Train a paper PaliGemma AdaLoop recipe on VisRO, optionally with a replay JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from adaloop_common.data import (
    add_training_data_arguments,
    image_content_fingerprint,
    text_content_fingerprint,
    training_set_from_args,
)

from ..config import PaliGemmaAdaLoopConfig
from ..data import iter_epoch_batches
from ..training import (
    TrainingState,
    build_optimizer,
    build_train_step,
    load_training_checkpoint,
    replicate_to_local_devices,
    save_training_checkpoint,
    trainable_parameter_counts,
    unreplicate_from_local_devices,
)
from . import add_matmul_precision_argument, apply_matmul_precision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Strict paper recipe YAML")
    parser.add_argument("--base-checkpoint", required=True, help="Pinned Big Vision NPZ")
    parser.add_argument("--tokenizer", required=True, help="Pinned SentencePiece model")
    add_training_data_arguments(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--export-dir",
        help=(
            "Optional compact public checkpoint destination after training "
            "(checked before training starts); the exact frozen-base proof is "
            "run before publication"
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Training snapshot directory, or 'latest' under --output-dir",
    )
    add_matmul_precision_argument(parser)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="Retain this many numbered snapshots; 0 keeps all",
    )
    parser.add_argument(
        "--jax-distributed",
        action="store_true",
        help=(
            "Join a multi-process (multi-host GPU or TPU) run with "
            "jax.distributed.initialize() before any JAX computation; Cloud "
            "TPU, SLURM, and Open MPI settings are auto-detected (Kubernetes "
            "needs jax[k8s]); otherwise pass --coordinator-address, "
            "--num-processes, and --process-id"
        ),
    )
    parser.add_argument(
        "--coordinator-address",
        help="With --jax-distributed: coordinator host:port for manual launches",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        help="With --jax-distributed: total process count for manual launches",
    )
    parser.add_argument(
        "--process-id",
        type=int,
        help="With --jax-distributed: this process's index for manual launches",
    )
    return parser


_DISTRIBUTED_OPTIONS = ("coordinator_address", "num_processes", "process_id")


def _prepare_output(path: str | Path, *, resume: str | None) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise SystemExit(f"--output-dir must not be a symlink: {requested}")
    root = requested.resolve()
    if root.exists() and not root.is_dir():
        raise SystemExit(f"--output-dir is not a directory: {root}")
    if resume is None and root.is_dir() and next(root.iterdir(), None) is not None:
        raise SystemExit(
            f"Refusing a fresh run in non-empty {root}; use an empty directory "
            "or --resume-from-checkpoint"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_resume(value: str | None, root: Path) -> Path | None:
    if value is None:
        return None
    if value != "latest":
        result = Path(value).expanduser().resolve()
        if not result.is_dir():
            raise SystemExit(f"Resume checkpoint is not a directory: {result}")
        return result
    candidates: list[tuple[int, Path]] = []
    for child in root.glob("checkpoint-*"):
        if not child.is_dir():
            continue
        try:
            step = int(child.name.removeprefix("checkpoint-"))
            state = TrainingState.from_file(child / "training_state.json")
        except (ValueError, OSError):
            continue
        if state.global_step == step and (child / "COMPLETE").is_file():
            candidates.append((step, child))
    if not candidates:
        raise SystemExit(f"No complete training snapshot exists under {root}")
    return max(candidates)[1]


def _check_export_dir(path: str | Path, *, output_dir: Path) -> None:
    """Apply the final export's destination rules before any training step.

    Nothing is left behind: a failed startup must not make an identical re-run
    find a non-empty --output-dir (the export path may be nested inside it).
    """

    # The checkpoint layer owns the rules; import it lazily like the export.
    from ..checkpoint import check_export_destination
    from ..errors import CheckpointValidationError

    root = Path(path).expanduser().resolve()
    if root == output_dir or root in output_dir.parents:
        raise SystemExit(f"--export-dir must not be or contain --output-dir: {root}")
    try:
        check_export_destination(path)
    except (CheckpointValidationError, OSError) as exc:
        raise SystemExit(f"Invalid --export-dir {path}: {exc}") from exc


def _initialize_distributed(jax: Any, args: argparse.Namespace) -> bool:
    """Join the multi-process JAX runtime; a no-op unless requested."""

    if not args.jax_distributed:
        return False
    jax.distributed.initialize(
        **{
            name: getattr(args, name)
            for name in _DISTRIBUTED_OPTIONS
            if getattr(args, name) is not None
        }
    )
    return True


def _shutdown_distributed(jax: Any) -> None:
    # A shutdown failure must not mask the run's own outcome or exception.
    try:
        jax.distributed.shutdown()
    except Exception as exc:
        print(f"Ignoring JAX distributed shutdown error: {exc!r}", file=sys.stderr)


def _reshape_local_batch(batch: dict[str, np.ndarray], local_devices: int):
    result = {}
    for name, value in batch.items():
        if value.shape[0] % local_devices:
            raise ValueError(
                f"Batch field {name} has {value.shape[0]} rows, not divisible "
                f"by {local_devices} local devices"
            )
        per_device = value.shape[0] // local_devices
        result[name] = value.reshape(local_devices, per_device, *value.shape[1:])
    return result


def _epoch_iterator(
    examples: Any,
    *,
    runtime: Any,
    recipe: PaliGemmaAdaLoopConfig,
    process_batch_size: int,
    epoch: int,
    skip_batches: int,
    process_index: int,
    process_count: int,
) -> Iterator[dict[str, np.ndarray]]:
    # Resume skips consumed batches by index, without decoding their images.
    return iter_epoch_batches(
        examples,
        tokenizer=runtime.tokenizer,
        batch_size=process_batch_size,
        max_text_len=int(recipe.preprocessing.max_text_tokens),
        seed=int(recipe.training.seed),
        epoch=epoch,
        process_index=process_index,
        process_count=process_count,
        drop_remainder=True,
        image_size=int(recipe.preprocessing.image_size),
        prefix_template=recipe.preprocessing.prompt_template,
        include_condition=recipe.conditioner.mode != "none",
        condition_question=(
            recipe.conditioner.constant_text
            if recipe.conditioner.mode == "constant_question"
            else None
        ),
        skip_batches=skip_batches,
    )


def _run_contract(
    *,
    training_set: Any,
    image_content_sha256: str,
    recipe: PaliGemmaAdaLoopConfig,
    jax: Any,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "data": {
            # The selection and the ordered contents, independent of paths:
            # sample ids with their question and answer, and their image bytes.
            **training_set.describe(),
            "text_content_sha256": text_content_fingerprint(training_set.examples),
            "image_content_sha256": image_content_sha256,
        },
        "distributed": {
            "process_count": int(jax.process_count()),
            "device_count": int(jax.device_count()),
            "local_device_count": int(jax.local_device_count()),
        },
        "external_assets": {
            "base_checkpoint_sha256": recipe.base_model.sha256,
            "base_checkpoint_size_bytes": recipe.base_model.size_bytes,
            "tokenizer_sha256": recipe.tokenizer.sha256,
            "tokenizer_size_bytes": recipe.tokenizer.size_bytes,
        },
    }


def _global_image_fingerprint(examples: Any, jax: Any) -> str:
    """Hash images once on process zero and broadcast the fixed-size digest."""

    from jax.experimental import multihost_utils

    if jax.process_index() == 0:
        raw = bytes.fromhex(image_content_fingerprint(examples))
        payload = np.frombuffer(raw, dtype=np.uint8).copy()
    else:
        payload = np.zeros(32, dtype=np.uint8)
    payload = np.asarray(multihost_utils.broadcast_one_to_all(payload), np.uint8)
    if payload.shape != (32,):
        raise RuntimeError(f"Image fingerprint broadcast has shape {payload.shape}")
    return payload.tobytes().hex()


def _save(
    *,
    output_dir: Path,
    params: Any,
    opt_state: Any,
    state: TrainingState,
    recipe: PaliGemmaAdaLoopConfig,
    contract: dict[str, Any],
    jax: Any,
    keep: int,
) -> None:
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices(f"before-save-{state.global_step}")
    if jax.process_index() == 0:
        target = output_dir / f"checkpoint-{state.global_step:08d}"
        save_training_checkpoint(
            target,
            params=unreplicate_from_local_devices(params),
            opt_state=unreplicate_from_local_devices(opt_state),
            state=state,
            config_yaml=recipe.to_yaml(),
            run_contract=contract,
        )
        if keep > 0:
            complete = sorted(
                child
                for child in output_dir.glob("checkpoint-*")
                if child.is_dir() and (child / "COMPLETE").is_file()
            )
            for obsolete in complete[:-keep]:
                # Checkpoint retention is intentionally narrow and never
                # touches directories that do not match our own exact prefix.
                import shutil

                shutil.rmtree(obsolete)
    multihost_utils.sync_global_devices(f"after-save-{state.global_step}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("save_steps", "log_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.keep_checkpoints < 0:
        parser.error("--keep-checkpoints must be non-negative")
    if not args.jax_distributed and any(
        getattr(args, name) is not None for name in _DISTRIBUTED_OPTIONS
    ):
        parser.error(
            "--coordinator-address, --num-processes, and --process-id "
            "require --jax-distributed"
        )
    if args.num_processes is not None and args.num_processes < 1:
        parser.error("--num-processes must be positive")
    if args.process_id is not None and not (
        0 <= args.process_id < (args.num_processes or float("inf"))
    ):
        parser.error("--process-id must lie in [0, --num-processes)")

    try:
        import jax
    except ImportError as exc:
        raise SystemExit(
            "Training requires JAX/Flax/Optax; install adaloop-paligemma[runtime]"
        ) from exc

    # jax.distributed.initialize() must precede every call that initializes a
    # JAX backend, including the model-stack import and device queries below.
    distributed = _initialize_distributed(jax, args)
    try:
        _train(args, jax)
    finally:
        if distributed:
            _shutdown_distributed(jax)


def _train(args: argparse.Namespace, jax: Any) -> None:
    # Importing the model runtime initializes the optional Flax/JAX model
    # stack.  Keep it after argument parsing so ``--help`` remains usable in
    # metadata-only environments.
    from ..runtime import initialize_runtime

    matmul_precision = apply_matmul_precision(jax, args.matmul_precision)
    recipe = PaliGemmaAdaLoopConfig.from_file(args.config)
    output_dir = _prepare_output(args.output_dir, resume=args.resume_from_checkpoint)
    resume = _resolve_resume(args.resume_from_checkpoint, output_dir)
    training_set = training_set_from_args(args)
    examples = training_set.examples
    image_fingerprint = _global_image_fingerprint(examples, jax)
    if args.export_dir is not None:
        # The fingerprint broadcast is the first collective: every host has
        # passed _prepare_output, so the probe cannot fail a fresh-run check.
        _check_export_dir(args.export_dir, output_dir=output_dir)
    global_batch = int(recipe.training.global_batch_size)
    if global_batch % jax.device_count():
        raise SystemExit(
            f"training.global_batch_size={global_batch} must be divisible by "
            f"global JAX devices={jax.device_count()}"
        )
    if global_batch % jax.process_count():
        raise SystemExit("Global batch must be divisible by the JAX process count")
    process_batch = global_batch // jax.process_count()
    if process_batch % jax.local_device_count():
        raise SystemExit("Per-process batch must be divisible by local JAX devices")

    runtime = initialize_runtime(recipe, args.base_checkpoint, args.tokenizer)
    tx, labels = build_optimizer(runtime.params, recipe)
    opt_state_template = tx.init(runtime.params)
    contract = _run_contract(
        training_set=training_set,
        image_content_sha256=image_fingerprint,
        recipe=recipe,
        jax=jax,
    )
    if resume is None:
        host_params = runtime.params
        host_opt_state = opt_state_template
        host_state = TrainingState(
            global_step=0,
            epoch=0,
            batches_consumed_in_epoch=0,
            rng_key=tuple(
                int(value)
                for value in np.asarray(
                    jax.random.PRNGKey(int(recipe.training.seed)), dtype=np.uint32
                )
            ),
        )
    else:
        host_params, host_opt_state, host_state = load_training_checkpoint(
            resume,
            params_template=runtime.params,
            opt_state_template=opt_state_template,
            expected_config_yaml=recipe.to_yaml(),
            expected_run_contract=contract,
        )

    counts = trainable_parameter_counts(host_params)
    if jax.process_index() == 0:
        print(
            json.dumps(
                {
                    "event": "start",
                    "variant": recipe.variant,
                    "global_step": host_state.global_step,
                    "global_devices": jax.device_count(),
                    "global_batch_size": global_batch,
                    "matmul_precision": matmul_precision,
                    "parameters": counts,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    params = replicate_to_local_devices(host_params)
    opt_state = replicate_to_local_devices(host_opt_state)
    step_fn = jax.pmap(
        build_train_step(runtime.model, tx, labels, recipe),
        axis_name="batch",
        donate_argnums=(0, 1),
    )
    state = host_state
    iterator = _epoch_iterator(
        examples,
        runtime=runtime,
        recipe=recipe,
        process_batch_size=process_batch,
        epoch=state.epoch,
        skip_batches=state.batches_consumed_in_epoch,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )
    started = time.monotonic()
    while state.global_step < int(recipe.training.steps):
        try:
            host_batch = next(iterator)
        except StopIteration:
            state = replace(state, epoch=state.epoch + 1, batches_consumed_in_epoch=0)
            iterator = _epoch_iterator(
                examples,
                runtime=runtime,
                recipe=recipe,
                process_batch_size=process_batch,
                epoch=state.epoch,
                skip_batches=0,
                process_index=jax.process_index(),
                process_count=jax.process_count(),
            )
            try:
                host_batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "Dataset has fewer host-sharded examples than one global batch"
                ) from exc

        batch = {
            name: jax.device_put(value)
            for name, value in _reshape_local_batch(
                host_batch, jax.local_device_count()
            ).items()
        }
        base_key = jax.random.PRNGKey(np.uint32(state.rng_key[0]))
        base_key = jax.random.fold_in(base_key, np.uint32(state.rng_key[1]))
        base_key = jax.random.fold_in(base_key, state.global_step)
        global_keys = jax.random.split(base_key, jax.device_count())
        offset = jax.process_index() * jax.local_device_count()
        local_keys = global_keys[offset : offset + jax.local_device_count()]
        params, opt_state, metrics = step_fn(params, opt_state, batch, local_keys)
        state = replace(
            state,
            global_step=state.global_step + 1,
            batches_consumed_in_epoch=state.batches_consumed_in_epoch + 1,
        )
        if state.global_step % args.log_steps == 0 and jax.process_index() == 0:
            host_metrics = unreplicate_from_local_devices(metrics)
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": state.global_step,
                        "epoch": state.epoch,
                        "elapsed_seconds": time.monotonic() - started,
                        **{name: float(value) for name, value in host_metrics.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if state.global_step % args.save_steps == 0:
            _save(
                output_dir=output_dir,
                params=params,
                opt_state=opt_state,
                state=state,
                recipe=recipe,
                contract=contract,
                jax=jax,
                keep=args.keep_checkpoints,
            )

    if state.global_step % args.save_steps:
        _save(
            output_dir=output_dir,
            params=params,
            opt_state=opt_state,
            state=state,
            recipe=recipe,
            contract=contract,
            jax=jax,
            keep=args.keep_checkpoints,
        )
    if jax.process_index() == 0:
        if args.export_dir is not None:
            from ..checkpoint import export_parameter_tree
            from ..runtime import reconstruct_parameter_template

            export_parameter_tree(
                unreplicate_from_local_devices(params),
                recipe,
                args.base_checkpoint,
                args.tokenizer,
                args.export_dir,
                reconstruction_hook=reconstruct_parameter_template,
            )
        print(
            json.dumps({"event": "complete", "step": state.global_step}),
            flush=True,
        )
    if args.export_dir is not None:
        from jax.experimental import multihost_utils

        # Keep worker processes alive until host zero has atomically published
        # and re-verified the public overlay.
        multihost_utils.sync_global_devices("after-final-export")


if __name__ == "__main__":  # pragma: no cover
    main()
