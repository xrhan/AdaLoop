"""Train a Qwen2.5-VL AdaLoop recipe on VisRO, optionally with a replay JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from adaloop_common.data import add_training_data_arguments, training_set_from_args

from ..config import CONDITION_MAX_TOKENS, QwenAdaLoopConfig

# As in the paper runs (--dataloader_drop_last True), each epoch stops after its last
# round of full per-device batches on every process instead of padding that
# round with samples from the start of the epoch.  HF Trainer also shortened an
# epoch's last update when its batches did not fill it; here accumulation
# windows may span epochs instead, so with gradient accumulation an update that
# crosses an epoch boundary can contain a row twice, and with fewer rows than
# training.global_batch_size every update crosses one.
_DROP_LAST = True
# The paper runs' HF Trainer initialized the process group with its default
# ddp_timeout of 1800 seconds instead of NCCL's 10 minutes.  Collectives may
# wait that long, e.g. while the main process hashes every training image.
_PROCESS_GROUP_TIMEOUT = timedelta(seconds=1800)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Strict AdaLoop recipe YAML")
    add_training_data_arguments(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Accelerate checkpoint directory, or 'latest' under --output-dir",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Defaults to the value implied by training.global_batch_size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help=(
            "Override preprocessing.max_length (paper recipes: 2048 for "
            "SFT/AdaLN/A/B, 4096 for Variant C)"
        ),
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        help="Override preprocessing.min_pixels (paper default: 12544)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        help="Override preprocessing.max_pixels (paper default: 451584)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help=(
            "Keep only the newest N complete periodic checkpoint-* directories "
            "(the paper runs kept 2) and delete older or incomplete ones; 0 "
            "keeps all. The final checkpoint in --output-dir is never deleted"
        ),
    )
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional override for training.seed in the recipe",
    )
    parser.add_argument(
        "--constant-question",
        help=(
            "Optional override for conditioner.constant_text when "
            "conditioner.mode=constant_question; any text other than the "
            "recipe's departs from the paper's parameter-matched control"
        ),
    )
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("bf16", "no"),
        help=(
            "Optional override for training.precision (bf16 in every release "
            "recipe); use 'no' only for CPU/debug execution, which trains in "
            "FP32 and records training.precision: fp32"
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for training.gradient_checkpointing",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    return parser


def _positive(name: str, value: int) -> None:
    if value < 1:
        raise SystemExit(f"{name} must be a positive integer")


def _prepare_training_output_dir(
    value: str | Path, *, resume_from_checkpoint: str | None
) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise SystemExit(f"--output-dir must not be a symlink: {requested}")
    root = requested.resolve()
    if root.exists() and not root.is_dir():
        raise SystemExit(f"--output-dir is not a directory: {root}")
    if (
        resume_from_checkpoint is None
        and root.is_dir()
        and next(root.iterdir(), None) is not None
    ):
        raise SystemExit(
            f"Refusing to start a fresh run in non-empty --output-dir {root}; "
            "choose an empty directory or pass --resume-from-checkpoint"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_resume(value: str | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    if value != "latest":
        result = Path(value).expanduser().resolve()
        if not result.is_dir():
            raise SystemExit(f"Resume checkpoint is not a directory: {result}")
        return result
    from ..training import (
        TRAINING_CONFIG,
        TRAINING_RUN_CONTRACT,
        TRAINING_STATE,
        TrainingState,
    )

    def warn(message: str) -> None:
        print(f"Warning: {message}", file=sys.stderr, flush=True)

    required = (TRAINING_STATE, TRAINING_CONFIG, TRAINING_RUN_CONTRACT)
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.glob("checkpoint-*"):
        try:
            step = int(child.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        # Pruning keeps checkpoints it cannot inspect or verify; skip them
        # with a warning instead of aborting the resume.
        try:
            if not child.is_dir():
                continue
            status = _resume_checkpoint_status(child, step)
        except OSError as exc:
            warn(f"skipping checkpoint {child}: cannot inspect it ({exc})")
            continue
        if status is None:
            warn(f"skipping checkpoint {child}: cannot verify its {TRAINING_STATE}")
        elif status:
            candidates.append((step, child))
    try:
        root_complete = all((output_dir / name).is_file() for name in required)
    except OSError as exc:
        warn(f"skipping checkpoint {output_dir}: cannot inspect it ({exc})")
        root_complete = False
    if root_complete:
        try:
            root_state = TrainingState.from_file(output_dir / TRAINING_STATE)
        except ValueError:
            warn(f"skipping checkpoint {output_dir}: cannot verify its {TRAINING_STATE}")
        else:
            candidates.append((root_state.global_step, output_dir))
    if not candidates:
        raise SystemExit(f"No resumable checkpoint exists under {output_dir}")
    return max(
        candidates,
        key=lambda item: (item[0], item[1] == output_dir),
    )[1]


def _resume_checkpoint_status(path: Path, step: int) -> bool | None:
    """Return True if complete, False if a required file is absent, else None.

    None means the files exist but the completion marker is unreadable,
    invalid, or records another step.  OSErrors other than a missing file,
    such as permission errors, propagate.
    """

    from ..training import (
        TRAINING_CONFIG,
        TRAINING_RUN_CONTRACT,
        TRAINING_STATE,
        TrainingState,
    )

    for name in (TRAINING_STATE, TRAINING_CONFIG, TRAINING_RUN_CONTRACT):
        try:
            mode = (path / name).stat().st_mode
        except (FileNotFoundError, NotADirectoryError):
            return False
        if not stat.S_ISREG(mode):
            return False
    try:
        state = TrainingState.from_file(path / TRAINING_STATE)
    except ValueError:
        return None
    return True if state.global_step == step else None


def _prune_resume_checkpoints(
    output_dir: Path, *, keep: int, current_step: int
) -> None:
    """Delete periodic checkpoints older than the newest ``keep`` complete ones.

    Only canonical ``checkpoint-<step>`` directories up to ``current_step`` are
    considered, so the root checkpoint and checkpoints from later steps (for
    example, left by an earlier run that this one restarted from) are never
    removed.  Directories missing a required file are incomplete and removed
    too.  A directory that cannot be inspected, or whose completion marker is
    present but unreadable, is kept; filesystem errors only print a warning,
    so pruning never stops training.  ``keep=0`` disables pruning.
    """

    if keep < 1:
        return
    from ..training import TRAINING_STATE

    def warn(message: str) -> None:
        print(f"Warning: {message}", file=sys.stderr, flush=True)

    try:
        children = list(output_dir.glob("checkpoint-*"))
    except OSError as exc:
        warn(f"skipping checkpoint pruning in {output_dir}: {exc}")
        return
    complete: list[tuple[int, Path]] = []
    obsolete: list[Path] = []
    for child in children:
        try:
            step = int(child.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if child.name != f"checkpoint-{step}" or step > current_step:
            continue
        try:
            if child.is_symlink() or not child.is_dir():
                continue
            status = _resume_checkpoint_status(child, step)
        except OSError as exc:
            warn(f"keeping checkpoint {child}: cannot inspect it ({exc})")
            continue
        if status is None:
            warn(f"keeping checkpoint {child}: cannot verify its {TRAINING_STATE}")
        elif status:
            complete.append((step, child))
        else:
            obsolete.append(child)
    complete.sort()
    obsolete.extend(path for _step, path in complete[:-keep])
    for path in obsolete:
        # Remove the completion marker first so a partially deleted directory
        # is never selected by ``--resume-from-checkpoint latest``.
        try:
            (path / TRAINING_STATE).unlink(missing_ok=True)
        except OSError as exc:
            warn(f"keeping checkpoint {path}: cannot delete it ({exc})")
            continue
        shutil.rmtree(path, ignore_errors=True)
        if os.path.lexists(path):
            warn(f"checkpoint {path} was only partially deleted")


def _epoch_sampler(torch: Any, dataset: Any, seed: int) -> Any:
    class _EpochRandomSampler(torch.utils.data.Sampler):
        def __init__(self) -> None:
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def __iter__(self) -> Iterator[int]:
            generator = torch.Generator()
            generator.manual_seed(int(seed) + self.epoch)
            yield from torch.randperm(len(dataset), generator=generator).tolist()

        def __len__(self) -> int:
            return len(dataset)

    return _EpochRandomSampler()


def _load_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from accelerate import Accelerator
        from accelerate.utils import (
            GradientAccumulationPlugin,
            InitProcessGroupKwargs,
            broadcast_object_list,
        )
        from torch.utils.data import DataLoader
        from transformers import AutoProcessor, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are unavailable; install adaloop-qwen[gpu]"
        ) from exc
    return (
        torch,
        Accelerator,
        GradientAccumulationPlugin,
        DataLoader,
        (AutoProcessor, set_seed),
        (InitProcessGroupKwargs, broadcast_object_list),
    )


def _qualified_class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _global_image_fingerprint(
    examples: Sequence[Any],
    accelerator: Any,
    broadcast_object_list: Callable[..., list[Any]],
) -> str:
    """Hash every image once on the main process and broadcast the digest.

    The other processes wait in the broadcast.  A fingerprinting error is
    broadcast in place of the digest, so a missing or unreadable image stops
    every process instead of leaving the others blocked.
    """

    from adaloop_common.data import image_content_fingerprint

    payload: list[Any] = [None]
    if accelerator.is_main_process:
        try:
            payload = [("sha256", image_content_fingerprint(examples))]
        except ValueError as exc:
            payload = [("error", str(exc))]
    kind, value = broadcast_object_list(payload, from_process=0)[0]
    if kind != "sha256":
        raise SystemExit(f"Could not fingerprint training images: {value}")
    return value


def _training_run_contract(
    processor: Any,
    config: QwenAdaLoopConfig,
    *,
    max_length: int,
    training_set: Any,
    image_content_sha256: str,
    attn_implementation: str,
    mixed_precision: str,
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_workers: int,
) -> dict[str, Any]:
    """Capture data, batching, and processor semantics outside the recipe."""

    image_processor = getattr(processor, "image_processor", None)
    serializer = getattr(image_processor, "to_dict", None)
    if image_processor is None or not callable(serializer):
        raise SystemExit(
            "The Qwen processor must expose a serializable image_processor"
        )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise SystemExit("The Qwen processor must expose a tokenizer")
    chat_template = getattr(processor, "chat_template", None)
    if chat_template is None:
        chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise SystemExit("The Qwen processor does not define a chat template")
    from adaloop_common.data import text_content_fingerprint

    payload = {
        "format_version": 3,
        "data": {
            # The selection and the ordered contents, independent of paths:
            # sample ids with their question and answer, and their image bytes.
            **training_set.describe(),
            "text_content_sha256": text_content_fingerprint(training_set.examples),
            "image_content_sha256": str(image_content_sha256),
        },
        "sequence": {
            "max_length": int(max_length),
            "padding_side": str(tokenizer.padding_side),
        },
        "runtime": {
            "attn_implementation": str(attn_implementation),
            "mixed_precision": str(mixed_precision),
            "world_size": int(world_size),
            "per_device_batch_size": int(per_device_batch_size),
            "gradient_accumulation_steps": int(
                gradient_accumulation_steps
            ),
            "num_workers": int(num_workers),
            "drop_last": _DROP_LAST,
        },
        "processor": {
            "source": config.base_model,
            "revision": config.base_revision,
            "class": _qualified_class_name(processor),
            "tokenizer_class": _qualified_class_name(tokenizer),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
            "image_processor_class": _qualified_class_name(image_processor),
            "image_processor_config": serializer(),
        },
    }
    try:
        # Normalize tuples/enums and reject non-finite or environment-specific
        # opaque objects before any training work begins.
        normalized = json.loads(
            json.dumps(payload, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "Qwen image preprocessing settings are not JSON-serializable"
        ) from exc
    return normalized


def _enable_cuda_tf32(torch: Any, device: Any) -> None:
    """Allow TF32 CUDA matmuls and convolutions, as the paper runs' ``--tf32 True``.

    Transformers set both backend flags for training only; the paper's
    separate evaluation processes kept PyTorch's defaults, and so does
    inference here.
    """

    if getattr(device, "type", None) == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _enable_nonreentrant_gradient_checkpointing(model: Any) -> None:
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise SystemExit(
            "training.gradient_checkpointing=true but the model does not expose "
            "gradient_checkpointing_enable()"
        )
    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError as exc:
        raise SystemExit(
            "This Transformers runtime cannot enable required non-reentrant "
            "gradient checkpointing; install the pinned adaloop-qwen[gpu] "
            "dependencies"
        ) from exc


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _positive("--per-device-batch-size", args.per_device_batch_size)
    if args.gradient_accumulation_steps is not None:
        _positive("--gradient-accumulation-steps", args.gradient_accumulation_steps)
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    if args.save_steps < 0 or args.log_steps < 1:
        raise SystemExit("--save-steps must be non-negative and --log-steps positive")
    if args.save_total_limit < 0:
        raise SystemExit("--save-total-limit must be non-negative")
    output_dir = _prepare_training_output_dir(
        args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    config = QwenAdaLoopConfig.from_file(args.config)
    max_length = (
        config.preprocessing.max_length
        if args.max_length is None
        else args.max_length
    )
    min_pixels = (
        config.preprocessing.min_pixels
        if args.min_pixels is None
        else args.min_pixels
    )
    max_pixels = (
        config.preprocessing.max_pixels
        if args.max_pixels is None
        else args.max_pixels
    )
    _positive("--max-length", max_length)
    _positive("--min-pixels", min_pixels)
    _positive("--max-pixels", max_pixels)
    if min_pixels > max_pixels:
        raise SystemExit("--min-pixels must not exceed --max-pixels")
    if config.conditioner.mode != "constant_question" and args.constant_question:
        raise SystemExit(
            "--constant-question is valid only when conditioner.mode=constant_question"
        )
    constant_condition = config.conditioner.constant_text
    if args.constant_question is not None:
        constant_condition = args.constant_question.strip()
        if not constant_condition:
            raise SystemExit("--constant-question must contain non-whitespace text")
    seed = config.training.seed if args.seed is None else args.seed
    if seed < 0:
        raise SystemExit("--seed must be non-negative")
    gradient_checkpointing = (
        config.training.gradient_checkpointing
        if args.gradient_checkpointing is None
        else args.gradient_checkpointing
    )
    mixed_precision = args.mixed_precision
    if mixed_precision is None:
        mixed_precision = "bf16" if config.training.precision == "bf16" else "no"
    # Persist explicit CLI overrides into both resume metadata and the final
    # canonical recipe; a checkpoint must never describe different training
    # semantics from those actually used.
    config = replace(
        config,
        conditioner=replace(
            config.conditioner, constant_text=constant_condition
        ),
        training=replace(
            config.training,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            precision="bf16" if mixed_precision == "bf16" else "fp32",
        ),
        preprocessing=replace(
            config.preprocessing,
            max_length=max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        ),
    )

    (
        torch,
        Accelerator,
        GradientAccumulationPlugin,
        DataLoader,
        transformers_bits,
        accelerate_utils,
    ) = _load_runtime_dependencies()
    AutoProcessor, set_seed = transformers_bits
    InitProcessGroupKwargs, broadcast_object_list = accelerate_utils
    from ..data import QwenDataCollator, QwenImageQADataset
    from ..model import AdaptiveQwen2_5_VLForConditionalGeneration
    from ..training import (
        TrainingState,
        apply_language_lora,
        build_adamw,
        build_cosine_scheduler,
        configure_trainable_parameters,
        load_resume_checkpoint,
        make_training_objective_module,
        register_trainable_only_resume_state,
        release_readout_objective_hooks,
        save_resume_checkpoint,
    )

    set_seed(seed)
    try:
        launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise SystemExit("WORLD_SIZE must be a positive integer") from exc
    if launched_world_size < 1:
        raise SystemExit("WORLD_SIZE must be a positive integer")
    denominator = args.per_device_batch_size * launched_world_size
    if args.gradient_accumulation_steps is None:
        if config.training.global_batch_size % denominator:
            raise SystemExit(
                "training.global_batch_size must be divisible by "
                "per-device batch size times process count: "
                f"{config.training.global_batch_size} % {denominator} != 0"
            )
        accumulation = config.training.global_batch_size // denominator
    else:
        accumulation = args.gradient_accumulation_steps
        effective = denominator * accumulation
        if effective != config.training.global_batch_size:
            raise SystemExit(
                "Explicit batch settings do not equal training.global_batch_size: "
                f"{denominator} * {accumulation} = {effective}, expected "
                f"{config.training.global_batch_size}"
            )
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=accumulation,
            sync_with_dataloader=False,
        ),
        mixed_precision=mixed_precision,
        # The schedule counts optimizer updates.  By default Accelerate steps
        # a prepared scheduler once per process at every update, compressing
        # it by the process count; the loop steps it once per update instead.
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=_PROCESS_GROUP_TIMEOUT)],
    )
    if accelerator.num_processes != launched_world_size:
        raise SystemExit(
            "Accelerate process count disagrees with WORLD_SIZE: "
            f"{accelerator.num_processes} != {launched_world_size}"
        )
    _enable_cuda_tf32(torch, accelerator.device)

    processor = AutoProcessor.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.model_max_length = max_length
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise SystemExit("The Qwen processor does not expose an image_processor")
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels
    include_condition = config.conditioner.mode != "none"
    include_trajectory = config.variant == "c"
    condition_max_tokens = config.conditioner.max_tokens or CONDITION_MAX_TOKENS
    training_set = training_set_from_args(args)
    dataset = QwenImageQADataset(
        training_set.examples,
        processor,
        include_condition=include_condition,
        condition_max_tokens=condition_max_tokens,
        constant_condition=constant_condition,
        include_trajectory=include_trajectory,
    )
    image_fingerprint = _global_image_fingerprint(
        dataset.examples, accelerator, broadcast_object_list
    )
    run_contract = _training_run_contract(
        processor,
        config,
        max_length=max_length,
        training_set=training_set,
        image_content_sha256=image_fingerprint,
        attn_implementation=args.attn_implementation,
        mixed_precision=mixed_precision,
        world_size=accelerator.num_processes,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=accumulation,
        num_workers=args.num_workers,
    )
    collator = QwenDataCollator(
        processor.tokenizer,
        max_length=max_length,
        include_condition=include_condition,
        include_trajectory=include_trajectory,
        readout_mode=config.readout.mode,
    )
    sampler = _epoch_sampler(torch, dataset, seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=_DROP_LAST,
        # Worker base seeds come from this generator instead of the global
        # RNG, so starting an epoch never shifts the RNG stream that a resume
        # checkpoint restores.
        generator=torch.Generator().manual_seed(seed),
    )

    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float32
    model = AdaptiveQwen2_5_VLForConditionalGeneration.from_base_pretrained(
        config.base_model,
        adaloop_config=config,
        revision=config.base_revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if gradient_checkpointing:
        _enable_nonreentrant_gradient_checkpointing(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # As in the paper runs, re-seed immediately before creating LoRA so its
    # initialization does not depend on how many random draws the optional
    # AdaLN/conditioner modules consumed; same-seed recipes stay paired.
    set_seed(seed)
    model = apply_language_lora(model, config.lora)
    report = configure_trainable_parameters(
        model,
        train_merger=config.training.train_merger,
        train_conditioner=include_condition,
        require_lora=True,
    )
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    conditioner = getattr(base_model, "question_conditioner", None)
    if conditioner is not None:
        conditioner.float()
    optimizer = build_adamw(model, config.training.optimizer)
    lr_scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=config.training.steps,
        warmup_steps=config.training.warmup_steps,
        min_lr_ratio=config.training.min_lr_ratio,
    )
    objective_hooks = (
        release_readout_objective_hooks(
            max_length=max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        if config.variant in {"a", "b", "c"}
        else None
    )
    training_module = make_training_objective_module(
        model, config, hooks=objective_hooks
    )
    # from_pretrained leaves the model in eval mode, where gradient
    # checkpointing is inactive; the paper runs' HF Trainer trained in train mode.
    training_module.train()
    training_module, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        training_module, optimizer, dataloader, lr_scheduler
    )
    if len(dataloader) == 0:
        raise SystemExit(
            "Training dataset produced no batches: a full batch on every "
            "process needs at least "
            f"{args.per_device_batch_size * accelerator.num_processes} samples; "
            f"got {len(dataset)}"
        )
    # Resume snapshots store only trainable tensors; the frozen base is rebuilt
    # from the pinned revision.  save/load_resume_checkpoint register this too.
    register_trainable_only_resume_state(accelerator)

    resume_dir = _resolve_resume(args.resume_from_checkpoint, output_dir)
    state = TrainingState()
    if resume_dir is not None:
        state = load_resume_checkpoint(
            accelerator,
            resume_dir,
            expected_config=config,
            expected_run_contract=run_contract,
        )
        if state.global_step > config.training.steps:
            raise SystemExit(
                f"Checkpoint step {state.global_step} exceeds configured "
                f"training.steps={config.training.steps}"
            )
        if state.global_step == config.training.steps:
            accelerator.print(
                f"Checkpoint is already at step {state.global_step}; "
                "finalizing the canonical release export"
            )

    accelerator.print(
        "AdaLoop training: "
        f"variant={config.variant} samples={len(dataset)} "
        f"global_batch={config.training.global_batch_size} "
        f"accumulation={accumulation} trainable={report.trainable_parameters:,}"
    )
    optimizer.zero_grad(set_to_none=True)
    global_step = state.global_step
    epoch = state.epoch
    skip_batches = state.batches_in_epoch
    max_grad_norm = float(config.training.optimizer.max_grad_norm)

    while global_step < config.training.steps:
        active_loader = dataloader
        if skip_batches:
            if skip_batches >= len(dataloader):
                epoch += skip_batches // len(dataloader)
                skip_batches %= len(dataloader)
            if skip_batches:
                active_loader = accelerator.skip_first_batches(
                    dataloader, skip_batches
                )
        # Accelerate's loaders reset the sampler epoch from their own pass
        # counter on every iteration (a skip loader restarts it at zero), so
        # set the epoch on each loader that is actually iterated.
        dataloader.set_epoch(epoch)
        if active_loader is not dataloader:
            active_loader.set_epoch(epoch)

        completed_epoch = True
        for local_batch_index, batch in enumerate(active_loader):
            batch_index = skip_batches + local_batch_index
            with accelerator.accumulate(training_module):
                loss = training_module(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        training_module.parameters(), max_grad_norm
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    # Log the rate this update used, as HF Trainer does.
                    update_lr = lr_scheduler.get_last_lr()[0]
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_steps == 0 or global_step == 1:
                    mean_loss = accelerator.gather(loss.detach().reshape(1)).mean()
                    accelerator.print(
                        f"step={global_step}/{config.training.steps} "
                        f"loss={float(mean_loss):.6f} "
                        f"lr={update_lr:.3e}"
                    )
                next_state = TrainingState(
                    global_step=global_step,
                    epoch=epoch,
                    batches_in_epoch=batch_index + 1,
                )
                if args.save_steps and global_step % args.save_steps == 0:
                    save_resume_checkpoint(
                        accelerator,
                        training_module,
                        output_dir / f"checkpoint-{global_step}",
                        next_state,
                        config=config,
                        run_contract=run_contract,
                    )
                    if accelerator.is_main_process:
                        _prune_resume_checkpoints(
                            output_dir,
                            keep=args.save_total_limit,
                            current_step=global_step,
                        )
                if global_step >= config.training.steps:
                    state = next_state
                    completed_epoch = False
                    break
        if completed_epoch:
            epoch += 1
            skip_batches = 0
            state = TrainingState(global_step=global_step, epoch=epoch)
        else:
            break

    save_resume_checkpoint(
        accelerator,
        training_module,
        output_dir,
        state,
        config=config,
        run_contract=run_contract,
    )
    if accelerator.is_main_process:
        from ..checkpoint import export_release_checkpoint

        unwrapped = accelerator.unwrap_model(training_module)
        export_release_checkpoint(
            unwrapped.adaloop_model,
            config,
            output_dir,
            processor=processor,
            overwrite=True,
        )
    accelerator.wait_for_everyone()
    accelerator.print(
        f"Training complete at step {global_step}; resumable and canonical "
        f"release checkpoint: {output_dir}"
    )


if __name__ == "__main__":
    main()
