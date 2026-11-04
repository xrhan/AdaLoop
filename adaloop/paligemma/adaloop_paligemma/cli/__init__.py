"""Command-line entry points for the PaliGemma backend."""

from __future__ import annotations

import argparse
from typing import Any

#: ``--matmul-precision`` values: the JAX setting each selects (None: JAX's default).
MATMUL_PRECISIONS = {"default": None, "bf16": "BF16_BF16_F32"}


def add_matmul_precision_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--matmul-precision",
        choices=tuple(MATMUL_PRECISIONS),
        default="default",
        help=(
            "Dot precision of the model on GPU. The model keeps float32 "
            "activations: on TPU every dot already takes bfloat16 inputs and "
            "accumulates in float32, on GPU the default is TF32. 'bf16' selects "
            "the TPU arithmetic and the faster bfloat16 tensor cores. TPU and "
            "CPU ignore it. The parameters and checkpoints do not change."
        ),
    )


def apply_matmul_precision(jax: Any, choice: str) -> str:
    """Set the default dot precision of every later trace; return the one in effect.

    Only GPU changes: TPU already computes this way, and XLA:CPU rejects the
    algorithm.
    """
    algorithm = MATMUL_PRECISIONS[choice]
    if algorithm is None or jax.default_backend() != "gpu":
        return "default"
    jax.config.update("jax_default_matmul_precision", algorithm)
    return choice
