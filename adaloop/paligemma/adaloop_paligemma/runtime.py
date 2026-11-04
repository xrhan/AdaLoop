# Copyright 2024 Big Vision Authors.
# Copyright 2026 The AdaLoop Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the AdaLoop Authors from Big Vision
# (https://github.com/google-research/big_vision, commit 0127fb6b)
# big_vision/utils.py: recover_tree, recover_dtype, and the NPZ branch of
# load_params were adapted to read one local regular file without importing
# Big Vision; model reconstruction and overlay application are new.

"""Construct a runnable PaliGemma AdaLoop model from licensed base assets.

The release never redistributes the PaliGemma base checkpoint or tokenizer.
This module is the built-in reconstruction hook: it loads the exact user-
provided Big Vision NPZ, restores its Flax parameter names, adds the paper LoRA and
identity-initialized adaptive modules, then applies a verified compact overlay.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import PaliGemmaAdaLoopConfig
from .model import (
    LoopedSigLIPVision,
    PaliGemmaTokenIds,
    QuestionConditioner,
    add_lora_params,
    build_model_from_config,
    unstack_scanned_vision_params,
)


@dataclass(frozen=True)
class PaliGemmaRuntime:
    """Fully reconstructed model, parameters, tokenizer, and signed recipe."""

    model: Any
    params: Mapping[str, Any]
    tokenizer: Any
    recipe: PaliGemmaAdaLoopConfig
    token_ids: PaliGemmaTokenIds | None


def _jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "PaliGemma reconstruction requires JAX; install "
            "adaloop-paligemma[runtime]"
        ) from exc
    return jax, jnp


def load_tokenizer(path: str | Path) -> Any:
    """Load the pinned SentencePiece model from a local regular file."""

    try:
        import sentencepiece
    except ImportError as exc:  # pragma: no cover - package dependency
        raise RuntimeError("PaliGemma tokenization requires sentencepiece") from exc
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"Tokenizer must be a regular local file: {resolved}")
    tokenizer = sentencepiece.SentencePieceProcessor()
    if not tokenizer.Load(str(resolved)):
        raise ValueError(f"SentencePiece failed to load {resolved}")
    return tokenizer


def draft_suppressed_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return PAD/BOS/UNK/control IDs, excluding EOS and content pieces."""

    eos = int(tokenizer.eos_id())
    values: set[int] = set()
    for name in ("pad_id", "bos_id", "unk_id"):
        getter = getattr(tokenizer, name, None)
        if callable(getter):
            token = int(getter())
            if token >= 0:
                values.add(token)
    is_control = getattr(tokenizer, "IsControl", None)
    piece_size = getattr(tokenizer, "GetPieceSize", None)
    if callable(is_control) and callable(piece_size):
        values.update(
            index for index in range(int(piece_size())) if bool(is_control(index))
        )
    values.discard(eos)
    return tuple(sorted(values))


def token_ids_from_tokenizer(tokenizer: Any) -> PaliGemmaTokenIds:
    separator = tuple(int(value) for value in tokenizer.Encode("\n", add_bos=False))
    return PaliGemmaTokenIds(
        eos=int(tokenizer.eos_id()),
        separator=separator,
        suppressed_draft=draft_suppressed_token_ids(tokenizer),
    )


def _recover_tree(keys: list[str], values: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    children: dict[str, list[tuple[str, Any]]] = collections.defaultdict(list)
    for key, value in zip(keys, values, strict=True):
        if "/" not in key:
            result[key] = value
        else:
            head, tail = key.split("/", 1)
            children[head].append((tail, value))
    for head, entries in children.items():
        child_keys, child_values = zip(*entries, strict=True)
        result[head] = _recover_tree(list(child_keys), list(child_values))
    return result


def _recover_dtype(value: np.ndarray) -> np.ndarray:
    if value.dtype.type is np.void:
        if value.itemsize != 2:
            raise ValueError(f"Unsupported void checkpoint dtype {value.dtype}")
        _, jnp = _jax()
        return value.view(jnp.bfloat16)
    return value


def load_big_vision_params(path: str | Path) -> dict[str, Any]:
    """Load the pinned combined Big Vision NPZ without importing Big Vision."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"Base checkpoint must be a regular local NPZ: {resolved}")
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            keys = list(archive.files)
            values = [_recover_dtype(np.asarray(archive[key])) for key in keys]
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load Big Vision checkpoint {resolved}: {exc}") from exc
    tree = _recover_tree(keys, values)
    if set(tree) == {"params"} and isinstance(tree["params"], Mapping):
        tree = dict(tree["params"])
    elif "opt" in tree and isinstance(tree["opt"], Mapping):
        target = tree["opt"].get("target")
        if isinstance(target, Mapping):
            tree = dict(target)
    if not isinstance(tree.get("img"), Mapping) or not isinstance(
        tree.get("llm"), Mapping
    ):
        raise ValueError("Combined base checkpoint must contain img and llm roots")
    return tree


class _QuestionInitLLM:
    def __init__(self, width: int) -> None:
        self.width = int(width)

    def embed_tokens(self, token_ids, train: bool = False):
        del train
        _, jnp = _jax()
        return jnp.zeros(token_ids.shape + (self.width,), jnp.float32)


# The question pooler is initialized from a fixed key, independent of the run
# seed, unlike the AdaLN generators below, which fold it in.  Every training
# seed therefore starts from the same pooler; the shipped recipes use seed 0.
_QUESTION_INIT_SEED = 0


def _initialize_question_params(recipe: PaliGemmaAdaLoopConfig, llm_width: int):
    jax, jnp = _jax()
    module = QuestionConditioner(
        max_tokens=recipe.conditioner.max_tokens,
        cond_dim=recipe.conditioner.hidden_size,
        num_heads=recipe.conditioner.attention_heads,
        detach=True,
    )
    token_ids = jnp.ones((1, 4), jnp.int32)
    mask = jnp.ones_like(token_ids, dtype=bool)
    variables = module.init(
        jax.random.fold_in(jax.random.PRNGKey(_QUESTION_INIT_SEED), 71_071),
        _QuestionInitLLM(llm_width),
        token_ids,
        mask,
        train=False,
    )
    return variables["params"]


def _initialize_adaptive_image_params(
    recipe: PaliGemmaAdaLoopConfig,
    *,
    language_width: int,
):
    """Initialize the adaptive image tree exactly as the paper's training did.

    Flax derives parameter RNGs from the complete module scope.  Initializing
    each AdaLN generator in isolation would give it the right shape but a
    different random ``cond_proj`` than the paper runs.  Build one depth-one
    tower with the original folded seed and later retain only its additive
    ``adaln`` subtrees; all pretrained leaves still come from the pinned NPZ.
    """

    jax, jnp = _jax()
    module = LoopedSigLIPVision(
        num_classes=int(language_width),
        loop_steps=1,
        use_adaln=True,
        adaln_hidden_dim=recipe.adaln.hidden_size,
        remat_policy="nothing_saveable",
    )
    key = jax.random.fold_in(
        jax.random.PRNGKey(int(recipe.training.seed)), 91_337
    )
    return module.init(
        {"params": key, "dropout": key},
        jnp.zeros(
            (
                1,
                int(recipe.backbone.image_size),
                int(recipe.backbone.image_size),
                3,
            ),
            jnp.float32,
        ),
        jnp.zeros((1, 1, recipe.conditioner.hidden_size), jnp.float32),
        train=False,
    )["params"]


def initialize_parameter_tree(
    recipe: PaliGemmaAdaLoopConfig,
    base_checkpoint: str | Path,
) -> dict[str, Any]:
    """Build the full fresh-training parameter tree from the exact base NPZ."""

    params = load_big_vision_params(base_checkpoint)
    params = dict(params)
    params["img"] = unstack_scanned_vision_params(dict(params["img"]))
    params["llm"] = add_lora_params(
        dict(params["llm"]),
        rank=int(recipe.lora.rank),
        targets=tuple(recipe.lora.target_modules),
        seed=int(recipe.lora.seed),
    )

    embedding = params["llm"].get("embedder", {}).get("input_embedding")
    if embedding is None or np.ndim(embedding) != 2:
        raise ValueError("Base LLM is missing llm/embedder/input_embedding")
    if int(embedding.shape[0]) != int(recipe.backbone.vocab_size):
        raise ValueError(
            "Pinned base vocabulary does not match the signed recipe: "
            f"base={embedding.shape[0]}, recipe={recipe.backbone.vocab_size}"
        )

    if recipe.conditioner.mode != "none":
        params["question_conditioner"] = _initialize_question_params(
            recipe, int(embedding.shape[1])
        )
        initialized_image = _initialize_adaptive_image_params(
            recipe, language_width=int(embedding.shape[1])
        )
        image = dict(params["img"])
        transformer = dict(image["Transformer"])
        initialized_transformer = initialized_image.get("Transformer")
        if not isinstance(initialized_transformer, Mapping):
            raise ValueError("Adaptive image initialization produced no Transformer")
        start, end = recipe.adaptive_span or (0, 0)
        for block_index in range(start, end):
            key = f"encoderblock_{block_index}"
            if key not in transformer:
                raise ValueError(f"Base vision tree is missing {key}")
            initialized_block = initialized_transformer.get(key)
            if not isinstance(initialized_block, Mapping) or "adaln" not in initialized_block:
                raise ValueError(
                    f"Adaptive image initialization produced no AdaLN for {key}"
                )
            block = dict(transformer[key])
            if "adaln" in block:
                raise ValueError(f"Base vision block {key} unexpectedly contains AdaLN")
            block["adaln"] = initialized_block["adaln"]
            transformer[key] = block
        image["Transformer"] = transformer
        params["img"] = image
    return params


def _overlay_tree(base: Any, overlay: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(overlay, Mapping):
        if not isinstance(base, Mapping):
            raise ValueError(f"Overlay mapping contradicts base at {'/'.join(path)}")
        unknown = set(overlay) - set(base)
        if unknown:
            raise ValueError(
                f"Overlay contains unknown keys at {'/'.join(path) or '<root>'}: "
                f"{sorted(unknown)}"
            )
        result = dict(base)
        for key, value in overlay.items():
            result[key] = _overlay_tree(base[key], value, path + (str(key),))
        return result
    if isinstance(base, Mapping):
        raise ValueError(f"Overlay leaf contradicts base mapping at {'/'.join(path)}")
    base_array = np.asarray(base)
    overlay_array = np.asarray(overlay)
    if base_array.shape != overlay_array.shape or str(base_array.dtype) != str(
        overlay_array.dtype
    ):
        raise ValueError(
            f"Overlay shape/dtype mismatch at {'/'.join(path)}: "
            f"base={base_array.shape}/{base_array.dtype}, "
            f"overlay={overlay_array.shape}/{overlay_array.dtype}"
        )
    return overlay


def initialize_runtime(
    recipe: PaliGemmaAdaLoopConfig,
    base_checkpoint: str | Path,
    tokenizer_path: str | Path,
    *,
    overlay: Mapping[str, Any] | None = None,
    verify_assets: bool = True,
) -> PaliGemmaRuntime:
    """Initialize fresh training or reconstruct a compact public checkpoint."""

    base_path = Path(base_checkpoint).expanduser().resolve()
    tokenizer_file = Path(tokenizer_path).expanduser().resolve()
    if verify_assets:
        # Keep direct fresh-training use as strict as public checkpoint load.
        # The checkpoint module does not import runtime at module import time,
        # so this local import avoids a package-level cycle.
        from .checkpoint import validate_external_asset

        base_path = validate_external_asset(
            base_path, recipe.base_model, "base checkpoint"
        )
        tokenizer_file = validate_external_asset(
            tokenizer_file, recipe.tokenizer, "tokenizer"
        )
    tokenizer = load_tokenizer(tokenizer_file)
    ids = token_ids_from_tokenizer(tokenizer) if recipe.variant == "c" else None
    model = build_model_from_config(recipe, token_ids=ids)
    params = initialize_parameter_tree(recipe, base_path)
    if overlay is not None:
        params = _overlay_tree(params, overlay)
    return PaliGemmaRuntime(
        model=model,
        params=params,
        tokenizer=tokenizer,
        recipe=recipe,
        token_ids=ids,
    )


def reconstruct_parameter_template(
    recipe: PaliGemmaAdaLoopConfig,
    base_checkpoint: Path,
    tokenizer: Path,
) -> Mapping[str, Any]:
    """Built-in reconstruction hook returning the exact full template.

    The exporter verifies the tokenizer bytes before invoking this hook.  We
    load it once as an additional format check even though parameter shapes do
    not depend on tokenization.
    """

    loaded = load_tokenizer(tokenizer)
    if int(loaded.GetPieceSize()) != int(recipe.backbone.vocab_size):
        raise ValueError(
            "Tokenizer vocabulary does not match the signed backbone contract"
        )
    return initialize_parameter_tree(recipe, base_checkpoint)


def reconstruct_runtime(
    recipe: PaliGemmaAdaLoopConfig,
    base_checkpoint: Path,
    tokenizer: Path,
    overlay: Mapping[str, Any],
) -> PaliGemmaRuntime:
    """Default hook accepted by :class:`VerifiedPaliGemmaCheckpoint`."""

    return initialize_runtime(
        recipe,
        base_checkpoint,
        tokenizer,
        overlay=overlay,
        # PaliGemmaCheckpointLoader verified both assets immediately before
        # invoking this reconstruction hook; avoid hashing the 6 GB NPZ twice.
        verify_assets=False,
    )


__all__ = [
    "PaliGemmaRuntime",
    "draft_suppressed_token_ids",
    "initialize_parameter_tree",
    "initialize_runtime",
    "load_big_vision_params",
    "load_tokenizer",
    "reconstruct_parameter_template",
    "reconstruct_runtime",
    "token_ids_from_tokenizer",
]
