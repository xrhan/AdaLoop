# AdaLoop: Thinking with the Vision Encoder

[![arXiv](https://img.shields.io/badge/arXiv-2610.21323-b31b1b.svg)](https://arxiv.org/abs/2610.21323)&nbsp;
[![Checkpoints](https://img.shields.io/badge/Hugging%20Face-AdaLoop-blue.svg)](https://huggingface.co/Lyy0725/AdaLoop)&nbsp;
[![Dataset](https://img.shields.io/badge/Hugging%20Face-VisRO-yellow.svg)](https://huggingface.co/datasets/Lyy0725/VisRO)&nbsp;
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](../LICENSE)&nbsp;

This directory holds the official AdaLoop implementation for the paper [AdaLoop: Thinking with the Vision Encoder](https://arxiv.org/abs/2610.21323): three backends that train the paper's models from the pinned base checkpoints and run the released checkpoints, and the contracts they share. The released checkpoints and their scores are listed in the [repository README](../README.md#inference). Commands below run from the repository root.

## Overview

AdaLoop adds question-conditioned, weight-tied recurrence to the vision encoder of a pretrained vision-language model. A consecutive span of frozen vision blocks is reused for several iterations while small AdaLN modules adapt each update to the question. The language model reads only the current visual state (Variant A), all states (Variant B), or states interleaved with bounded answer drafts (Variant C).

| Path | Contents |
| --- | --- |
| [`common/`](common/) | framework-independent variant, depth-weighting, layout and answer-stability contracts, plus the VisRO training data and evaluation shared by the backends |
| [`qwen/`](qwen/) | PyTorch implementation for `Qwen/Qwen2.5-VL-3B-Instruct` |
| [`paligemma/`](paligemma/) | JAX/Flax implementation for `google/paligemma2-3b-mix-448` |
| [`llava/`](llava/) | PyTorch implementation for LLaVA-1.5-7B (Vicuna-7B-v1.5 + CLIP ViT-L/14-336 at 448 px, 2×2 pooling) |
| [`docs/`](docs/) | architecture, the exact A/B/C interfaces, the portable checkpoint format |

| Backend | Framework | SFT | AdaLN | Variant A | Variant B | Variant C | Entry points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-VL-3B-Instruct | PyTorch | yes | yes | yes | yes | yes | train, evaluate, inference and inspect CLIs, Python APIs |
| PaliGemma2-3B-Mix-448 | JAX/Flax | yes | yes | yes | yes | yes | train, evaluate, inference and inspect CLIs, Python APIs |
| LLaVA-1.5-7B (448 px) | PyTorch | yes | yes | yes | yes | yes | train, evaluate, inference, convert and inspect CLIs, Python APIs |

Every backend trains and evaluates on the public VisRO release, read in place with the reader and scorer of the [`visro`](https://github.com/Lyy-iiis/VisRO) package (see [Data](#data) and [Evaluation](#evaluation)). Qwen3 and Qwen3-MoE are intentionally unsupported; the Qwen package is pinned to the Qwen2.5-VL implementation in Transformers 4.57.1. LLaVA ships the three settings the paper trained on it plus Variants B and C, and is pinned to Transformers 4.57.6 because its paper loss normalization depends on `Trainer` internals.

## Initialization

Python 3.10 or newer is required. Run from the repository root and install the common package and one backend. The common package automatically installs the VisRO reader and scorer; use a separate environment per backend, since they pin different framework versions:

```bash
pip install -e ./adaloop/common -e './adaloop/qwen[gpu]'
```

The VisRO reader and scorer need only Pillow, and the common package has no tensor-framework dependency. Dataset generation uses a separate environment; see the [VisRO generation guide](https://github.com/Lyy-iiis/VisRO#initialization).

- **Qwen.** The `qwen[gpu]` extra has binary wheels for Python 3.10–3.12; [`qwen/environment.yml`](qwen/environment.yml) pins Torch, Transformers, PEFT and Accelerate.
- **PaliGemma.** Choose exactly one of [`paligemma/environment-tpu.yml`](paligemma/environment-tpu.yml) or [`paligemma/environment-cuda.yml`](paligemma/environment-cuda.yml), then install `./adaloop/common` and `./adaloop/paligemma` the same way. Both use Python 3.10 and pin the stack exactly (jax 0.6.2 with its TPU or CUDA 12 plugin, flax 0.10.7, optax 0.2.8, orbax-checkpoint 0.11.39, einops 0.8.2). The TPU file needs x86_64 Linux with glibc 2.31 or newer; the CUDA file Linux with glibc 2.27 or newer on x86_64 (2.28 on aarch64). The `runtime` extra pins the same stack with CPU wheels, or jax 0.11.2, flax 0.12.10 and orbax-checkpoint 0.12.7 on Python 3.14 or newer. Verify `jax.devices()` and freeze the environment before a production run.
- **LLaVA.** [`llava/environment.yml`](llava/environment.yml) pins Python 3.11, Torch 2.6.0, Transformers 4.57.6, PEFT 0.17.1 and Accelerate 1.15.0; the `llava[gpu]` extra pins the same versions and has binary wheels for Python 3.10–3.12:

```bash
conda env create -f adaloop/llava/environment.yml
conda activate adaloop-llava
pip install -e ./adaloop/common -e './adaloop/llava[gpu]'
```

## Data

All three backends read the public VisRO release in place, with `visro.dataset`:

```text
<VisRO root>/tasks/<category>/<task>/{train,eval,ood}/NNNNN.tar
    <key>.json  <key>.png  ...   (WebDataset: per example its record, then its image)
```

A shard is read from its tar file or from an extracted shard directory. `--tasks` takes task names, `all`, or `core6` (the default): the paper's six tasks `count_clusters`, `flow_free`, `maze`, `metro_map`, `overlapping_glyphs` and `path_tracing`. `--difficulty` restricts training to some levels. Every image is checked against its record's `image_sha256` when it is read.

**Training set.** The `train` split of the chosen tasks (100,000 rows per task, 600,000 for `core6`) in dataset order (tasks, shards, records), followed by the rows of an optional replay JSONL file in file order. Every row has the same weight. The training question is the record's `question` and the target its `answer`.

**Replay.** Every paper run mixed 100,000 replay rows into the 600,000 VisRO rows to limit forgetting of general visual question answering. They are single-turn rows from `lmms-lab/LLaVA-NeXT-Data` (revision `c8aef391ce214167c4ebc7f06bc05a50dee2e75f`): the first valid human and assistant turn of each conversation, with the original assistant answer, in a fixed sample (seed 42) of 100,000 rows whose answers are at most 128 tokens long. They use 87,391 distinct images; 89,465 are general VQA prompts, 5,766 counting, 4,412 captioning and 357 OCR. Replay is not part of VisRO and is not distributed here. Supply your own as a JSONL file whose lines hold exactly `image`, `question` and `answer` plus an optional unique `id`; relative image paths are resolved against `--replay-image-root` (default: the file's directory):

```json
{"id":"replay-1","image":"images/0001.jpg","question":"What color are the sheep?\nAnswer the question using a single word or phrase.","answer":"White"}
```

**Sampling.** Each backend shuffles the pooled rows with its seeded sampler, so a seed fixes the order of every epoch and a resumed run replays it. Qwen draws `torch.randperm` with seed `training.seed + epoch` and drops the last incomplete round; PaliGemma shuffles with `random.Random(training.seed + 1_000_003 * epoch)`, keeps whole global batches and shards them by host; LLaVA uses the Hugging Face `Trainer`'s random sampler seeded with `train.seed`. The Qwen and PaliGemma resume contracts record the selection and digests of the ordered sample ids, questions, answers and image bytes, so a resumed run must see the same data.

## Evaluation

`adaloop-qwen-eval`, `adaloop-paligemma-eval` and `adaloop-llava-eval` (a configuration file per setting) evaluate a checkpoint on the `eval` and `ood` splits (3,600 items each for `core6`):

- every item is answered greedily at the fixed depths 1..D (default: the training depth; 1 for SFT and single-pass AdaLN);
- `p=1` answer stability is applied to those answers with budget T (default: the training depth): an item stops at the first depth whose answer equals the previous one after case and whitespace normalization, else at T;
- answers are scored with `visro.scoring.score_record`, which uses only the published record fields, after checking that every reference answer scores as correct;
- the condition is the bare question. The language-model text is the question plus the paper's answer-format instruction for Qwen and PaliGemma (`--prompt-style instruction`) and the bare question for LLaVA (`eval.prompt_style: question`), as in the paper.

The output directory holds `<split>/depth<d>.jsonl` and `<split>/p1_T<T>.jsonl` (one scored row per item: id, task, difficulty, depth, prediction, correct, parsed answer, parsed reference, scoring rule), `summary.json` (accuracy per task and difficulty, the macro and micro means, and the mean depth, per split and setting), and `results.md`:

```text
| ood | Count | Flow | Glyph | Metro | Maze | Path | Avg | Depth |
| depth 1 | ... |
| depth 3 | ... |
| p=1 (T=3) | ... |
```

Each backend guide has a numbered sanity check with the expected scores of a released checkpoint: [Qwen](qwen/README.md#evaluation), [PaliGemma](paligemma/README.md#evaluation), [LLaVA](llava/README.md#evaluation).

## Inference

Run a portable checkpoint at a fixed depth or with the paper's `p=1` answer-stability rule. `--question` is the bare question used for AdaLN conditioning, and `--prompt` optionally replaces the language-model text, for example to append one of the paper's answer-format instructions:

```bash
adaloop-qwen-infer \
    --checkpoint checkpoints/qwen/variant_a \
    --image example.png --question "Which path reaches the star?" \
    --prompt $'Which path reaches the star?\nAnswer with just the letter.' \
    --p1 --max-depth 3 --fallback-depth 3
```

`$'...'` (bash, zsh) passes a real newline; inside plain double quotes, `\n` reaches the model as a backslash and an `n`. Use `--depth 3` instead of `--p1 ...` for a fixed depth. `adaloop-paligemma-infer` and `adaloop-llava-infer` work the same way with their base-model flags; see the backend guides.

## Training

Each backend trains from a recipe in its `configs/` directory; the command line supplies the paths. The Qwen recipes cover plain SFT, single-pass AdaLN, Variants A/B/C and the paper's component controls (Variant A without AdaLN, and parameter-matched constant-question AdaLN and Variant A); PaliGemma has the five paper recipes (SFT, AdaLN, A, B, C); LLaVA has SFT, single-pass AdaLN and Variants A, B and C. The [repository README](../README.md#training) shows one training command per backend, and the guides cover every recipe, resuming and checkpoint conversion: [Qwen](qwen/README.md#training), [PaliGemma](paligemma/README.md#training), [LLaVA](llava/README.md#training).

The PaliGemma CLI uses JAX data parallelism and joins multi-host GPU or TPU runs with `--jax-distributed`; applications with custom meshes may compose the lower-level model, data, runtime-reconstruction and checkpoint APIs.

## Design Contracts

- [Architecture](docs/architecture.md)
- [Variant A/B/C language interfaces](docs/variants_abc.md)
- [Checkpoint format](docs/checkpoint_format.md)

Released checkpoints serialize the full AdaLoop topology and paper input limits. Qwen pins an immutable Hub revision and bundles a canonical processor snapshot; PaliGemma pins the external base NPZ and tokenizer by exact byte size and SHA-256; LLaVA pins CLIP by Hub revision and the released stage-2 base by the size and SHA-256 of each of its files. All three loaders validate metadata and artifact hashes instead of guessing missing flags.

## Scope

All three backends include end-to-end training, evaluation and inference CLIs plus lower-level model, data, recipe and checkpoint APIs. Not included: Qwen3 and Qwen3-MoE; external baseline implementations; dataset generation, which lives in [the VisRO repository](https://github.com/Lyy-iiis/VisRO); learned halting heads; and training controls or ablations that the paper does not report. The LLaVA backend keeps a few behaviors of the paper's LLaVA runs that differ from the Qwen and PaliGemma contracts; the [LLaVA guide](llava/README.md) lists them.

## License

The code is under the Apache License 2.0, see [LICENSE](../LICENSE). See [NOTICE](../NOTICE) for upstream attribution and the [repository README](../README.md#citation) for the citation. Source files that adapt Apache-2.0 code from Big Vision (PaliGemma backend), Transformers (Qwen backend and the LLaVA backend's CLIP layer), LLaVA, or the FastChat and Stanford Alpaca code that LLaVA's training script adopts (LLaVA backend) keep the upstream copyright and license header, state the AdaLoop modifications, and are listed in `NOTICE`. `LICENSE` and `NOTICE` at the repository root apply to every package (`common/`, `qwen/`, `paligemma/`, `llava/` and `visro`); keep both with any redistribution.
