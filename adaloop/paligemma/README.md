# AdaLoop for PaliGemma 2

[![arXiv](https://img.shields.io/badge/arXiv-2610.21323-b31b1b.svg)](https://arxiv.org/abs/2610.21323)&nbsp;
[![Checkpoints](https://img.shields.io/badge/Hugging%20Face-AdaLoop-blue.svg)](https://huggingface.co/Lyy0725/AdaLoop/tree/main/paligemma)&nbsp;
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](../../LICENSE)&nbsp;

This is the official JAX/Flax implementation of AdaLoop for `google/paligemma2-3b-mix-448`: the 27-block SigLIP So400m/14 vision tower, the PaliGemma connector and the Gemma 2 2B language model. It provides the paper's plain SFT and single-pass AdaLN controls and recurrent Variants A/B/C, trained and evaluated on VisRO. The paper's runs used TPUs; the CUDA setup is provided for local inference and development. Commands run from the repository root.

PaliGemma and Qwen live in separate packages because their accelerator stacks, checkpoint formats, attention conventions and language interfaces differ. They share only the small contracts and the VisRO data and evaluation in `adaloop-common`.

## Initialization

Do not install the TPU and CUDA JAX plugins into the same environment. Choose one environment file:

```bash
# TPU
conda env create -f adaloop/paligemma/environment-tpu.yml
conda activate adaloop-paligemma-tpu

# Or NVIDIA CUDA 12
conda env create -f adaloop/paligemma/environment-cuda.yml
conda activate adaloop-paligemma-cuda
```

Then install this backend with the common package, which automatically installs the VisRO reader and scorer:

```bash
pip install -e ./adaloop/common -e ./adaloop/paligemma
```

Both environment files use Python 3.10 and pin the stack exactly: the `runtime` extra's `jax==0.6.2` (with its `tpu` or `cuda12` extra), `flax==0.10.7`, `optax==0.2.8`, `orbax-checkpoint==0.11.39` and `einops==0.8.2`, plus exact versions of the package's other dependencies. `jax[tpu]==0.6.2` installs libtpu 0.0.17, whose only wheel needs x86_64 Linux with glibc 2.31 or newer. `jax[cuda12]==0.6.2` installs the JAX CUDA 12 plugin and NVIDIA's CUDA libraries from PyPI, and the CUDA file has wheels for x86_64 Linux with glibc 2.27 or newer and aarch64 Linux with glibc 2.28 or newer. Because the versions are exact, pip installs this stack or fails; it cannot fall back to an older JAX. Transitive dependencies, such as the NVIDIA libraries, SciPy and TensorStore, are not pinned, so for a production run freeze the resolved environment after verifying `jax.devices()` on the target TPU or GPU. jaxlib 0.6.2 has no Python 3.14 wheels, so on Python 3.14 and newer the extra pins `jax==0.11.2`, `flax==0.12.10` and `orbax-checkpoint==0.12.7` instead.

## Base Model

The released checkpoints do not redistribute Google's gated base weights. Obtain PaliGemma 2 under its model license and provide the Big Vision NPZ and SentencePiece model expected by the recipe. The paper recipes pin both by content rather than by a machine-specific path:

| Asset | Expected value |
| --- | --- |
| Model | `google/paligemma2-3b-mix-448` |
| NPZ size | `6,065,970,800` bytes |
| NPZ SHA-256 | `2d426f6d23ec5ecd9b49f01a815bfb3ce5a373fa98bd154568cc402dcf8fc049` |
| Tokenizer size | `4,264,023` bytes |
| Tokenizer SHA-256 | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |

The NPZ is in the gated Hugging Face repository [`google/paligemma2-3b-mix-448-jax`](https://huggingface.co/google/paligemma2-3b-mix-448-jax): accept the license on its page, log in to Hugging Face, and download the 6 GB file. The package's `huggingface_hub` dependency provides the Python calls below in every version the package allows, unlike its command-line tools (`huggingface-cli` was removed in 1.0, and `hf` first appeared in 0.34). The tokenizer is public in Big Vision's Cloud Storage bucket. Then check both files against the table:

```bash
mkdir -p paligemma-assets && cd paligemma-assets
python -c "from huggingface_hub import login; login()"  # or set HF_TOKEN
python - <<'EOF'
from huggingface_hub import hf_hub_download

hf_hub_download(
    "google/paligemma2-3b-mix-448-jax",
    "paligemma2-3b-mix-448.b16.npz",
    revision="d370875d16d1592eee9e579794dd380770381a22",
    local_dir=".",
)
EOF
curl -fL -o paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
cat > paligemma-assets.sha256 <<'EOF'
2d426f6d23ec5ecd9b49f01a815bfb3ce5a373fa98bd154568cc402dcf8fc049  paligemma2-3b-mix-448.b16.npz
8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6  paligemma_tokenizer.model
EOF
sha256sum -c paligemma-assets.sha256  # macOS: shasum -a 256 -c paligemma-assets.sha256
```

Pass the two files as `--base-checkpoint` and `--tokenizer`. Loaders reject mismatched bytes. Paths are supplied at runtime and are never serialized into a portable checkpoint.

## Checkpoints

The released checkpoints are in [`Lyy0725/AdaLoop`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/paligemma). Scores are macro accuracy on VisRO-Core6 (3,600 items per split), *measured from this checkpoint / reported in the paper*; Variants A, B and C use `p=1` stopping with at most 3 iterations. The Variant B and C checkpoints were trained by us with the same recipes and are not in the paper.

| Checkpoint | Setting | ID | OOD |
| --- | --- | ---: | ---: |
| [`paligemma/sft`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/paligemma/sft) | SFT | 62.7 / 62.6 | 31.5 / 31.5 |
| [`paligemma/adaln`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/paligemma/adaln) | single-pass AdaLN | 72.4 / 72.4 | 40.0 / 40.0 |
| [`paligemma/variant_a`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/paligemma/variant_a) | AdaLoop-A | 75.2 / 75.2 | 43.9 / 43.9 |
| [`paligemma/variant_b`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/paligemma/variant_b) | AdaLoop-B | 74.0 / – | 43.7 / – |
| [`paligemma/variant_c`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/paligemma/variant_c) | AdaLoop-C | 74.1 / – | 41.1 / – |

## Evaluation

`adaloop-paligemma-eval` evaluates a checkpoint on the VisRO `eval` and `ood` splits at the fixed depths 1..`--max-depth` and with `p=1` stopping over 1..`--p1-max-depth` (both default to the training depth), scores every answer with `visro.scoring`, and writes per-item rows, `summary.json` and a per-task table `results.md` (see [Evaluation](../README.md#evaluation)). The condition is the bare question and the language prompt the question plus the paper's answer-format instruction (`--prompt-style instruction`, the default; `question` passes the bare question). `--per-cell N` evaluates the first N items of each (task, difficulty) cell, for a quick check.

#### Sanity Check

1. **Download the checkpoint, the evaluation splits and the [base model](#base-model):**
    ```bash
    hf download Lyy0725/AdaLoop --include "paligemma/variant_a/*" --local-dir checkpoints
    hf download Lyy0725/VisRO --repo-type dataset --local-dir data/VisRO --include "tasks/*/*/eval/*"
    hf download Lyy0725/VisRO --repo-type dataset --local-dir data/VisRO --include "tasks/*/*/ood/*"
    ```

2. **Check the checkpoint and the base assets** (no JAX needed):
    ```bash
    adaloop-paligemma-inspect checkpoints/paligemma/variant_a \
        --base-checkpoint paligemma-assets/paligemma2-3b-mix-448.b16.npz \
        --tokenizer paligemma-assets/paligemma_tokenizer.model
    ```

3. **Launch evaluation:**
    ```bash
    adaloop-paligemma-eval \
        --checkpoint checkpoints/paligemma/variant_a \
        --base-checkpoint paligemma-assets/paligemma2-3b-mix-448.b16.npz \
        --tokenizer paligemma-assets/paligemma_tokenizer.model \
        --visro-root data/VisRO --tasks core6 --splits eval,ood \
        --output-dir outputs/eval/paligemma_variant_a
    ```
    The expected `p=1 (T=3)` scores in `results.md` are about 75.2 (ID) and 43.9 (OOD).

## Inference

The Python pipeline reconstructs the exact base plus overlay and provides fixed-depth greedy decoding and the paper's `p=1` adjacent-answer stability rule:

```python
from adaloop_paligemma.inference import AdaLoopPaliGemmaPipeline

pipe = AdaLoopPaliGemmaPipeline.from_pretrained(
    "checkpoints/paligemma/variant_a",
    base_checkpoint="paligemma-assets/paligemma2-3b-mix-448.b16.npz",
    tokenizer="paligemma-assets/paligemma_tokenizer.model",
)

question = "Which path reaches the star?"
prompt = question + "\nAnswer with just the letter."
fixed = pipe.generate("example.png", question, prompt=prompt, depth=3)
adaptive = pipe.generate_p1("example.png", question, prompt=prompt, max_depth=3, fallback_depth=3)
print(fixed.answer)
print(adaptive.answer, adaptive.depth, adaptive.stabilized)
```

The equivalent fixed-depth and `p=1` CLI calls:

```bash
adaloop-paligemma-infer \
    --checkpoint checkpoints/paligemma/variant_a \
    --base-checkpoint paligemma-assets/paligemma2-3b-mix-448.b16.npz \
    --tokenizer paligemma-assets/paligemma_tokenizer.model \
    --image example.png --question "Which path reaches the star?" \
    --prompt $'Which path reaches the star?\nAnswer with just the letter.' \
    --depth 3

adaloop-paligemma-infer \
    --checkpoint checkpoints/paligemma/variant_a \
    --base-checkpoint paligemma-assets/paligemma2-3b-mix-448.b16.npz \
    --tokenizer paligemma-assets/paligemma_tokenizer.model \
    --image example.png --question "Which path reaches the star?" \
    --prompt "$(printf 'Which path reaches the star?\nAnswer with just the letter.')" \
    --p1 --max-depth 3 --fallback-depth 3
```

Both quoting forms pass a real newline: `$'...'` works in bash and zsh, and `"$(printf ...)"` in any POSIX shell. A `\n` inside ordinary double quotes reaches the program as a literal backslash and `n`, which silently changes the prompt.

`question` is the bare question. It is the AdaLN condition and, unless `prompt` is given, also the language prompt. `prompt` is the full language-visible text: it fills the signed `{question}` evaluation template (`<bos>{prompt}\n`) and is repeated in every Variant C round, but never conditions vision. The paper's benchmark prompts chose the answer-format instruction, which starts with a newline, by task and answer type:

- maze multiple choice: `\nAnswer with just the letter.`
- maze true/false: `\nAnswer with only True or False.`
- maze counting: `\nAnswer with just the number.`
- other multiple choice: `\nAnswer with just the option label.`
- path tracing and open-ended questions, such as `How many clusters?`: none (the prompt is the bare question)

`<bos>{prompt}` may use at most 255 tokens (256 with the newline separator); a longer prompt raises an error rather than being truncated.

A recipe with `conditioner.mode: constant_question` conditions on its `constant_text` instead of the question; the language prompt is unchanged. For PaliGemma this mode is a release extension that any conditioned recipe (AdaLN, A, B or C) may use: it keeps exactly the question recipe's trainable leaves (39,163,392 parameters) and greedy decoding, and only the condition text changes. The paper's constant-question control is Qwen-only, and no PaliGemma constant recipe or checkpoint ships.

The Python API accepts the image as a path, encoded bytes, a `PIL.Image`, or a `uint8` array shaped `[H, W]`, `[H, W, 1]`, `[H, W, 3]` (RGB) or `[H, W, 4]` (RGBA); each is converted to RGB, resized bicubically to 448 by 448, and normalized. A floating-point array is accepted only as preprocessed `[448, 448, 3]` pixels in `[-1, 1]`. A non-negative float array that is not constant is rejected as raw `[0, 1]` pixels; constant arrays, such as zero placeholders, are accepted.

SFT and single-pass AdaLN accept only depth 1. Adaptive stopping requires an A/B/C checkpoint, evaluates depths in increasing order, and returns as soon as the first adjacent pair agrees. For Variants B and C, the packed prompt plus `max_new_tokens` must fit the 8192-token bound, or decoding raises a `ValueError`: Variant B allows at most 768 new tokens at depth 7, and Variant C at most 387 at depth 6. Adaptive stopping checks its deepest depth before decoding depth 1, so it fails up front rather than midway.

## Training

The portable trainer uses all visible JAX devices with data-parallel `pmap`. Run the following command to train Variant A:

```bash
adaloop-paligemma-train \
    --config adaloop/paligemma/configs/variant_a.yaml \
    --base-checkpoint paligemma-assets/paligemma2-3b-mix-448.b16.npz \
    --tokenizer paligemma-assets/paligemma_tokenizer.model \
    --visro-root data/VisRO \
    --replay-jsonl /path/to/replay.jsonl \
    --output-dir outputs/paligemma_variant_a \
    --export-dir outputs/paligemma_variant_a_release \
    --save-steps 1000 --log-steps 10
```

The same command accepts the SFT, AdaLN, A, B or C recipe. `training.global_batch_size` in the selected recipe must be divisible by the global device count and by the process count. For a multi-process run (multi-host TPU or GPU), pass `--jax-distributed` on every process; it calls `jax.distributed.initialize()` before any other JAX work. Cloud TPU (including GKE TPU), SLURM and Open MPI `mpirun` settings are detected automatically (with the pinned JAX, Open MPI before version 5). Kubernetes is detected only when the `kubernetes` package (`jax[k8s]`) is installed; neither the environment files nor the `runtime` extra install it. Other launches add `--coordinator-address HOST:PORT`, `--num-processes` and `--process-id`. Larger deployments may compose the model, data, training, inference and runtime primitives with their own mesh, sharding, compilation and launcher policy.

On GPU, add `--matmul-precision bf16` to training, inference and evaluation. The model keeps float32 activations, so by default every GPU dot runs in TF32; `bf16` runs it with bfloat16 inputs and float32 accumulation, which is how a TPU computes the same dot, on the faster bfloat16 tensor cores. It changes neither the parameters nor the checkpoint format. TPU and CPU runs ignore the flag.

#### Recipes

The strict YAML files under [`configs/`](configs/) are the serialized source of truth:

| Recipe | Visual execution | Language readout | Objective |
| --- | --- | --- | --- |
| `sft.yaml` | one stock pass | single image | terminal CE |
| `adaln.yaml` | one conditioned pass | single image | terminal CE |
| `variant_a.yaml` | latent-carry T=1–3 | current state | geometric deep supervision |
| `variant_b.yaml` | latent-carry T=1–3 | accumulated states | geometric deep supervision |
| `variant_c.yaml` | latent-carry T=1–3 | states and drafts | geometric deep supervision |

Shared paper defaults are:

- fixed 448 by 448 RGB input, bicubic resize, and normalization to `[-1, 1]`;
- loop span `[3, 15)` in the 27-block vision tower and train depth 3;
- frozen pretrained vision weights and a trainable connector;
- rank-64, alpha-128 LoRA on the Gemma attention `q`, fused `kv` and `o` projections;
- a detached bare-question condition with no BOS/EOS, at most 128 tokens, one shared 512-dimensional eight-head pooler, and independent hidden-128 zero-initialized AdaLN generators;
- 256-token training and evaluation text widths;
- per-example answer-token CE and normalized geometric depth weights with ratio 0.8; as in Big Vision and the paper's training runs, the CE uses the plain tied-embedding logits of the text positions only, without Gemma's final-logit softcap;
- 30,000 AdamW updates, global batch 64, `beta2=0.999`, 200 warmup updates, and cosine decay to 0.1 times the peak learning rate;
- deterministic final-answer decoding with a default budget of 128 new tokens, which inference's `max_new_tokens` (`--max-new-tokens`) overrides.

Several keys, such as `vision_round_mode`, `lora.layers`, `conditioner.include_bos`, `adaln.zero_init`, `training.precision` and `generation.do_sample`, accept exactly one value: they are fixed contract fields that record the recipe in the config, and the strict parser rejects any other value.

Variant B uses a chained block-causal prefix: each visual readout attends to itself and earlier readouts but not later ones; the final readout shares its bidirectional block with the question. The packed context, plus the decoded answer at inference, is bounded at 8192 tokens.

During training, Variant C repeats PaliGemma's native `<bos> answer en {question}\n` language prefix each round; evaluation repeats the signed `<bos>{prompt}\n` prefix instead, where the prompt defaults to the bare question (see [Inference](#inference)). It uses fresh reprefill, keeps the vision condition question-only, suppresses non-EOS special draft tokens, excludes stopping EOS from stored drafts, uses a newline separator, and caps drafts at 24 tokens. Its packed sequence of image readouts, prefixes, drafts and separators, plus the decoded answer at inference, is bounded at 8192 tokens. Depth 6 uses 7805 prompt slots, which leaves room for the 128-token answer budget; depth 7 does not fit. The paper runs trained Variant C with a 6144-token bound and raised it to 8192 to evaluate depths 5 and 6. The separate AdaLN question condition remains bare and has no BOS.

#### Config System

Parse a recipe without initializing JAX devices:

```python
from adaloop_paligemma.config import PaliGemmaAdaLoopConfig

recipe = PaliGemmaAdaLoopConfig.from_file("adaloop/paligemma/configs/variant_a.yaml")
print(recipe.variant, recipe.adaptive_span, recipe.train_depth)
```

As in the paper's trainer, `training.seed` sets the data order and the AdaLN generator initialization, `lora.seed` sets the LoRA initialization, and the question pooler starts from the paper trainer's fixed initialization whatever the seeds.

#### Data

Training reads the `train` split of VisRO tasks (`--tasks`, default `core6`) in place (`--visro-root`), optionally followed by a replay JSONL file (`--replay-jsonl`, with `--replay-image-root` for relative image paths), exactly as the Qwen backend does; see [Data](../README.md#data) for the layout, the paper's replay set and the replay row format. Images are checked against their `image_sha256` when they are read. Each epoch shuffles the pooled rows with `random.Random(training.seed + 1_000_003 * epoch)`, keeps whole global batches, and shards them by host.

Text length is not checked at load time. Like the paper's loader, batch construction truncates a row that exceeds the 256-token training width: the prefix is right-truncated first so that the whole answer still fits (BOS is kept, but the newline separator may be cut), and only an answer that alone exceeds the budget is truncated, always ending in EOS. Before training, process zero computes path-independent digests of the ordered sample IDs, questions and answers and of every sample-to-image association (a VisRO image contributes its record's `image_sha256`, a replay image is hashed) and stores them with the data selection in the resume contract, so changed rows or image bytes cannot silently alter a resumed run.

The normal language trajectory retains PaliGemma's native BOS and uses `answer en {question}\n` before the answer. The AdaLN condition is tokenized separately from the raw question with `add_bos=false`, excludes EOS, and is right-padded only after the 128-token cap. Training uses the question for both. Evaluation intentionally omits the training cue: its signed `{question}` template is filled with the language prompt, giving `<bos>{prompt}\n`, while the condition remains the bare question. Both templates are signed in the recipe, and evaluation uses the paper runtime's 256-token prompt width. A `constant_question` recipe conditions on its `constant_text` instead of the question, in training and evaluation alike; the language input is unchanged.

#### Resuming

Numbered training snapshots include the complete parameter and Adam states, PRNG key, epoch and batch cursor, strict recipe, and a path-free data/runtime contract; `--keep-checkpoints` (default 2, 0 keeps all) sets how many complete snapshots are retained. Resuming skips the batches already consumed in the current epoch by index, without decoding their images. Resume from one explicit snapshot or the highest complete one:

```bash
adaloop-paligemma-train \
    --config adaloop/paligemma/configs/variant_a.yaml \
    --base-checkpoint paligemma-assets/paligemma2-3b-mix-448.b16.npz \
    --tokenizer paligemma-assets/paligemma_tokenizer.model \
    --visro-root data/VisRO --replay-jsonl /path/to/replay.jsonl \
    --output-dir outputs/paligemma_variant_a \
    --resume-from-checkpoint latest
```

`--export-dir` is written only after successful completion and is a compact, hashed initialization/inference artifact. It intentionally excludes optimizer and cursor state; keep the numbered snapshot when exact training resume is needed. The destination is checked at startup, after the dataset is loaded and fingerprinted and before the base checkpoint is loaded: it must not be a symlink, must be absent or an empty directory (the trainer never overwrites an export), must not be or contain `--output-dir`, and its nearest existing ancestor must be a writable directory. The check creates nothing, so a run that fails it can be restarted unchanged.

## Portable Checkpoints

Inspect a portable PaliGemma overlay without reconstructing the model:

```bash
adaloop-paligemma-inspect /path/to/checkpoint
```

Internal hashes are verified by default. Add `--base-checkpoint` and `--tokenizer` to verify the two external assets as well; `--skip-hashes` is a debug-only escape hatch for internal files. The inspector needs neither JAX nor Flax: it validates the manifest and file hashes and reads the tensor inventory from the overlay's msgpack headers, which must hold exactly the recipe's trainable leaves with their shapes and dtypes.

Reconstruct a verified overlay with the built-in runtime:

```python
from adaloop_paligemma.checkpoint import PaliGemmaCheckpointLoader

verified = PaliGemmaCheckpointLoader.from_pretrained(
    "/path/to/checkpoint",
    base_checkpoint="paligemma-assets/paligemma2-3b-mix-448.b16.npz",
    tokenizer="paligemma-assets/paligemma_tokenizer.model",
)
runtime = verified.reconstructed
```

See [`checkpoint_format.md`](../docs/checkpoint_format.md) for the portable layout and loader contract. A portable checkpoint initializes training or inference but does not contain optimizer, input-pipeline or RNG state for exact mid-run resume.

## License

Apache License 2.0. The [LICENSE](../../LICENSE) and [NOTICE](../../NOTICE) files at the repository root apply to this package; keep both with any redistribution. The checkpoints build on PaliGemma 2 and are subject to its terms.
