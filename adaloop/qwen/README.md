# AdaLoop for Qwen2.5-VL

[![arXiv](https://img.shields.io/badge/arXiv-2610.21323-b31b1b.svg)](https://arxiv.org/abs/2610.21323)&nbsp;
[![Checkpoints](https://img.shields.io/badge/Hugging%20Face-AdaLoop-blue.svg)](https://huggingface.co/Lyy0725/AdaLoop/tree/main/qwen)&nbsp;
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](../../LICENSE)&nbsp;

This is the official PyTorch implementation of AdaLoop for `Qwen/Qwen2.5-VL-3B-Instruct`. It supports the paper's plain SFT and single-pass AdaLN controls, recurrent Variants A/B/C, deep geometric supervision, Variant-A sampled-depth training on VisRO, fixed-depth inference, `p=1` answer-stability stopping, and the VisRO evaluation. It does not contain Qwen3, Qwen3-MoE or dataset generators. Commands run from the repository root.

## Initialization

Run from the repository root and install this backend with the common package, which automatically installs the VisRO reader and scorer:

```bash
pip install -e ./adaloop/common -e './adaloop/qwen[gpu]'
```

The environment is pinned in [`environment.yml`](environment.yml): Torch 2.6.0, Torchvision 0.21.0, Transformers 4.57.1, PEFT 0.17.1 and Accelerate 1.7.0; the pinned `gpu` extra has binary wheels for Python 3.10–3.12. SDPA is the portable default.

#### FlashAttention (optional)

To match the paper environment, install FlashAttention 2.7.4.post1 after the GPU dependencies and select it explicitly with `--attn-implementation flash_attention_2`. It needs Linux and an NVIDIA Ampere, Ada or Hopper GPU (for example A100, RTX 3090 or 4090, or H100); Turing (T4) and older GPUs are not supported.

PyPI has only FlashAttention's source distribution. Its `setup.py` imports Torch, so it must run with `--no-build-isolation`, after its other build-time imports are installed. It needs a CUDA toolkit even when it compiles nothing: it finds `nvcc` through `CUDA_HOME`, `PATH` or `/usr/local/cuda` and refuses releases older than 11.7. It then downloads the GitHub release's prebuilt wheel for the installed Python, Torch, CUDA major version and C++ ABI; v2.7.4.post1 has `linux_x86_64` wheels for Torch 2.6 with CUDA 12 and CPython 3.9–3.13. If no wheel matches or GitHub is unreachable, or with `FLASH_ATTENTION_FORCE_BUILD=TRUE`, it compiles from source instead. That needs the full toolkit of Torch's CUDA major version (CUDA 12 for the pinned Torch builds; see `python -c "import torch; print(torch.version.cuda)"`) and `ninja`, without which the build is serial and can take hours. `MAX_JOBS` caps the parallel compiler jobs; unset, the script allows at most half the CPU cores and one job per 9 GB of available memory, measured with `psutil`.

```bash
pip install -e './adaloop/qwen[gpu]'
pip install setuptools wheel packaging ninja psutil
nvcc --version  # 11.7 or newer; 12.x to compile against the pinned Torch
MAX_JOBS=4 pip install 'flash-attn==2.7.4.post1' --no-build-isolation
python -c "import flash_attn, torch; from flash_attn import flash_attn_func; q = torch.randn(1, 16, 2, 128, device='cuda', dtype=torch.bfloat16); print(flash_attn.__version__, flash_attn_func(q, q, q, causal=True).shape)"
```

The last command imports the compiled kernels and runs them once on the GPU. Without a CUDA toolkit, install the release wheel directly: choose `cxx11abiTRUE` or `cxx11abiFALSE` as `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` prints, and the interpreter's CPython tag, for example:

```bash
pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
```

## Base Model

Every recipe and checkpoint pins `Qwen/Qwen2.5-VL-3B-Instruct` at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`; it is fetched from the Hugging Face Hub at that revision when a model is built. Released checkpoints also bundle a canonical processor snapshot.

## Checkpoints

The released checkpoints are in [`Lyy0725/AdaLoop`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/qwen). Scores are macro accuracy on VisRO-Core6 (3,600 items per split), *measured from this checkpoint / reported in the paper*; Variants A, B and C use `p=1` stopping with at most 3 iterations.

| Checkpoint | Setting | ID | OOD |
| --- | --- | ---: | ---: |
| [`qwen/sft`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/qwen/sft) | SFT | 49.9 / 49.9 | 23.6 / 23.9 |
| [`qwen/adaln`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/qwen/adaln) | single-pass AdaLN | 59.9 / 60.0 | 30.2 / 30.2 |
| [`qwen/variant_a`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/qwen/variant_a) | AdaLoop-A | 64.6 / 64.6 | 32.5 / 32.4 |
| [`qwen/variant_b`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/qwen/variant_b) | AdaLoop-B | 62.1 / 62.1 | 31.1 / 31.1 |
| [`qwen/variant_c`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/qwen/variant_c) | AdaLoop-C | 63.0 / 63.0 | 32.0 / 32.0 |

The SFT checkpoint was trained by us with the paper's recipe; the others are the paper's runs. Download one with:

```bash
hf download Lyy0725/AdaLoop --include "qwen/variant_a/*" --local-dir checkpoints
```

## Evaluation

`adaloop-qwen-eval` evaluates a checkpoint on the VisRO `eval` and `ood` splits at the fixed depths 1..`--max-depth` and with `p=1` stopping over 1..`--p1-max-depth` (both default to the training depth), scores every answer with `visro.scoring`, and writes per-item rows, `summary.json` and a per-task table `results.md` (see [Evaluation](../README.md#evaluation)). The condition is the bare question, and the language-model text is the question plus the paper's answer-format instruction (`--prompt-style instruction`, the default; `question` passes the bare question; see [Inference](#inference) for the instructions). `--per-cell N` evaluates the first N items of each (task, difficulty) cell, for a quick check.

#### Sanity Check

1. **Download the checkpoint and the evaluation splits:**
    ```bash
    hf download Lyy0725/AdaLoop --include "qwen/variant_a/*" --local-dir checkpoints
    hf download Lyy0725/VisRO --repo-type dataset --local-dir data/VisRO --include "tasks/*/*/eval/*"
    hf download Lyy0725/VisRO --repo-type dataset --local-dir data/VisRO --include "tasks/*/*/ood/*"
    ```

2. **Check the checkpoint** (no Torch or base model needed):
    ```bash
    adaloop-qwen-inspect checkpoints/qwen/variant_a --verify-hashes
    ```

3. **Launch evaluation:**
    ```bash
    adaloop-qwen-eval --checkpoint checkpoints/qwen/variant_a \
        --visro-root data/VisRO --tasks core6 --splits eval,ood \
        --output-dir outputs/eval/qwen_variant_a
    ```
    The expected `p=1 (T=3)` scores in `results.md` are about 64.6 (ID) and 32.5 (OOD). The other checkpoints are evaluated the same way; the table above lists their expected scores.

## Inference

Fixed-depth greedy generation:

```bash
adaloop-qwen-infer \
    --checkpoint checkpoints/qwen/variant_a \
    --image example.png \
    --question "How many closed regions are present?" \
    --depth 3 --max-new-tokens 64
```

`--question` (`question=` in Python) is the bare question. It is the text the AdaLN conditioner pools, except that constant-question checkpoints always condition on their recipe constant and SFT and no-AdaLN checkpoints have no conditioner. It is also the language-model text unless `--prompt` (`prompt=`) supplies the full language-model text, which is used for the chat turn and every Variant C round. The paper's VisRO evaluation (and `adaloop-qwen-eval`) kept the condition bare and chose the prompt's answer-format instruction by task and answer type:

- maze multiple choice: `\nAnswer with just the letter.`
- maze true/false: `\nAnswer with only True or False.`
- maze counting: `\nAnswer with just the number.`
- other multiple choice: `\nAnswer with just the option label.`
- path tracing and open-ended questions: none (the prompt is the question)

The newline must be a real one; a `\n` typed inside double quotes reaches the model as a backslash followed by `n`. Use `printf` or bash/zsh `$'...\n...'` quoting:

```bash
Q="Which path reaches the star?"
adaloop-qwen-infer \
    --checkpoint checkpoints/qwen/variant_a \
    --image example.png \
    --question "$Q" \
    --prompt "$(printf '%s\nAnswer with just the letter.' "$Q")" \
    --depth 3
```

Adaptive generation evaluates increasing depths and returns the later member of the first adjacent normalized-answer match; if no pair matches, it returns the requested fallback depth:

```bash
adaloop-qwen-infer \
    --checkpoint checkpoints/qwen/variant_a \
    --image example.png \
    --question "How many closed regions are present?" \
    --p1 --max-depth 3 --fallback-depth 3
```

For A/B/C, the high-level `p=1` path preprocesses and question-pools once and runs the recurrent vision trajectory once through `--max-depth`; only the language readouts are decoded successively. Variant C still performs its exact fresh-KV draft reprefills and reuses each generated prefix draft at later depths.

The equivalent Python API:

```python
from adaloop_qwen.inference import AdaLoopQwenPipeline

pipe = AdaLoopQwenPipeline.from_pretrained("checkpoints/qwen/variant_a", device="cuda")

fixed = pipe.generate(
    "example.png",
    "Which path reaches the star?",
    prompt="Which path reaches the star?\nAnswer with just the letter.",
    depth=3,
)
adaptive = pipe.generate_p1("example.png", "How many clusters?", max_depth=3, fallback_depth=3)
print(fixed.answer)
print(adaptive.output.answer, adaptive.depth, adaptive.stabilized)
```

Image paths are opened with their EXIF orientation applied before RGB conversion, as the paper's Hugging Face processor did. In-memory PIL images are used as given; apply `PIL.ImageOps.exif_transpose` to camera photos you open yourself. SFT and single-pass AdaLN accept only depth 1.

**Decoding.** Decoding is greedy and pinned in code to the paper's evaluation; a checkpoint's `generation_config.json` is ignored. SFT, AdaLN and Variants A and B, including the constant-question controls, apply a repetition penalty of 1.05 with Hugging Face semantics (prompt tokens count too) at every depth, including each `p=1` depth. Variant C drafts (at most 24 tokens) and its final answer use no penalty and suppress every special token except `<|im_end|>`. Every path stops only at `<|im_end|>`, which is excluded from the returned `token_ids`; final answers are capped at 64 new tokens unless `--max-new-tokens` is given. Each `p=1` depth uses the fixed-depth decode policy. Variants B and C decode both paths with the same cached runtime, so the `p=1` answer at a depth matches the fixed-depth answer. Variant A's fixed-depth path uses Hugging Face `generate` instead: the two agree in float32, but in BF16 different kernels can rarely flip a near-tie. Every decode must fit the checkpoint's `max_length`; one whose context plus token budget exceeds it is refused.

## Training

Run the following command to train Variant A on the six core VisRO tasks plus a replay set with four GPUs:

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m adaloop_qwen.cli.train \
    --config adaloop/qwen/configs/variant_a.yaml \
    --visro-root data/VisRO \
    --replay-jsonl /path/to/replay.jsonl \
    --output-dir outputs/variant-a \
    --per-device-batch-size 1 \
    --gradient-checkpointing \
    --save-steps 1000 --log-steps 10
```

Training reads the `train` split of the VisRO tasks (`--tasks`, default `core6`) in place, optionally followed by a replay JSONL file (`--replay-jsonl`, with `--replay-image-root` for relative image paths); see [Data](../README.md#data) for the layout, the paper's replay set and the row format. Images are checked against their `image_sha256` when they are read and decoded with their EXIF orientation applied before RGB conversion, as the paper's processor loaded them.

#### Recipes

The YAML files in [`configs/`](configs/) are strict, executable recipes:

| Recipe | Visual passes | Language readout | Training objective |
| --- | ---: | --- | --- |
| `sft.yaml` | 1 | one image | terminal CE |
| `adaln.yaml` | 1 | one conditioned image | terminal CE |
| `variant_a.yaml` | 1–3 | current image state | deep geometric CE |
| `variant_a_sampled.yaml` | sampled 1–3 | current image state | one terminal CE per example |
| `variant_b.yaml` | 1–3 | accumulated image states | deep geometric CE |
| `variant_c.yaml` | 1–3 | images interleaved with greedy drafts | deep geometric CE |

The other three files are the paper's component controls, not additional AdaLoop variants. `adaln_constant.yaml` and `variant_a_constant.yaml` differ from `adaln.yaml` and `variant_a.yaml` only in the AdaLN condition: the same question pooler receives the paper's fixed text `Please answer the visual reasoning question.` for every example, while the language model still reads the real question. Each therefore has exactly its twin's trainable parameters (82,039,040, in identical tensors) and the same training and decoding settings; train it with the twin's command, changing only `--config` and `--output-dir`. The training CLI's `--constant-question` override departs from that control. A learned-null control (one extra learned 512-value condition) is not part of this code. `variant_a_no_adaln.yaml` is the recurrence-without-modulation control, with 66,199,808 trainable parameters (15,839,232 fewer).

All main recipes use a frozen vision encoder, a trainable vision-language merger, rank-64 language self-attention LoRA, per-example answer-token CE, AdamW with `beta2=0.99`, and BF16; they train for 40,000 steps at a global batch size of 64.

#### Config System

Recipes are parsed strictly. Unknown fields, duplicate YAML keys, and non-finite or overflowing numbers (for example `.nan` or `.inf`) are rejected. `base_revision` must be a 40–64 character lowercase hexadecimal commit hash; this is checked when the recipe is loaded, before training starts.

#### Customizing Training

A copied YAML may change settings such as steps, batch size, seed, warmup, learning rates and the minimum-LR ratio, the other AdamW constants, LoRA rank, alpha and dropout, the geometric ratio, loop span and depth, pooler width and heads, token budgets, preprocessing limits and gradient checkpointing, within their validated ranges. The following are fixed, and other values are rejected:

- `schema_version: 1`, `backend: qwen2.5-vl`, `vision_round_mode: latent_carry`, and LoRA `target_modules: [q_proj, k_proj, v_proj, o_proj]` in that order;
- `precision: bf16` (or `fp32`, which only CPU/debug runs record), `loss_normalization: per_example_answer_tokens`, `scheduler: cosine`, `train_vision_encoder: false`, `train_merger: true`, optimizer `name: adamw`, and `generation.do_sample: false`;
- a conditioner with `question_input: bare`, `include_bos: false` and `max_tokens: 128`; `constant_text` is required for `mode: constant_question` and rejected otherwise. `mode: none` (SFT and the Variant A no-AdaLN control only) omits every other conditioner field and requires `conditioner_lr: 0`;
- the variant's readout mode (`single_pass` for SFT and AdaLN, then `current_image`, `accumulated_images` or `interleaved_drafts` for A/B/C); Variant C also requires `question_mode: each_round`, `rollout_engine: reprefill`, `suppress_special_draft_tokens: true` and `separator: newline`;
- SFT has `adaptive_span: null` and every other variant a loop span; SFT and AdaLN use `train_depth: 1` with `terminal` supervision; A/B/C use a `train_depth` of at least 2 with `deep_geometric` supervision (or `sampled_geometric`, Variant A only) and a `geometric_ratio` in (0, 1].

The paper preprocessing contract is part of every recipe: `max_length=2048` (4096 for Variant C, as in the paper's Variant-C runs), `min_pixels=12544` and `max_pixels=451584`; a recipe without a `preprocessing` block gets these defaults. The training CLI applies these values to the tokenizer and the Qwen image processor. `--max-length`, `--min-pixels` and `--max-pixels` are explicit recipe overrides and are serialized into the resulting checkpoint. Final-answer generation defaults to 64 new tokens, the paper setting; Variant C's nonterminal drafts have a separate 24-token cap, and EOS ends a draft but is not stored in its later language context.

#### Batches, Rows and Epochs

`training.global_batch_size` must equal `processes * per-device batch size * gradient accumulation`. When `--gradient-accumulation-steps` is omitted it is inferred exactly; an inconsistent explicit value is refused. A fresh run also refuses a non-empty `--output-dir`; use a new directory or resume the existing run.

The normal language input uses Qwen's native chat template with the row's `question`; training rows have no separate prompt. The AdaLN condition is the role-free question (constant-question recipes condition on their fixed constant), tokenized without BOS, EOS, answer cue or chat-role tokens and capped at 128 tokens. Only answer tokens receive language-model loss. The pooled rows have equal weight; each epoch draws `torch.randperm` seeded with `training.seed + epoch`.

Long rows follow the paper's collator. For SFT, AdaLN and Variant A, a row longer than `--max-length` keeps its first `--max-length` tokens, so a long answer is cut; the image placeholders precede the answer, so a row that keeps an answer token keeps its whole image. Such a row is rejected only if truncation would leave no answer token. Variant B rows are not truncated: a row whose depth-T readout (all T image states, the question and the answer) exceeds `--max-length` is rejected, and the error reports that length. Variant C rejects a row whose worst-case trajectory (every draft at its token budget) exceeds `--max-length`. As in the paper runs, these errors are raised when the row's batch is reached during training, so filter such rows beforehand.

Each epoch ends after the last round of full per-device batches on every process, as in the paper's runs. Up to `processes * per-device batch size - 1` samples, a different subset each epoch, are left out, and the dataset must contain at least `processes * per-device batch size` rows. Gradient-accumulation windows may span an epoch boundary, and an update that does can contain a row twice; the paper's HF Trainer instead ran a shorter last update in each epoch. With fewer rows than `training.global_batch_size`, every update spans epochs.

#### Optimization

The cosine schedule counts optimizer updates for any number of processes; the logged `lr` (LoRA group) is the rate applied in that update, so it is 0 at step 1 during warmup. The model trains in train mode, so the recipes' gradient checkpointing is active: activation memory goes down and step time up. The paper's recurrent runs also used it; its single-pass SFT and AdaLN runs, including the AdaLN constant control, did not. Every dropout is 0, so it changes only memory and speed; `--no-gradient-checkpointing` turns it off. Recipes that share a seed share their LoRA initialization.

On CUDA, the training CLI enables TF32 matmuls and cuDNN convolutions, as the paper's training did; inference keeps PyTorch's defaults, as the paper's evaluation did. AdamW weight decay follows the paper trainer's rule: biases and normalization weights, recognized by `nn.LayerNorm` type or by a bias or norm name, are exempt, and every other trainable tensor decays, including the merger's `ln_q` RMSNorm weight.

For CPU debugging, pass `--mixed-precision no` to the training CLI and either add `--cpu` to `accelerate launch` or run `adaloop-qwen-train` without the launcher and with `ACCELERATE_USE_CPU=1` (`accelerate launch` overwrites that variable); otherwise Accelerate selects MPS on Apple Silicon. Such a run loads and trains the model in FP32 and records `precision: fp32` in its checkpoint recipes, so it never claims BF16. Without `--mixed-precision`, the CLI trains at the recipe's `precision`.

#### Resuming

Resume from an explicit Accelerate checkpoint or, with `latest`, the highest numbered valid checkpoint under the output directory; `latest` skips, with a warning, any checkpoint it cannot inspect or verify:

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m adaloop_qwen.cli.train \
    --config adaloop/qwen/configs/variant_a.yaml \
    --visro-root data/VisRO --replay-jsonl /path/to/replay.jsonl \
    --output-dir outputs/variant-a \
    --per-device-batch-size 1 --gradient-checkpointing \
    --resume-from-checkpoint latest
```

Numbered `checkpoint-<step>` directories store the trainable tensors (LoRA, merger, pooler and AdaLN) but not the frozen base, plus optimizer, scheduler and RNG state, a compact PEFT adapter and sidecar, the recipe, and the precise dataset epoch/offset: about 0.85 GiB for 3B Variant A instead of about 7.8 GiB with the frozen model. Resume rebuilds the frozen base from `base_revision` and rejects resume checkpoints in any other format. It replays the uninterrupted run's data order, learning rate and RNG stream exactly. On CPU, parameters then match bitwise on one or two processes and up to all-reduce rounding on more, because DDP rebuilds its gradient buckets after a restart. On GPUs the default CUDA kernels are not deterministic: the first resumed step reproduces the uninterrupted run's loss, and later steps drift slightly, as two runs resumed from the same checkpoint also do. `--save-total-limit N` (default 2, as in the paper's runs; 0 keeps all) keeps the newest `N` complete numbered checkpoints after each save and deletes older or incomplete ones. A checkpoint it cannot verify or delete is kept with a warning, and the final checkpoint in `--output-dir` is never deleted. At completion, the top-level output directory retains the resume state and is also exported in the hashed portable format, so it can be passed directly to `adaloop-qwen-infer`.

Every checkpoint also stores `training_run_contract.json`. It pins the command-level maximum sequence length, right-padding policy, processor source and revision/classes, chat-template digest and complete image-processor settings. It also records the actual attention implementation, mixed precision, world size, per-device batch size, gradient accumulation, worker count, drop-last policy, and the data selection: the VisRO tasks, difficulty levels and per-task row counts, the replay row count, `data.text_content_sha256` over the ordered sample IDs, questions and answers, and `data.image_content_sha256`, which signs every sample's ID with the SHA-256 of its image bytes (a VisRO image contributes its record's `image_sha256`; the main process hashes each replay image once at startup while the other processes wait; like every collective, that wait times out after 30 minutes, the paper's HF Trainer's default). Dataset paths themselves are not serialized, so the same data in another directory keeps the contract, while changed rows or image bytes break it. Resume requires an exact recipe and run-contract match before optimizer state is loaded, and a refusal names each differing field, such as `data.image_content_sha256`; changing data, images, preprocessing, parallelism or sequence limits requires a fresh run. See [`checkpoint_format.md`](../docs/checkpoint_format.md) for the resume-snapshot files.

#### Python Training Code

Python code that builds the training objective itself should pass `pad_token_id=processor.tokenizer.pad_token_id` to `release_readout_objective_hooks` for Variants B and C, as the CLI does; otherwise the ID is read from the model or generation config (151643 for the pinned base). It should also build `QwenDataCollator(..., readout_mode=config.readout.mode)`, as the CLI does; without it, Variant B rows are truncated like flat rows and an over-length error reports the truncated row's length instead of the full one. For a constant-question recipe, it should likewise build `QwenImageQADataset(..., constant_condition=config.conditioner.constant_text)`; without it, the conditioner pools each row's real question instead of the constant. `QwenImageQADataset` takes the examples of `adaloop_common.data.load_training_set`.

## Portable Checkpoints

A portable checkpoint contains the PEFT adapter, all trainable non-LoRA tensors, the strict recipe, a canonical processor snapshot and a runtime contract. The manifest SHA-256-hashes every one of those artifacts. Inspect it without loading Torch or the base model:

```bash
adaloop-qwen-inspect /path/to/checkpoint --verify-hashes
```

The loader accepts only adapter and sidecar tensors that match, by name and shape, the model it builds for the recipe; missing, unexpected or mis-shaped tensors are refused. The adapter config must also keep PEFT's defaults for LoRA options the recipe does not set, such as `exclude_modules` or `use_rslora`, and the sidecar check refuses an extra `null_condition` tensor. Options outside the released recipes (learned-null conditioning, timestep embeddings, progress losses, prefix-reset recurrence, draft-conditioned vision, a fixed image size or per-token CE) are rejected by every shipped recipe. See [`checkpoint_format.md`](../docs/checkpoint_format.md) for the full layout, load order, validation rules and programmatic export API.

## Reproducibility Notes

- Loop spans are half-open: `[2, 14]` means blocks 2 through 13.
- Every recurrent mode uses latent carry; the next iteration receives the previous loop state rather than pixels or a reset prelude state.
- Question-pooler token embeddings are detached and processed in FP32. The shared pooler produces a 512-dimensional condition; AdaLN generators are layer-specific and zero-initialized at their output.
- Variants A/B/C share the same recurrent vision execution and differ only in what the language model reads.
- `p=1` normalization changes case and whitespace only. It deliberately does not invoke a task scorer or reorder structured answers.
- Variant C performs a fresh language-model KV reprefill after every new image/draft segment by design.
- Test depth may exceed train depth, but such extrapolation is not guaranteed to be stable.

## License

Apache License 2.0. The [LICENSE](../../LICENSE) and [NOTICE](../../NOTICE) files at the repository root apply to this package; keep both with any redistribution. The checkpoints build on Qwen2.5-VL-3B-Instruct and are subject to its terms.
