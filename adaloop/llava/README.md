# AdaLoop for LLaVA-1.5

[![arXiv](https://img.shields.io/badge/arXiv-2610.21323-b31b1b.svg)](https://arxiv.org/abs/2610.21323)&nbsp;
[![Checkpoints](https://img.shields.io/badge/Hugging%20Face-AdaLoop-blue.svg)](https://huggingface.co/Lyy0725/AdaLoop/tree/main/llava)&nbsp;
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](../../LICENSE)&nbsp;

This is the official PyTorch implementation of AdaLoop for LLaVA-1.5-7B (Vicuna-7B-v1.5 with CLIP ViT-L/14-336 run at 448 px and 2×2 pooling). It covers the three settings the paper trained on LLaVA (plain SFT, single-pass AdaLN and Variant A), Variants B and C, zero-shot evaluation of the stage-2 base, per-depth evaluation and `p=1` answer-stability halting. It does not implement the constant-question controls or sampled-depth training for LLaVA; those are available in the [Qwen](../qwen/README.md) and [PaliGemma](../paligemma/README.md) backends.

Training and evaluation read the public VisRO release through `adaloop_common`, as in the other backends. The evaluation (`adaloop-llava-eval`) is configuration-driven and generates in batches; [`tools/`](tools/) rebuilds the paper's LLaVA table from its outputs. Commands run from the repository root.

## Overview

| Setting | Recipe | `variant` | `model.loop.mode` | Band applications | Round embedding | Deep supervision |
|---|---|---|---|---|---|---|
| SFT (LoRA) | [`configs/sft.yaml`](configs/sft.yaml) | `sft` | `none` | – (plain CLIP) | – | – (plain LM loss) |
| Single-pass AdaLN | [`configs/adaln.yaml`](configs/adaln.yaml) | `adaln` | `single_pass` | 1 | none | weight 1.0 |
| Variant A (AdaLoop) | [`configs/variant_a.yaml`](configs/variant_a.yaml) | `a` | `adaloop` | 3 | sinusoidal | 9/49, 15/49, 25/49 (`geometric:0.6`) |
| Variant B | [`configs/variant_b.yaml`](configs/variant_b.yaml) | `b` | `adaloop`, readout `image_prefix` | 3 | sinusoidal | – (one loss on the answer) |
| Variant C | [`configs/variant_c.yaml`](configs/variant_c.yaml) | `c` | `adaloop`, readout `answer_trajectory` | 3 | sinusoidal | – (one loss on the answer) |

The vision tower runs the features of the second-to-last CLIP layer, so 23 layers execute. The loop splits them into three segments:

```
image ─► prelude: layers 0–1 (once)
           │
           ▼
         band: layers 2–7, applied K times with shared, frozen CLIP weights ◄── condition c_k
           │                                                                  = pool(question) + round_emb(k)
           ▼
         coda: layers 8–22 (once, on the last iterate)
           │
           ▼
         drop CLS ─► 2×2 average pool (1024 → 256 tokens) ─► mlp2x_gelu projector ─► Vicuna-7B
```

- **Question condition.** The question tokens are embedded with the language model's input embeddings, and a `QuestionAttentionPooler` (one learned query, 8 heads) pools them into a 512-d condition.
- **AdaLN-Zero.** For each band layer, a small MLP maps the condition to shift, scale and gate for the attention and MLP branches: `x ← x + (1 + g)·f(LN(x)·(1 + s) + b)`. The last projection is zero-initialised, so at initialisation every band application is the original CLIP block. CLIP stays frozen and the modulation generators are shared across rounds.
- **Trainable parameters.** All three settings train LoRA (r = 128, α = 256, on q/k/v/o/gate/up/down of every language-model layer) and the projector. The loop settings also train the *loop core*: six AdaLN generators, the question pooler and, for Variant A, the round embedder. New modules are kept in fp32.
- **Deep supervision.** During training, the coda, pooling and projector run on the iterate of every round k = 1..K, and each readout gets its own answer loss. The losses are combined with normalised geometric weights (r = 0.6, largest on the last round). The schema also accepts `model.loop.deep_sup_weight: uniform` and `explicit:W1,...,WK`; the recipes use only `geometric:0.6`.
- **Variants B and C.** Variant B splices the projected readouts of all K rounds, in round order, into the image position of one language-model pass (K × 256 image tokens). Variant C runs the rounds one at a time. After every round but the last, the model greedily decodes 16 tokens from that round's readout and keeps those before the first EOS; this draft is pooled by a second question pooler and added to the next round's condition through a zero-initialised tanh gate, and it enters the language-model context. With `head` and `tail` the conversation text before and after `<image>`, the context is `head <image₁> tail D₁ \n <image₂> tail D₂ \n <image₃> tail`, followed by the answer. In training the drafts are decoded without gradients by the model being trained (LoRA dropout on). Both variants train with one answer loss and no deep supervision. At depth d, Variant B reads rounds 1..d and Variant C drafts after rounds 1..d−1; depth 3 is the trained layout.
- **Halting (p = 1, T = 3).** Depth d means d band applications. The answer at depth t is compared with the answer at depth t−1 after lowercasing and collapsing whitespace; the loop stops at the first t ∈ {2, …, T} where they agree, and otherwise at T. T̄ is the mean number of band applications.

#### LLaVA-Specific Behaviour

Each of these keeps the numerics of the paper's LLaVA runs, although they differ from the Qwen and PaliGemma contracts.

1. **Round embedding.** Variant A adds a learned sinusoidal embedding of the round index to the pooled condition before every round: frequency dimension 64, maximum period 32, then Linear–SiLU–Linear (295,936 parameters). Single-pass AdaLN has none. The other backends reuse the condition unchanged at every round.
2. **BOS in the condition.** The question condition is BOS followed by at most 127 question tokens (128 in total). The other backends exclude BOS.
3. **Rank-mean loss normalisation** (`train.loss_normalization: rank_mean`, see [Loss Normalisation](#loss-normalisation)) for single-pass AdaLN and Variant A; Variants B and C use the SFT normalisation (`token_mean`). The other backends average per example.
4. **Fused readout.** Variant A runs its three readouts through the language model as one batch. This equals three separate passes up to floating-point associativity.
5. **Own stage-2 base.** The recipes train from a LLaVA-1.5-7B stage-2 checkpoint at 448 px with 2×2 pooling, released with the AdaLoop checkpoints (see [Base Models](#base-models)).
6. **Bare-question evaluation prompt.** The paper's LLaVA evaluation passed the bare question as the language-model text (`eval.prompt_style: question`); Qwen and PaliGemma append an answer-format instruction.
7. **Variant C drafts.** Each draft also updates the next round's AdaLN condition, through the draft pooler and gate; it is 16 decoded tokens cut at the first EOS, followed by a newline (token 13), and the conversation text around `<image>` is repeated after every image block. The Qwen and PaliGemma Variant C condition every round on the question only and use 24-token drafts.

LLaVA validates the band as an `adaloop_common.LoopSpan`, and its recipes are checked by the package's own schema (`adaloop_llava.config`).

## Initialization

Python 3.10 or newer is required; the pinned `gpu` extra has binary wheels for Python 3.10–3.12 (NumPy 1.26.4 and SentencePiece 0.2.0 have none for newer versions), and `environment.yml` uses Python 3.11. The stack is pinned in the `gpu` extra and in [`environment.yml`](environment.yml): Torch 2.6.0, Transformers 4.57.6, PEFT 0.17.1, Accelerate 1.15.0, Tokenizers 0.22.2, SentencePiece 0.2.0, safetensors 0.6.2, Pillow 11.3.0 and NumPy 1.26.4. Transformers is pinned to 4.57.6 because the paper's loss normalisation depends on how its `Trainer` passes loss arguments to the model, so this backend needs an environment of its own: the Qwen backend pins Transformers 4.57.1.

```bash
conda env create -f adaloop/llava/environment.yml
conda activate adaloop-llava
pip install -e ./adaloop/common -e './adaloop/llava[gpu]'
```

Run the commands from the repository root. The common package automatically installs the VisRO reader and scorer, whose only dependency is Pillow.

Training uses FlashAttention-2 by default (`model.attn_implementation: flash_attention_2`), which needs an NVIDIA Ampere or newer GPU. The `flash` extra pins `flash-attn==2.7.4.post1`; install it with `--no-build-isolation` after the GPU dependencies (the [Qwen guide](../qwen/README.md#flashattention-optional) describes the build). Without it, pass `--attn-implementation sdpa`; this is numerically slightly different from the paper runs. On GPUs older than Ampere, Transformers also rejects `train.tf32` and, depending on the Torch build, `train.bf16`; add `--set train.tf32=false --set train.bf16=false` there, which trains an fp32 language model with an fp16 CLIP tower.

## Base Models

| Default path | What | Source |
|---|---|---|
| `checkpoints/clip-vit-large-patch14-336` | CLIP ViT-L/14-336 | [`openai/clip-vit-large-patch14-336`](https://huggingface.co/openai/clip-vit-large-patch14-336), revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |
| `checkpoints/llava-v1.5-7b-448-s2` | Stage-2 LLaVA-1.5-7B at 448 px with 2×2 pooling: the zero-shot row and the starting point of all three settings | [`Lyy0725/AdaLoop`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/llava/base/llava-v1.5-7b-448-s2), `llava/base/llava-v1.5-7b-448-s2` |

```bash
for p in "*.json" "*.bin" "*.txt"; do
    hf download openai/clip-vit-large-patch14-336 --revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1 \
        --include "$p" --local-dir checkpoints/clip-vit-large-patch14-336
done
hf download Lyy0725/AdaLoop --include "llava/base/llava-v1.5-7b-448-s2/*" --local-dir checkpoints
mv checkpoints/llava/base/llava-v1.5-7b-448-s2 checkpoints/
```

The stage-2 base is Vicuna-7B-v1.5 trained with the LLaVA-1.5 recipe at 448 px with 2×2 average pooling of the CLIP patch grid (256 image tokens): stage 1 pre-trains the projector, stage 2 fine-tunes on the LLaVA-1.5 `mix665k` instruction data. The checkpoint holds the language model, the projector and the tokenizer but no CLIP weights; the CLIP tower is always loaded from `paths.vision_tower`.

## Checkpoints

The released checkpoints are in [`Lyy0725/AdaLoop`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/llava). Scores are macro accuracy on VisRO-Core6 (3,600 items per split), *measured from this checkpoint / reported in the paper*; Variants A, B and C use `p=1` stopping with at most 3 iterations. The Variant B and C checkpoints were trained by us with the same recipe and are not in the paper.

| Checkpoint | Setting | ID | OOD |
| --- | --- | ---: | ---: |
| [`llava/sft`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/llava/sft) | SFT (LoRA) | 50.4 / 50.4 | 24.8 / 24.8 |
| [`llava/adaln`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/llava/adaln) | single-pass AdaLN | 57.7 / 57.7 | 29.6 / 29.6 |
| [`llava/variant_a`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/llava/variant_a) | Variant A (AdaLoop) | 61.9 / 61.8 | 31.2 / 31.1 |
| [`llava/variant_b`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/llava/variant_b) | Variant B | 59.3 / – | 29.7 / – |
| [`llava/variant_c`](https://huggingface.co/Lyy0725/AdaLoop/tree/main/llava/variant_c) | Variant C | 60.0 / – | 30.6 / – |

```bash
hf download Lyy0725/AdaLoop --include "llava/variant_a/*" --local-dir checkpoints
```

#### Paper Results

The paper's table on the six core VisRO tasks (`core6`) with LLaVA-1.5-7B at 448 px. Accuracy is in %. Overall is the average over the six tasks; every task has 600 rows per split, so it equals the micro average. ID is the `eval` split and OOD the `ood` split; the per-task columns are on the ID split. Params counts trainable parameters.

| Method | Params (M) | OOD | ID | Count | Flow | Glyph | Metro | Maze | Path | T̄ |
|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot (stage-2 base) | 0 | 8.6 | 10.4 | 6.0 | 0.2 | 0.5 | 13.7 | 29.5 | 12.7 | 1 |
| SFT (LoRA) | 340.8 | 24.8 | 50.4 | 48.3 | 18.0 | 62.8 | 51.7 | 59.5 | 62.3 | 1 |
| Single-pass AdaLN | 350.2 | 29.6 | 57.7 | 52.5 | 29.5 | 82.0 | 53.0 | 63.5 | 65.7 | 1 |
| Variant A (p=1, T=3) | 350.4 | **31.1** | **61.8** | 56.8 | 38.7 | 84.7 | 59.0 | 64.2 | 67.3 | 2.2 (OOD 2.3) |

Variant A at a fixed depth of 3 (no halting) reaches ID 62.11 (Count 56.67, Flow 40.00, Glyph 85.00, Metro 57.67, Maze 65.67, Path 67.67) and OOD 31.00.

Per-task accuracy on the OOD split:

| Method | Count | Flow | Glyph | Metro | Maze | Path | Overall |
|---|---|---|---|---|---|---|---|
| Zero-shot | 0.0 | 0.0 | 0.0 | 12.7 | 26.2 | 12.5 | 8.6 |
| SFT (LoRA) | 2.3 | 1.2 | 24.7 | 37.8 | 34.5 | 48.2 | 24.8 |
| Single-pass AdaLN | 2.3 | 4.2 | 37.3 | 41.3 | 36.3 | 55.8 | 29.6 |
| Variant A (p=1, T=3) | 2.3 | 6.0 | 42.3 | 43.3 | 33.3 | 59.3 | 31.1 |
| Variant A, fixed depth 3 | 2.2 | 5.5 | 42.2 | 42.8 | 33.7 | 59.7 | 31.0 |

The Params column counts LoRA (319,815,680), the projector (20,979,712) and the loop core (0 for SFT, 9,356,544 for single-pass AdaLN, 9,652,480 for Variant A): 340,795,392, 350,151,936 and 350,447,872 trainable parameters.

## Evaluation

```bash
adaloop-llava-eval --config adaloop/llava/configs/eval/zeroshot.yaml     # the stage-2 base, depth 1
adaloop-llava-eval --config adaloop/llava/configs/eval/sft.yaml          # depth 1
adaloop-llava-eval --config adaloop/llava/configs/eval/adaln.yaml        # depth 1
torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.evaluate \
    --config adaloop/llava/configs/eval/variant_a.yaml                   # depths 1, 2, 3 and p=1 (T=3)
torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.evaluate \
    --config adaloop/llava/configs/eval/variant_b.yaml                   # the same for Variant B
torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.evaluate \
    --config adaloop/llava/configs/eval/variant_c.yaml                   # the same for Variant C
```

The trained run is read from `eval.adapter`, which defaults to the training output directory of the setting. It can also be a portable checkpoint, which is first verified against its manifest together with the base and CLIP directories of `paths`; both give the same outputs.

#### Sanity Check

1. **Download the checkpoint, the [base models](#base-models) and the evaluation splits:**
    ```bash
    hf download Lyy0725/AdaLoop --include "llava/variant_a/*" --local-dir checkpoints
    hf download Lyy0725/VisRO --repo-type dataset --local-dir data/VisRO --include "tasks/*/*/eval/*"
    hf download Lyy0725/VisRO --repo-type dataset --local-dir data/VisRO --include "tasks/*/*/ood/*"
    ```

2. **Check the checkpoint** (no Torch or base model needed):
    ```bash
    adaloop-llava-inspect checkpoints/llava/variant_a --verify-hashes
    ```

3. **Launch evaluation:**
    ```bash
    torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.evaluate \
        --config adaloop/llava/configs/eval/variant_a.yaml \
        --set eval.adapter=checkpoints/llava/variant_a
    ```
    The configuration reads the base models from `checkpoints/` and VisRO from `data/VisRO`. The expected `p=1 (T=3)` scores in `outputs/eval/adaloop/results.md` are about 61.9 (ID) and 31.2 (OOD); [Paper Results](#paper-results) lists the per-task values.

Other checkpoints and variations:

```bash
# an intermediate training checkpoint, in a directory of its own
adaloop-llava-eval --config adaloop/llava/configs/eval/variant_a.yaml \
    --set eval.adapter=outputs/adaloop/checkpoint-20000 --set eval.name=adaloop_20k
# one split only, and the per-depth curve up to depth 6, in a directory of its own
adaloop-llava-eval --config adaloop/llava/configs/eval/variant_a.yaml --set 'eval.splits=[ood]' \
    --set eval.max_depth=6 --set eval.name=adaloop_curve
```

Give every evaluation that differs from the standard one its own `eval.name`. Otherwise it writes into `outputs/eval/<setting>/`, the directory that `make_results_table.py` reads.

An evaluation

1. **reads the items** of each split in `eval.splits` (`eval`, `ood`) for `eval.tasks` under `eval.visro_root` (at most `eval.per_cell` per task and difficulty when set), and checks that every reference answer scores as correct, before the model is loaded;
2. **loads the model.** With an adapter, the base model is loaded, the projector and loop core are applied and the LoRA is merged; without one (zero-shot), the base model alone is loaded. The language model and CLIP run in fp16 with SDPA attention. The loop core is reloaded in fp32 and checked bit for bit against the file;
3. **generates.** For each depth 1..`eval.max_depth`: Vicuna-v1 prompt `<image>\n{question}`, left padding, batch size 16, greedy decoding with at most 32 new tokens and no repetition penalty. The code sets only greedy decoding, the token limit and the KV cache; any other decoding option comes from the base's `generation_config.json` when it has one, and a portable checkpoint's manifest pins that file by size and SHA-256. The prediction is the stripped decoded text. Loop models get the question condition and `loop_steps = depth`; at depth d, Variant B reads the readouts of rounds 1..d, and Variant C first drafts after rounds 1..d−1 from the evaluation prompt, then reads the d image blocks interleaved with the d−1 drafts. The loop mode and depth semantics come from the checkpoint's `config.json`, and a model without a loop accepts only `max_depth: 1`;
4. **scores** the predictions with `visro.scoring` and applies `p=1` stopping over depths 1..`eval.p1_max_depth` (the Variant A, B and C configs set 3).

With several processes (`eval.shard: batch`), process r generates batches r, r+W, … of the same batch partition, so the outputs are identical to a single-process run.

**Outputs** go to `outputs/eval/<eval.name>/`: `resolved_eval_config.yaml`, `<split>/depth<d>.jsonl` and `<split>/p1_T<T>.jsonl` (one scored row per item), `summary.json` and `results.md`, in the format the [shared evaluation contract](../README.md#evaluation) describes.

**The full table.** [`tools/reproduce_table.sh`](tools/reproduce_table.sh) runs the four evaluations on both splits and prints the table with every cell next to the paper's value; the table can also be rebuilt from existing outputs:

```bash
NGPU=8 TOL=0.17 adaloop/llava/tools/reproduce_table.sh
python adaloop/llava/tools/make_results_table.py --eval-root outputs/eval --tol 0.17
```

`TOL` / `--tol` is the allowed difference per cell in points; 0.17 (one record of a 600-row task) covers the scorer revision noted under [Numerics](#numerics). `--strict` turns a difference beyond the tolerance into an error.

## Inference

`adaloop-llava-infer` answers one question about one image:

```bash
adaloop-llava-infer --checkpoint checkpoints/llava/variant_a \
    --base-model checkpoints/llava-v1.5-7b-448-s2 --vision-tower checkpoints/clip-vit-large-patch14-336 \
    --image example.png --question "How many red circles are there?" --p1 --json
```

```python
from adaloop_llava.inference import AdaLoopLlavaPipeline

pipe = AdaLoopLlavaPipeline.from_pretrained(
    "checkpoints/llava/variant_a",
    base_model="checkpoints/llava-v1.5-7b-448-s2",
    vision_tower="checkpoints/clip-vit-large-patch14-336",
    device="cuda",
)
fixed = pipe.generate("example.png", "How many red circles are there?", depth=3)
halted = pipe.generate_p1("example.png", "How many red circles are there?")
print(fixed.answer, halted.answer, halted.depth, halted.stabilized)
```

- **Verification.** Before loading, the checkpoint's own files are checked against the manifest and against each other, and the supplied base and CLIP directories must hold the files the checkpoint was made with: the same sizes, and the same SHA-256 for the configuration and tokenizer files. `--verify-base-hashes` also hashes the weight files (several GB); `--no-verify-hashes` checks sizes only.
- **Depth.** `--depth N` (`depth=`) answers at one depth. The default is the training depth: 1 for SFT and single-pass AdaLN, 3 for Variants A, B and C. An SFT checkpoint has no loop and only depth 1. Every depth reruns the vision tower.
- **Halting.** `--p1` (`generate_p1`, Variants A, B and C) evaluates depths 1, 2, … in order and stops at the first answer that repeats the previous depth's answer after case and whitespace normalisation (`adaloop_common.select_p1_answer`). Without a repeat it returns the answer at `--fallback-depth`, which defaults to `--max-depth` (the training depth, 3).
- **Prompt.** `--question` is the bare question: the question condition of the loop and the user turn of the prompt. `--prompt` replaces only the language-model text, for example to add an answer-format instruction; it never reaches the condition.
- **Numerics.** Decoding is the evaluation code on a batch of one record, in fp16 by default (`--dtype`). The paper's numbers come from batches of 16 left-padded records per depth, with halting applied afterwards to the stored answers. The two agree except where a floating-point near-tie, which batch composition can flip, decides a token.

## Training

Run the following commands to train the settings with 8 GPUs:

```bash
torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.train \
    --config adaloop/llava/configs/sft.yaml --output-dir outputs/sft
torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.train \
    --config adaloop/llava/configs/adaln.yaml --output-dir outputs/adaln_single_pass
torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.train \
    --config adaloop/llava/configs/variant_a.yaml --output-dir outputs/adaloop \
    --base-model checkpoints/llava-v1.5-7b-448-s2 --vision-tower checkpoints/clip-vit-large-patch14-336 \
    --visro-root data/VisRO --replay-jsonl data/llava_replay/replay.jsonl
torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.train \
    --config adaloop/llava/configs/variant_b.yaml --output-dir outputs/adaloop_b
torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.train \
    --config adaloop/llava/configs/variant_c.yaml --output-dir outputs/adaloop_c
```

This is one process per GPU with plain DDP through the Hugging Face `Trainer`; a single process also works (`adaloop-llava-train` with the same arguments). The run names of `sft.yaml`, `adaln.yaml` and `variant_a.yaml` (`sft`, `adaln_single_pass`, `adaloop`) are the paper's, and so are the evaluation output directories that the table tools read; Variants B and C use `adaloop_b` and `adaloop_c`.

#### Data

| Default path | Content | Rows |
|---|---|---|
| `data/VisRO` | The downloaded VisRO release (`tasks/<category>/<task>/{train,eval,ood}/*.tar`); the recipes train on the `train` split of `core6` and evaluate on `eval` (ID, 6 tasks × 4 difficulties × 150) and `ood` (6 tasks × 600, all `xxhard`) | 600,000 / 3,600 / 3,600 |
| `data/llava_replay/replay.jsonl` | The replay set, one `{id, image, question, answer}` object per line; not part of VisRO | 100,000 in the paper |

`data.visro_root` and `data.replay_jsonl` (`--visro-root`, `--replay-jsonl`) locate them; `data.tasks` selects the tasks and `--set data.replay_jsonl=null` trains without replay. The [shared data contract](../README.md#data) describes the shard layout, the checks made when shards and images are read, and how the paper's replay set was built. The VisRO rows (tasks, shards and records in dataset order) and then the replay rows are pooled into one list with equal weight per row. Each sample is a single turn, `USER: <image>\n{question} ASSISTANT: {answer}</s>` (Vicuna v1), and the loss covers the answer and `</s>` only. The image is padded to a square with the CLIP mean colour.

#### Config System

Each recipe is a flat file that lists every key. A recipe starts with `schema_version: 1`, `backend: llava-1.5` and `variant`, which must agree with `model.loop.mode` and `model.loop.readout`. The `base_model` and `vision_tower` blocks are read-only identities: the stage-2 base the recipes train from and the CLIP tower at its Hub revision. The paper's three recipes differ only in `variant`, `run_name` and the `model.loop` block; the Variant B and C recipes also set `model.model_max_length: 4096` and `train.loss_normalization: token_mean`.

Each command-line flag is a dotted override of the recipe: `--output-dir` sets `paths.output_dir`, `--base-model` and `--vision-tower` the model directories, `--visro-root`, `--tasks`, `--replay-jsonl` and `--replay-image-root` the data, and `--per-device-batch-size`, `--gradient-accumulation-steps`, `--max-steps`, `--save-steps`, `--save-total-limit`, `--num-workers`, `--attn-implementation`, `--seed` and `--resume-from-checkpoint` the matching `train.*` and `model.*` keys. `--set KEY=VALUE` passes any other override, with the value parsed as YAML; quote values that contain brackets.

The shared recipe:

- 40,000 steps with a global batch of 64 (8 GPUs × 8 × 1 accumulation step), seed 42.
- AdamW with β = (0.9, 0.95) and no weight decay. The learning rate is 1e-4 for LoRA and the loop core and 5e-5 for the projector, with a cosine schedule and 3% warmup. Gradients are clipped at 1.0.
- bf16 with tf32, FlashAttention-2, and non-reentrant gradient checkpointing.
- LoRA r = 128, α = 256, dropout 0.05, on all linear projections of the language model.
- Answer-only loss, maximum length 2048 (4096 for Variants B and C). The question condition is at most 128 tokens.

The paper runs used one node with 8 × 80 GB GPUs. With the fused readout, Variant A runs 3 × 8 = 24 language-model rows per GPU per step.

#### Customizing Training

```bash
# a short debug run
adaloop-llava-train --config adaloop/llava/configs/variant_a.yaml --output-dir outputs/adaloop_debug \
    --max-steps 20 --save-steps 10
# smaller GPUs: keep the global batch at 64 with gradient accumulation
torchrun --standalone --nproc_per_node 8 -m adaloop_llava.cli.train --config adaloop/llava/configs/variant_a.yaml \
    --output-dir outputs/adaloop --per-device-batch-size 4 --gradient-accumulation-steps 2
torchrun --standalone --nproc_per_node 4 -m adaloop_llava.cli.train --config adaloop/llava/configs/variant_a.yaml \
    --output-dir outputs/adaloop --gradient-accumulation-steps 2
# no FlashAttention-2
adaloop-llava-train --config adaloop/llava/configs/sft.yaml --output-dir outputs/sft --attn-implementation sdpa
# GPUs older than Ampere (no FlashAttention-2, tf32 or bf16)
adaloop-llava-train --config adaloop/llava/configs/sft.yaml --output-dir outputs/sft --attn-implementation sdpa \
    --set train.bf16=false --set train.tf32=false
```

Training warns when the global batch is not 64. Under the default `rank_mean` loss normalisation, `train.rank_mean_scale` keeps the paper's loss scale for any GPU count and accumulation (see [Loss Normalisation](#loss-normalisation)).

#### Outputs and Resuming

Outputs go to `--output-dir`:

- `adapter_model.safetensors` and `adapter_config.json`: the LoRA adapter.
- `non_lora_trainables.bin`: the projector and the loop core.
- `config.json`: the model config, including `loop_mode` and the `loop_*` keys that rebuild the tower. A `config.json` without `loop_mode` is normalised in memory when loaded.
- `resolved_config.yaml`: the fully resolved recipe.
- `checkpoint-<step>/`: the same files plus optimizer, scheduler and RNG state, every 1000 steps; only the latest is kept.
- `release/`: the portable checkpoint, exported by the main process when training finishes (see [Portable Checkpoints](#portable-checkpoints)). `--no-export` skips it; `adaloop-llava-convert` exports the same checkpoint later.

Unlike the Qwen backend, which refuses a non-empty output directory, the recipe default `train.resume: auto` resumes: starting the same command again continues from the latest `checkpoint-<step>`. `--resume-from-checkpoint never` refuses a directory that already has checkpoints, and `--resume-from-checkpoint <dir>` resumes from a given checkpoint. Resuming needs Torch 2.6 or newer, because Transformers 4.57 restores the optimizer state with `torch.load` only there.

At startup, training checks the trainable set: LoRA must be on the language-model projections only, and with the paper's dimensions the total must equal the Params column of [Paper Results](#paper-results) (350,447,872 for Variant B and 354,649,857 for Variant C, whose draft feedback has 4,201,985). Single-pass AdaLN and Variant A runs also log `loop_loss_<k>`, the unweighted loss of each round.

## Portable Checkpoints

A portable checkpoint is a compact overlay on the stage-2 base and the CLIP tower. It holds no base weights and no local paths, and `adaloop-llava-train` exports one to `<output-dir>/release` at the end of a run:

| File | Content |
|---|---|
| `adaloop_manifest.json` | Written last. The format version; the base and CLIP identity with the size and SHA-256 of each of their files; the topology (variant, loop mode, half-open span, training depth, step embedding, readout, deep supervision, question condition, vision geometry); the LoRA settings; the name, shape and dtype of every sidecar tensor; the size and SHA-256 of the six files below |
| `adaloop_config.yaml` | The resolved recipe, with `paths` and the dataset locations recorded as null |
| `adaloop_runtime.json` | The inference contract: Vicuna-v1 conversation; 448 px, 2×2 pooling, padding to a square; a question condition of at most 128 tokens including BOS; greedy decoding with at most 32 new tokens and no repetition penalty; fp16 language model with an fp32 loop core |
| `config.json` | The model config with the `loop_*` keys; `mm_vision_tower` is the CLIP Hub id |
| `adapter_config.json`, `adapter_model.safetensors` | The LoRA adapter: one A/B pair for each projection of every language-model layer |
| `non_lora_trainables.safetensors` | The projector (4 tensors) and the loop core (48 tensors for AdaLN, 52 for Variants A and B, 66 for Variant C) |

An adapter directory, such as a training output or one of its `checkpoint-<step>` directories, converts with the recipe it was trained with:

```bash
adaloop-llava-convert --adapter-dir outputs/adaloop --config adaloop/llava/configs/variant_a.yaml \
    --output-dir outputs/adaloop/release --base-model checkpoints/llava-v1.5-7b-448-s2 \
    --vision-tower checkpoints/clip-vit-large-patch14-336
adaloop-llava-inspect outputs/adaloop/release --verify-hashes
```

Conversion reads the run's `config.json` the way the evaluation loader does (the paper runs recorded `model_type: llama` for their `LlavaLlamaForCausalLM`; the portable config says `llava_llama`) and compares it with the recipe field by field: loop mode, band, depth, step embedding, deep supervision, fused readout, readout, condition length and dimensions, vision geometry and projector. It also compares `adapter_config.json` with the recipe's LoRA settings and refuses any contradiction; for example, a loop run without the step embedding does not convert with the Variant A recipe. Both weight files must hold exactly the tensors of the recipe's model. `non_lora_trainables.bin` is rewritten to safetensors under the same tensor names, and every tensor must reload bit for bit. The base and CLIP directories are hashed into the manifest; `--base-fingerprint <adaloop_manifest.json>` reuses the hashes of an earlier conversion against the same base. The checkpoint is staged and verified before it is published, and a non-empty destination needs `--overwrite`. `adaloop-llava-inspect` checks the manifest, the files and the tensor inventories without Torch and without the base model, and prints the manifest.

#### Key Names

The released files spell some settings differently, and their names are fixed. The code uses these names:

| Recipe | `config.json` | Manifest `topology` | Code |
|---|---|---|---|
| `model.loop.band: [2, 8]` | `loop_start`, `loop_end` | `loop_span` | the band (half-open, checked as an `adaloop_common.LoopSpan`) |
| `model.loop.steps` | `loop_steps` | `train_depth` | the training depth; at inference the depth (`--depth`, `eval.max_depth`, `set_active_depth`), passed to the model as `loop_steps` |
| `model.loop.readout`: `last`, `image_prefix`, `answer_trajectory` | `loop_readout_mode` | `readout` | Variants A, B, C; Variant C code says *trajectory* for the interleaved drafts and *draft* for one decoded draft |
| `data.condition_max_length` | `condition_max_length` | `condition.max_tokens` | the question condition length, BOS included |
| `model.loop.step_embed_max_steps` | `loop_step_embed_max_steps` | `step_embed_max_steps` | recorded only: the sinusoidal round embedding has no cap |

In the sidecar, `question_conditioner` is the question pooler and `draft_feedback.answer_pooler` pools the Variant C draft tokens.

## Reproducibility Notes

#### Loss Normalisation

`train.loss_normalization: rank_mean` is the default and the recipe of single-pass AdaLN and Variant A. It is kept on purpose, because the SFT and loop runs of the paper normalise their loss differently:

- **SFT** (`mode: none`) passes the `Trainer`'s `num_items_in_batch` to the language-model loss. The objective is the global token mean over all GPUs and accumulation micro-batches.
- **Loop settings** (`single_pass`, `adaloop`) use the deep-supervision path. Each round's loss is the mean over the supervised tokens of the local micro-batch, and `num_items_in_batch` is not used. Transformers 4.57 still multiplies the loss by the number of processes, because the model accepts loss keyword arguments. With 8 GPUs × 1 accumulation step, the effective objective is therefore **8 × the mean over GPUs of the per-GPU token means**, weighted over rounds.
- The package reproduces this on any topology: the loop loss is multiplied by `rank_mean_scale / (world_size × grad_accum)`, with `train.rank_mean_scale: 8`. That factor is exactly 1 on the paper's 8 × 1 setup, so the paper runs' code path is unchanged there.
- The logged `loss` and `grad_norm` of the loop runs are therefore 8× a per-GPU token mean. Gradient clipping at 1.0 acts on that scaled gradient, so the scale is not neutral under Adam.
- **Variants B and C** have one answer loss, computed in one language-model pass that receives `num_items_in_batch`: the SFT normalisation. Their recipes set `train.loss_normalization: token_mean`, which validation requires for them.

`train.loss_normalization=token_mean` gives the loop path the same normalisation as SFT: for each round, Σ CE / `num_items_in_batch`. It is meant for new experiments; it is **not** the paper recipe and its effect on the results has not been measured.

#### Numerics

- **Training precision.** The language model is in bf16 and the CLIP tower is cast to bf16. The projector is loaded in bf16 and upcast, so it starts from the bf16-rounded stage-2 projector. The loop core and LoRA weights are fp32. The AdaLN generators and the question pooler compute in fp32 with autocast disabled.
- **Evaluation.** fp16 with SDPA, LoRA merged, batch size 16, left padding, greedy decoding. Batched fp16 generation can depend on batch composition and on the GPU or attention kernel, so near-tie answers can differ from a one-example-at-a-time run. Keep `eval.batch_size` and the record order unchanged to compare per-row predictions with the paper's outputs.
- **Scorer.** `visro.scoring` gives the same per-item result as the scorer that produced the paper's OOD and zero-shot numbers; this was checked on 72,000 saved LLaVA predictions. For the ID numbers of the three trained rows, compact multi-digit Path answers were not split with the scene-label vocabulary. That affects at most one ID record: 0.03 points on overall ID and 0.17 points on the ID Path column. Compare those cells with `--tol 0.17`.
- **Retraining.** This code seeds model construction (`train.seed`); the paper runs did not. GPU bf16 and FlashAttention kernels are non-deterministic. A retrained model therefore matches the paper within run-to-run noise, not bit for bit. GPU retraining was not repeated for this release.
- **Sampling and DDP.** Training uses the `Trainer`'s default random sampler, seeded with 42, over the pooled rows in dataset order. `train.dataloader_num_workers` does not affect the sample order. DDP runs with `find_unused_parameters: false`, since every trainable parameter receives a gradient in every setting. With `true` the gradients are identical.

## Package Layout

```
llava/
  configs/
    sft.yaml, adaln.yaml, variant_{a,b,c}.yaml the five flat recipes
    eval/                                     zeroshot / sft / adaln / variant_{a,b,c} evaluation configs
  tools/
    make_results_table.py                     evaluation outputs -> paper table, compared with the paper
    reproduce_table.sh                        the four evaluations + the paper table
  adaloop_llava/
    config.py                                 recipe schema, overrides, mode presets, validation
    errors.py                                 typed exceptions
    checkpoint.py                             portable checkpoints: export, conversion, validation, loading
    inference.py                              AdaLoopLlavaPipeline: fixed-depth and p=1 single-example inference
    generation.py                             batched greedy generation, shared by the pipeline and the evaluation
    conversation.py, constants.py, mm_utils.py Vicuna-v1 prompt, image token, image and prompt tokenisation
    model/
      llava_llama.py                          LlavaLlamaForCausalLM: plain path, deep-supervised loop path, Variants B and C, generate
      llava_arch.py                           vision modules, multimodal input splicing
      loading.py                              build for training, load for evaluation, config normalization
      loop_utils.py                           loop modes, readouts, loop config defaults, deep-supervision weights
      fp32.py                                 fp32 master weights of the loop core and the projector
      multimodal_encoder/                     CLIP tower (448 px, pooling) and the looped AdaLN tower + loop core
      multimodal_projector/                   mlp2x_gelu projector (fp32)
    data/                                     Vicuna-v1 tokenisation with answer masking, training dataset and collator
    training/                                 training entry point, Trainer subclass
    eval/                                     per-depth generation for the evaluation, sharded over processes
    cli/                                      adaloop-llava-train, -infer, -inspect, -convert, -eval
```

## License and Acknowledgements

The code is under the Apache License 2.0 ([LICENSE](../../LICENSE)); the citation is in the [repository README](../../README.md#citation). Parts of the code are adapted from [LLaVA](https://github.com/haotian-liu/LLaVA) (Copyright 2023 Haotian Liu), from the FastChat and Stanford Alpaca code that LLaVA's training script adopts, and from the CLIP implementation of [Transformers](https://github.com/huggingface/transformers), all under the Apache License 2.0. Those files keep the upstream copyright and license notice and state the AdaLoop modifications, and the repository's [NOTICE](../../NOTICE) lists them.

The base model and every checkpoint derived from it, including the trained adapters, derive from Vicuna-7B-v1.5 and are subject to the Llama 2 Community License Agreement. CLIP ViT-L/14-336 is provided by OpenAI under its own terms. This repository contains no model weights; the stage-2 base released with the AdaLoop checkpoints is distributed under the Llama 2 Community License Agreement. The replay data the paper used derives from the LLaVA-NeXT instruction data; follow the licenses of the original datasets (for example COCO, GQA, OCR-VQA, TextVQA and Visual Genome).

This backend builds on [LLaVA](https://github.com/haotian-liu/LLaVA), [Vicuna](https://github.com/lm-sys/FastChat), [CLIP](https://github.com/openai/CLIP), and the Hugging Face [Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft) libraries. If you use it, please also cite:

```bibtex
@inproceedings{liu2023llava,
  title     = {Visual Instruction Tuning},
  author    = {Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  booktitle = {NeurIPS},
  year      = {2023}
}
@inproceedings{liu2024improved,
  title     = {Improved Baselines with Visual Instruction Tuning},
  author    = {Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
  booktitle = {CVPR},
  year      = {2024}
}
@misc{vicuna2023,
  title  = {Vicuna: An Open-Source Chatbot Impressing {GPT-4} with 90\%* {ChatGPT} Quality},
  author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and
            Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and
            Xing, Eric P.},
  url    = {https://lmsys.org/blog/2023-03-30-vicuna/},
  year   = {2023}
}
@inproceedings{radford2021clip,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and
               Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and
               Krueger, Gretchen and Sutskever, Ilya},
  booktitle = {ICML},
  year      = {2021}
}
```
