# AdaLoop architecture

AdaLoop divides a pretrained vision encoder into three consecutive parts:

```text
pixels -- P (prelude, once) --> z0
             z(t-1) -- R(.; q) --> zt       for t = 1, ..., T
             zt -- C (coda + connector) --> It
             language prefix(It, q) -- LM --> answer at depth t
```

`P` contains the vision blocks before the selected loop span, `R` is the
weight-tied recurrent span, and `C` contains the remaining vision blocks and
the vision-language connector. The released recurrence is **latent carry**:
the output state of `R` is the input state at the next round. It does not reset
to pixels or to the prelude output.

The pretrained vision weights remain frozen. Training updates the language
LoRA adapters, the connector, a shared question pooler, and the layer-specific
AdaLN generators in the loop span. Plain SFT omits the pooler/generators;
single-pass AdaLN applies the same conditioned span once and is the
parameter-matched no-recurrence control.

## Question conditioning

The condition input is the role-free question only. It contains no BOS, EOS,
answer cue, or chat-role token and is capped at 128 tokens. LLaVA is the one
exception: as in its paper runs, its condition starts with BOS, followed by at
most 127 question tokens. The language-model
input embeddings are detached, converted to FP32, RMS-normalized, and
projected from the language width to a 512-dimensional conditioning space.
One learned query uses eight-head attention to pool them, followed by a
residual MLP. The pooler is shared across conditioned vision blocks; each
block has its own zero-initialized AdaLN generator. The resulting condition is
reused at every recurrent round; LLaVA's Variant A, B and C recipes add a
learned sinusoidal round embedding to it before each round, and LLaVA's
Variant C also folds each round's draft into the next round's condition (see
[Variants A, B and C](variants_abc.md)).

Qwen's constant-question controls instead pool the fixed text
`Please answer the visual reasoning question.` for every example through the
same pooler and generators. They add no tensor, such as a learned null
condition, so they are parameter-matched to the question-conditioned models.
PaliGemma accepts the same mode as a release extension; the paper has no
PaliGemma constant-question control. At inference, an optional prompt, such
as the question followed by an answer-format instruction, replaces only the
language-model text; the condition remains the bare question (or the fixed
constant for constant-question controls), as in the paper's evaluation.

For an attention or MLP residual branch `f`, the modulated update is

```text
h <- h + (1 + alpha) * f((1 + gamma) * LayerNorm(h) + beta)
```

The multipliers are not squashed. AdaLN changes a branch's normalized input
and residual update; it does not directly normalize the recurrent state.

## Training across depth

At every supervised depth, the coda and connector map the recurrent state to
the language model, which predicts the same final answer. Static deep
supervision uses normalized geometric weights

```text
w_t = r ** (T - t) / sum_j r ** (T - j).
```

The paper uses `r=0.6` for Qwen and LLaVA and `r=0.8` for PaliGemma. Language
loss is averaged across each example's supervised answer tokens, then across
examples; prompt and padding tokens are ignored. LLaVA keeps the rank-mean
normalization of its paper runs instead: SFT and Variants B and C average over
every supervised token of the global batch, and the single-pass and Variant A
recipes scale the mean of the per-rank token means (see the
[LLaVA guide](../llava/README.md#loss-normalisation)).

Qwen also supports the paper's per-example sampled-depth objective for
Variant A, which samples one terminal depth from the same distribution and
computes one answer loss per example. With several processes, rank 0 draws
the depths for every row of the global microbatch and broadcasts them, so
rows on different GPUs receive independent depths, as in the paper runs;
every rank must therefore run every microbatch. A `generator` passed to
`sample_geometric_depths` or `compute_training_objective` must be a
`torch.Generator` on the sampling device; it is checked on every rank, so a
bad one raises everywhere instead of leaving other ranks waiting.

## Inference

A caller may generate at a fixed depth or apply the framework-neutral `p=1`
answer-stability policy. The latter compares adjacent greedy answers after
lowercasing and whitespace normalization, returns the later answer in the
first matching pair, and otherwise returns the configured fallback depth.
Task-specific canonicalization is deliberately not part of the halting rule.
Decoding is greedy with a policy pinned in code, never read from a
checkpoint's generation config, and every evaluated depth uses the same
decode policy as fixed-depth generation, so the answer at each depth matches
the fixed-depth answer. One exception: Qwen Variant A's fixed-depth and `p=1`
paths use different decoders, which agree in float32 but can rarely flip a
near-tie in BF16. The [Qwen guide](../qwen/README.md#inference) lists the
Qwen policy. The Qwen high-level implementation preprocesses and pools the
question once, then collects the full visual trajectory in one recurrent
execution through the maximum tested depth. It decodes only the successive
language readouts; Variant C retains its required fresh-KV reprefill at every
draft boundary. The PaliGemma pipeline evaluates depths sequentially and stops
immediately at the first matching adjacent pair; it never evaluates deeper
answers after the criterion fires.

LLaVA decodes greedily with at most 32 new tokens and no repetition penalty,
with an FP16 language model and an FP32 loop core, and reruns the vision tower
for every depth. Its decoding call matches the paper evaluation: it pins greedy
decoding, the token limit, and the KV cache, and any other decoding option
comes from the stage-2 base's `generation_config.json` when it has one; a
portable checkpoint's manifest pins that file by size and SHA-256. Its `generate_p1` evaluates depths in order and stops at the
first matching adjacent pair, like PaliGemma. The paper's LLaVA numbers come
from left-padded batches of 16 records per depth with halting applied
afterwards to the stored answers. For `p=1` this selects the same answers as
online stopping, but a floating-point near-tie can flip a token compared with
decoding one example at a time.

The language prefix differs across Variants A, B, and C. See
[`variants_abc.md`](variants_abc.md) for the exact layouts.

## Backend boundary

`common/` defines the variant names, layouts, depth weights, stopping, and the
VisRO training data and evaluation without depending on a tensor framework.
`qwen/` owns the PyTorch/CUDA implementation for Qwen2.5-VL; `paligemma/` owns
the JAX implementation for TPU or CUDA; and `llava/` owns the PyTorch
implementation for LLaVA-1.5. The three backends share semantics, not tensor
abstractions, recipe schemas, or a combined runtime environment; each backend
validates its recipes with its own schema.

The same concept can therefore carry a different name in each backend:

| Concept | Qwen2.5-VL | PaliGemma2 | LLaVA-1.5 |
| --- | --- | --- | --- |
| Loop span, zero-indexed `[start, end)` | `adaptive_span` | `adaptive_span` | `model.loop.band` |
| Loop span in `adaloop_manifest.json` | `loop_span` | `topology.adaptive_span` | `topology.loop_span` |
| Training depth | `train_depth` | `train_depth` | `model.loop.steps` |
| Deep supervision | `training.supervision: deep_geometric` and `training.geometric_ratio` | as Qwen | `model.loop.deep_sup_weight: "geometric:<ratio>"` |
| Readout of SFT/AdaLN, A, B, C | `readout.mode`: `single_pass`, `current_image`, `accumulated_images`, `interleaved_drafts` | as Qwen | `model.loop.readout`: `last`, `last`, `image_prefix`, `answer_trajectory` |
| Variant C draft length | `readout.draft_max_tokens` | as Qwen | fixed at 16 tokens |
| Loss normalization | `training.loss_normalization: per_example_answer_tokens` | `training.loss_normalization: per_example` (the same rule) | `train.loss_normalization: rank_mean` or `token_mean` |
| Hash check of the inspect CLI | opt-in `--verify-hashes` | on by default, `--skip-hashes` | opt-in `--verify-hashes` |
| Periodic checkpoints kept by training | `--save-total-limit` | `--keep-checkpoints` | `--save-total-limit` |
