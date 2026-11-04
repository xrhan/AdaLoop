# Portable checkpoint format

AdaLoop checkpoints are compact overlays on a pinned base model. They contain
every trained tensor needed to reconstruct the selected backend and do not
duplicate frozen base weights. Qwen, PaliGemma, and LLaVA use different native
weight formats, so each has its own strict manifest contract.

## Qwen2.5-VL overlay

### Directory layout

Every public checkpoint directory or Hub repository has these canonical files:

```text
checkpoint/
├── adaloop_manifest.json
├── adaloop_config.yaml
├── adaloop_runtime.json
├── adapter_config.json
├── adapter_model.safetensors
├── non_lora_trainables.safetensors
└── processor/                          # required canonical snapshot
    └── tokenizer/image-processor files ...
```

The three weight roles are intentionally separate:

| Manifest role | Canonical file | Contents |
| --- | --- | --- |
| `adapter_config` | `adapter_config.json` | PEFT LoRA topology |
| `adapter_weights` | `adapter_model.safetensors` | language q/k/v/o LoRA tensors |
| `non_lora_weights` | `non_lora_trainables.safetensors` | merger, question pooler, and AdaLN tensors |

SFT has no pooler or AdaLN tensors, but the sidecar still carries its trainable
merger. Variant A without modulation likewise omits conditioner/AdaLN keys.
Constant-question controls store exactly the tensors of their
question-conditioned twins, by name and shape; no condition vector is stored.

`adapter_config.json` must agree with the recipe's LoRA settings and pinned
base. The LoRA options that change which modules are adapted or how adapters
act (for example `exclude_modules`, `layers_to_transform`, `rank_pattern`,
`alpha_pattern`, `use_rslora`, or `use_dora`) must keep PEFT's defaults, as
training leaves them; absent keys take the default.
Both weight files must hold exactly the tensors, by name and shape, of the
model the loader builds: the recipe's model class wrapped by PEFT from
`adapter_config.json`, with LoRA on every text layer. Adapter tensors keep the
names PEFT saves, e.g.
`base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight`;
PEFT's loader skips some other spellings, such as one with the adapter name,
with only a warning, keeping untrained LoRA weights. Export, conversion, and
loading all refuse missing, unexpected, or mis-shaped tensors.

### Manifest contract

`adaloop_manifest.json` records:

- format version and Qwen2.5-VL backend;
- base model ID and immutable 40--64 character lowercase hexadecimal
  revision;
- recipe variant, half-open loop span, train depth, and readout mode;
- exact LoRA and conditioner settings;
- the recipe and runtime-contract filenames and SHA-256 values;
- one canonical path, format, and SHA-256 for each required weight role; and
- the exact path and SHA-256 of every file under `processor/`.

Paths must be portable and relative to the checkpoint root. Absolute paths,
home expansion, and `..` traversal are rejected. The loader cross-checks
duplicated topology fields against `adaloop_config.yaml`; neither file can
silently override the other. It also rejects missing, extra, or modified
processor files.

`adaloop_runtime.json` pins the inference-critical Qwen input limits. The
paper defaults are a 2048-token context (4096 for Variant C), 12,544 minimum
pixels, and 451,584 maximum pixels. These values must equal the
`preprocessing` block in the recipe. They are applied to the bundled
tokenizer and image processor at load time. Variant-B and Variant-C readout
assembly enforce the signed context limit, and every decode path, including
Hugging Face `generate` for SFT, AdaLN, and fixed-depth Variant A, refuses a
decode whose context plus new-token budget would exceed it.

Inspect metadata and verify every digest without downloading the base model:

```bash
adaloop-qwen-inspect /path/to/checkpoint --verify-hashes
```

### Loading

The high-level inference API reconstructs the model from the manifest:

```python
from adaloop_qwen.inference import AdaLoopQwenPipeline

pipe = AdaLoopQwenPipeline.from_pretrained(
    "/path/to/checkpoint",       # or a Hugging Face Hub repository ID
    device="cuda",
    attn_implementation="sdpa",
)
result = pipe.generate("image.png", "What is connected?", depth=3)
```

`generate` and `generate_p1` also accept `prompt=`, which replaces only the
language-model text and never reaches the AdaLN condition. Decoding is greedy
under a policy pinned in code (repetition penalty 1.05 for SFT, AdaLN, A, and
B, none for Variant C; stop only at `<|im_end|>`). No `generation_config.json`
setting is used, and the recipe's `generation` block sets only the default
answer budget.

For lower-level access:

```python
from adaloop_qwen.checkpoint import QwenCheckpointLoader

loaded = QwenCheckpointLoader.from_pretrained(
    "/path/to/checkpoint",
    device="cuda",
    merge_lora=True,
    verify_hashes=True,
)
model = loaded.model
processor = loaded.processor
recipe = loaded.config
```

The loader performs these steps in order:

1. resolve the local directory or Hub snapshot;
2. parse the manifest, verify files/hashes, and cross-check the recipe;
3. validate the runtime contract and bundled processor snapshot;
4. download the exact pinned Qwen2.5-VL base revision;
5. instantiate the topology named by the recipe;
6. attach the PEFT adapter and require exactly the adapter tensors it loads;
7. strictly restore the trained non-LoRA tensors;
8. optionally merge LoRA for inference; and
9. load the bundled processor and apply the signed input limits.

Use `merge_lora=False, is_trainable=True` to keep adapters trainable. A
trainable adapter cannot be merged. Hash verification is enabled by default.

### Exporting a trained model

The canonical exporter expects a live PEFT model and strict recipe:

```python
from adaloop_qwen.checkpoint import export_release_checkpoint
from adaloop_qwen.config import QwenAdaLoopConfig

recipe = QwenAdaLoopConfig.from_file("adaloop/qwen/configs/variant_a.yaml")
manifest_path = export_release_checkpoint(
    peft_model,
    recipe,
    "release-checkpoint",
    processor=processor,
)
print(manifest_path)
```

Only tensors with `requires_grad=True` and without a LoRA name are written to
the sidecar. This makes the configured trainability policy part of export
correctness: the exporter refuses a sidecar that is not exactly the merger,
question-pooler, and AdaLN tensors the loader restores, e.g. when part of a
module was frozen. The exporter requires an immutable base revision and fails if
PEFT does not produce the canonical safetensors files. The supplied processor
is first configured from the recipe and then saved into `processor/`; both it
and `adaloop_runtime.json` are covered by the final manifest.

Export is staged and validated before publication, with the manifest written
last as the completion marker. A non-empty destination is rejected by default.
Pass `overwrite=True` only to replace its canonical checkpoint files; unrelated
files such as trainer resume state are preserved. Adapter metadata must identify
the recipe's pinned base model, and the exported metadata is canonicalized to
the public model ID so local cache or cluster paths are never published.
PEFT's auto-generated `README.md` model card, which can embed such a path, is
not exported.

The trainer's final output directory is directly loadable and is also a
resume snapshot. Numbered intermediate `checkpoint-*` directories are
resume-format snapshots; the canonical portable export is written at
successful completion. A resume snapshot holds Accelerate's optimizer,
scheduler, and RNG state, a `model.safetensors` with only the trainable
tensors (unwrapped names, exact dtypes), the compact adapter and sidecar, the
recipe and run contract, and `training_state.json` with the step and data
position, written last as the completion marker. Resuming replays the
uninterrupted run's data order, learning-rate schedule, and RNG stream. The
frozen base is rebuilt from the pinned `base_revision` before the trainable
tensors are restored. Resume snapshots that store the full model are
rejected.
`adaloop_qwen.training.save_resume_checkpoint` and `load_resume_checkpoint`
use this format automatically; code that calls `accelerator.save_state` or
`load_state` directly must first call
`register_trainable_only_resume_state(accelerator)`. Resume-only files are
operational state and are not listed in the portable manifest.

After each periodic save, `--save-total-limit N` (default 2; 0 keeps all)
keeps the newest N complete numbered snapshots at or below the current step
and deletes older or incomplete ones. Snapshots it cannot verify or delete
are kept with a warning, and the final output directory is never deleted.
`--resume-from-checkpoint latest` picks the newest complete snapshot and
skips, with a warning, any it cannot inspect or verify.

Each resume snapshot includes `training_run_contract.json`, which records the
maximum sequence length, right-padding policy, pinned processor source and
revision, processor class names, chat-template SHA-256, full image-processor
configuration, actual attention implementation and mixed precision, world
size, per-device batch size, gradient accumulation, data-loader worker count,
and drop-last policy. It also records the data selection (VisRO tasks,
difficulty levels, per-task and replay row counts) and two path-independent
digests: one of the ordered sample IDs, questions, and answers, and one of
every sample's ID and image-byte SHA-256 (a VisRO image contributes its
record's `image_sha256`; replay images are hashed once at startup on the main
process); machine-specific paths are not serialized. Resume first
compares both the complete training recipe and this complete run contract. It
fails before loading optimizer or model state if either differs, naming each
differing field.

## PaliGemma2 overlay

PaliGemma checkpoints are compact Flax overlays on the exact public Big Vision
NPZ and SentencePiece model pinned by the recipe. The external asset paths are
provided only when loading; the recipe records their expected byte sizes and
SHA-256 values, never a workstation or cluster path.

### Directory layout

```text
checkpoint/
├── adaloop_manifest.json
├── adaloop_config.yaml
├── adaloop_runtime.json
└── trainable.params.msgpack
```

`trainable.params.msgpack` contains the language LoRA leaves, trainable image
connector, and, for conditioned recipes, the shared question pooler and all
in-span AdaLN leaves. Loaders derive this complete leaf set, with every shape
and dtype, from the recipe and refuse an overlay that omits, adds, reshapes,
or retypes a leaf, even if its manifest was re-signed to match: the runtime
would otherwise keep a missing leaf's base or initial value. The manifest
records the size and SHA-256 of the two sidecars (`adaloop_config.yaml`,
`adaloop_runtime.json`) and of the overlay. It also signs the exact overlay
inventory: every leaf's slash-separated path, shape, dtype, and byte size. Its
topology section duplicates the variant, half-open span, training depth, round
mode, readout, and base/tokenizer hashes so these values can be cross-checked
against the recipe rather than silently inferred.

`adaloop_runtime.json` repeats the inference-critical base, tokenizer,
backbone, preprocessing, generation, and parameter-overlay contracts. The
paper recipe uses a 448-pixel input, 256-token training and evaluation text
widths, an `answer en {question}` training template, a `{question}`
evaluation template that inference fills with the language prompt (the bare
question by default), a separate bare 128-token no-BOS question condition,
and deterministic answers with a default budget of 128 new tokens (the paper
setting, which `--max-new-tokens` overrides). Variants B and C bound their
packed sequence at 8192 tokens; at inference the bound also covers the
decoded answer. Variant C retains its separate 24-token draft cap.

Inspect the signed manifest and internal file hashes without loading JAX,
Flax, or the external base model; the tensor inventory is read from the
overlay's msgpack headers and checked against the recipe's complete leaf set:

```bash
adaloop-paligemma-inspect /path/to/checkpoint
```

Internal hashes are checked by default. Supplying `--base-checkpoint` and
`--tokenizer` verifies the external assets too.

To verify and reconstruct the runtime object:

```python
from adaloop_paligemma.checkpoint import PaliGemmaCheckpointLoader

verified = PaliGemmaCheckpointLoader.from_pretrained(
    "/path/to/checkpoint",
    base_checkpoint="/path/to/paligemma2-3b-mix-448.b16.npz",
    tokenizer="/path/to/tokenizer.model",
    verify_hashes=True,
)
runtime = verified.reconstructed
```

Validation proceeds in this order: strict manifest and portable relative
paths; internal file sizes and hashes; strict YAML/runtime parsing and mutual
consistency; overlay restore and an exact tensor-inventory match with both the
manifest and the recipe's trainable leaves; and exact external base/tokenizer
size and hash. Only then is the lazy built-in runtime hook invoked. Advanced
callers may supply a compatible
`reconstruction_hook(recipe, base_npz, tokenizer, overlay)` or request
`reconstruct=False` for validation only. Keeping mesh, sharding, and
compilation outside the artifact avoids serializing a private TPU topology.

### Exporting a fresh training parameter tree

Export a completed in-memory training tree with a strict frozen-leaf proof:

```python
from adaloop_paligemma.checkpoint import export_parameter_tree
from adaloop_paligemma.runtime import reconstruct_parameter_template

manifest = export_parameter_tree(
    params,
    "adaloop/paligemma/configs/variant_a.yaml",
    "/path/to/paligemma2-3b-mix-448.b16.npz",
    "/path/to/tokenizer.model",
    "/path/to/release-checkpoint",
    reconstruction_hook=reconstruct_parameter_template,
)
```

The exporter keeps exactly the recipe-defined trainable leaves and proves that
every omitted leaf still matches the reconstructed base. Training resume state
is saved separately.

## LLaVA-1.5 overlay

LLaVA checkpoints are compact PyTorch overlays on the stage-2 LLaVA-1.5-7B
base (448 px, 2×2 pooling), released with the checkpoints, and the CLIP
ViT-L/14-336 tower. The manifest records the base by identity and by the size
and SHA-256 of each of its files; the CLIP tower is pinned by Hub revision and
file hashes. Both directories are supplied at load time.

### Directory layout

```text
checkpoint/
├── adaloop_manifest.json
├── adaloop_config.yaml
├── adaloop_runtime.json
├── config.json
├── adapter_config.json
├── adapter_model.safetensors
└── non_lora_trainables.safetensors
```

| Manifest role | Canonical file | Contents |
| --- | --- | --- |
| `recipe` | `adaloop_config.yaml` | resolved recipe; `paths` and the data manifests are null |
| `runtime` | `adaloop_runtime.json` | inference contract |
| `model_config` | `config.json` | `LlavaConfig` with the `loop_*` keys |
| `adapter_config` | `adapter_config.json` | PEFT LoRA topology |
| `adapter_weights` | `adapter_model.safetensors` | language-model LoRA tensors |
| `non_lora_weights` | `non_lora_trainables.safetensors` | projector and loop core |

The sidecar holds the four `mlp2x_gelu` projector tensors and, for the loop
recipes, the loop core: 48 tensors for single-pass AdaLN (one AdaLN generator
per band layer and the question pooler), 52 for Variants A and B (plus the
sinusoidal round embedding) and 66 for Variant C (plus the draft feedback: a
second pooler and its gate), under the names the trainer saves, such as
`base_model.model.model.loop_core.adaln.2.to_adaln.weight`, in the dtypes they
were trained in. The adapter holds one A/B pair for each of the q/k/v/o and
gate/up/down projections of every language-model layer. `config.json` names the
CLIP tower by its Hub ID and records no `_name_or_path`; `adapter_config.json`
names the base by its ID and targets the language-model projections with the
trainer's regular expression.

### Manifest contract

`adaloop_manifest.json` is written last and records:

- format version and the `llava-1.5` backend;
- the variant (`sft`, `adaln`, `a`, `b`, or `c`);
- the base model ID and format, and the CLIP Hub ID and immutable revision,
  each with the size and SHA-256 of every file the loaders read:
  `config.json`, the tokenizer files, `generation_config.json` when present,
  and the safetensors weights (the index and every shard) for the base;
  `config.json`, `preprocessor_config.json`, and the weights for CLIP;
- the topology: loop mode, half-open span, training depth, step embedding,
  readout, fused readout, deep-supervision weights, the question condition
  (BOS included, at most 128 tokens, dimension 512), the AdaLN hidden size,
  the vision geometry (448 px, 2×2 pooling, layer -2, patch tokens), and the
  projector;
- the LoRA rank, alpha, dropout, bias, and target projections;
- the name, shape, and dtype of every sidecar tensor; and
- the role, size, and SHA-256 of each of the six files above.

`adaloop_runtime.json` pins the Vicuna-v1 conversation, the image geometry with
padding to a square, the question condition, the 2048-token context, and the
decoding policy: greedy, at most 32 new tokens, no repetition penalty, left
padding, an fp16 language model, and an fp32 loop core. It must equal the
contract derived from the recipe.

Validation needs neither Torch nor the base model:

```bash
adaloop-llava-inspect /path/to/checkpoint --verify-hashes
```

It checks, in order: the manifest and its portable relative paths; the size
and, with `--verify-hashes`, the SHA-256 of every file; that no
`non_lora_trainables.bin` or `adapter_model.bin` would shadow the canonical
files; the topology, LoRA settings, and identities of the manifest against
`adaloop_config.yaml`; the runtime contract against the recipe; the `loop_*`
and vision keys of `config.json` and the PEFT settings of
`adapter_config.json` against the recipe, with no local paths in either; and
the exact tensor inventories of both weight files, read from their safetensors
headers and derived from `config.json` and the recipe. The sidecar inventory
must also equal the manifest's.

### Loading

```python
from adaloop_llava.checkpoint import LlavaCheckpointLoader

loaded = LlavaCheckpointLoader.from_pretrained(
    "/path/to/checkpoint",
    base_model="/path/to/llava-v1.5-7b-448-s2",
    vision_tower="/path/to/clip-vit-large-patch14-336",
    device="cuda",
    dtype="float16",
)
model, tokenizer = loaded.model, loaded.tokenizer
```

After the checks above, the supplied base and CLIP directories must hold
exactly the fingerprinted files with the recorded sizes; the SHA-256 of their
configuration, tokenizer, and preprocessor files is checked by default, and
`verify_base_hashes=True` (`--verify-base-hashes`) also hashes the weights.
Their `config.json` must describe the same language model and CLIP width as
the checkpoint's. The model is then built as the paper evaluation built it:
base weights, the sidecar, the LoRA adapter merged into the weights, the CLIP
tower retargeted to 448 px, and the loop core reloaded in fp32 and compared bit
for bit with the file. `AdaLoopLlavaPipeline` in `adaloop_llava.inference`
wraps this loader for single-example inference.

### Exporting a trained model

`adaloop-llava-train` exports `<output-dir>/release` on the main process when
training finishes, unless `--no-export` is given. The export is
`export_release_checkpoint`, which is the conversion below applied to the
training output with the run's resolved recipe:

```python
from adaloop_llava.checkpoint import export_release_checkpoint

manifest_path = export_release_checkpoint("outputs/adaloop")
```

### Converting adapter directories

A training output directory, or one of its `checkpoint-<step>` directories,
converts when it holds
`adapter_config.json`, `adapter_model.safetensors`, `non_lora_trainables.bin`,
and `config.json`:

```bash
adaloop-llava-convert \
  --adapter-dir /path/to/run \
  --config adaloop/llava/configs/variant_a.yaml \
  --base-model /path/to/llava-v1.5-7b-448-s2 \
  --vision-tower /path/to/clip-vit-large-patch14-336 \
  --output-dir /path/to/release-checkpoint
```

Its `config.json` is read the way the evaluation loader reads it: a
missing `loop_use_adaln` means the conditioned loop, an SFT config's
`loop_start: null` with an inert `loop_steps: 3` means no loop, and
`loop_steps: 1` means single-pass AdaLN. It is then compared field by field
with the recipe: loop mode, band, depth, step embedding (sinusoidal for
Variant A, none for AdaLN), deep-supervision weights, fused readout, readout,
AdaLN use, condition length and dimensions, round-embedding size, vision
geometry, projector, and context length. `adapter_config.json` must match the
recipe's rank, alpha, dropout, bias, and target projections, with PEFT's
defaults for every option that changes the adapted modules; listed projection
names are rewritten as the trainer's language-model regular expression, which PEFT would otherwise also match in
the question pooler. Any contradiction is refused; for example, a loop run
trained without the round embedding does not convert with the Variant A
recipe.

`non_lora_trainables.bin` is loaded with `torch.load(weights_only=True)` and
written to safetensors under its own tensor names; every tensor must reload
with the same dtype and equal values. The adapter is copied byte for byte.
Both weight files must hold exactly the tensors of the recipe's model. The base
and CLIP directories are hashed into the manifest; `--base-fingerprint` reuses
the base and CLIP blocks of an earlier manifest, checking the directories by
size and small-file hashes. The checkpoint is staged next to the destination,
verified, and published with the manifest last. A non-empty destination needs
`--overwrite`, which replaces only the checkpoint files.

## Compatibility policy

All three loaders reject unsupported backends, unknown fields, topology
mismatches, missing or extra weight roles, malformed hashes, unsafe paths, and
unexpected tensors. Qwen LoRA validation requires every pinned language layer
to contain exactly one A/B pair for each q/k/v/o projection, and LLaVA LoRA
validation requires every language-model layer to contain exactly one A/B pair
for each q/k/v/o/gate/up/down projection. PaliGemma requires
the exact q/fused-kv/o LoRA leaves and exact connector/conditioner/AdaLN tree
defined by its recipe. A local absolute Qwen base path in `adapter_config.json`
is validated and rewritten to the canonical public model ID on export;
PaliGemma paths never enter the public recipe; LLaVA conversion replaces the
local CLIP and base paths of the metadata with their IDs and records the
recipe's paths as null. This strictness is deliberate: silently
substituting defaults can create a model that loads successfully but implements
a different recurrence or language interface. Metadata that contradicts the
recipe is refused.
