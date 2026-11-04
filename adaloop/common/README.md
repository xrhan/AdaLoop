# AdaLoop Common Contracts

[![arXiv](https://img.shields.io/badge/arXiv-2610.21323-b31b1b.svg)](https://arxiv.org/abs/2610.21323)&nbsp;
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](../../LICENSE)&nbsp;

This is the framework-independent part of the official AdaLoop implementation, shared by the [Qwen](../qwen/README.md), [PaliGemma](../paligemma/README.md) and [LLaVA](../llava/README.md) backends. It automatically installs the [VisRO reader and scorer](https://github.com/Lyy-iiis/VisRO) from a pinned source revision and intentionally on neither PyTorch nor JAX. Install it together with a backend, see [Initialization](../README.md#initialization).

## Contents

`adaloop_common` contains:

- the Variant A/B/C names and the loop-span interval (`Variant`, `LoopSpan`);
- the reference geometric depth weights (`geometric_depth_weights`);
- answer-stability selection (`normalize_for_stability`, `select_p1_answer`);
- the symbolic Variant A/B/C language-prefix layouts (`layout_notation`);
- the training data of every backend (`adaloop_common.data`): VisRO shards, read with `visro.dataset`, plus an optional replay JSONL, with the checks and fingerprints of the resume contracts (see [Data](../README.md#data));
- the VisRO evaluation (`adaloop_common.evaluation`): the paper's prompts, the reference-answer check, scoring with `visro.scoring`, `p=1` stopping over fixed-depth answers, and the result files and tables (see [Evaluation](../README.md#evaluation)).

Backend packages implement these semantics and own all tensor, tokenizer and checkpoint-weight code; each validates its recipes with its own schema. PaliGemma's `p=1` stopping uses the common answer normalization, LLaVA's `p=1` inference uses `select_p1_answer`, and LLaVA validates its loop band as a `LoopSpan`.

## License

Apache License 2.0. The [LICENSE](../../LICENSE) and [NOTICE](../../NOTICE) files at the repository root apply to this package; keep both with any redistribution.
