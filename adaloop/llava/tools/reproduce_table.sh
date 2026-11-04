#!/usr/bin/env bash
# Reproduce the paper's LLaVA table on the six core VisRO tasks: zero-shot, SFT, single-pass AdaLN
# and Variant A (AdaLoop: per depth 1..3, p=1 stopping with T=3) on the eval and ood splits, then
# print the table next to the paper's values.
#
#   adaloop/llava/tools/reproduce_table.sh [--set KEY=VALUE ...]   # overrides apply to all four evaluations
#
# Run from the repository root. Expects the base model, CLIP, the three trained output
# directories (outputs/sft, outputs/adaln_single_pass, outputs/adaloop) and the VisRO dataset at
# the paths of adaloop/llava/configs/eval/*.yaml (relative to the working directory).
#
# Environment:
#   NGPU   processes per evaluation (default 1); the outputs do not depend on it
#   TOL    allowed difference from the paper per cell, in points (default 0); the comparison is
#          printed either way and never fails the script
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LLAVA="$(dirname "${HERE}")"
NGPU="${NGPU:-1}"
for config in zeroshot sft adaln variant_a; do
  echo "=== ${config} ==="
  torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" -m adaloop_llava.cli.evaluate \
    --config "${LLAVA}/configs/eval/${config}.yaml" "$@"
done
python "${HERE}/make_results_table.py" --eval-root outputs/eval --tol "${TOL:-0}" --json outputs/eval/table.json
