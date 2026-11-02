#!/usr/bin/env bash
# Reproduction entry point for the B1 exact quantized speculative-decoding study.
# Usage: bash reproduce.sh [--quick]
#   --quick : CPU-only checks (unit tests + import audit of all scripts).
#   default : quick checks + one short GPU B1 pilot (needs CUDA, Triton, bitsandbytes).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PWD/scripts"

echo "=== [1/2] unit tests ==="
python3 -m pytest tests -q

if [[ "${1:-}" == "--quick" ]]; then
  echo "Quick mode: skipping GPU pilot."
  exit 0
fi

echo "=== [2/2] GPU B1 pilot (Qwen2.5-3B NF4, K=2, 3 fixed rows, 64 tokens) ==="
OUT="experiments/exact_quantized_feasibility/repro_$(date +%Y%m%d_%H%M%S)"
python3 scripts/run_b1_block_pilot.py \
  --output-dir "$OUT" \
  --quantized-linear-strategy triton_nf4_fixed_three_rows_group_scale_packed \
  --attention-strategy rowwise --physical-rows 3 --k 2 \
  --max-new-tokens 64 --per-task 1 --audit-state
echo "Done. Rows and manifest in $OUT"
