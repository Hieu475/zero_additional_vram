#!/usr/bin/env bash
# Reproduction entry point for the B1 exact quantized speculative-decoding study.
# Usage: bash reproduce.sh [--quick | --b2]
#   --quick : CPU-only checks (unit tests + B2 manifest validation).
#   --b2    : quick checks + short GPU B2 checks (reject-path audit, per-step KV audit, lazy-LM-head benchmark).
#   default : quick checks + one short GPU B1 pilot (needs CUDA, Triton, bitsandbytes).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PWD/scripts"

echo "=== [1/2] unit tests and B2 manifest validation ==="
python3 -m pytest tests -q
python3 scripts/validate_activationvec_b2_manifest.py --manifest experiments/exact_quantized_feasibility/activationvec_b2_v3/experiment_manifest.json

if [[ "${1:-}" == "--quick" ]]; then
  echo "Quick mode: skipping GPU pilot."
  exit 0
fi

if [[ "${1:-}" == "--b2" ]]; then
  echo "=== [2/2] GPU B2 checks (Qwen2.5-3B NF4, K=2, 3 rows, development prompts) ==="
  OUT="experiments/exact_quantized_feasibility/repro_b2_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT"
  python3 scripts/audit_activationvec_reject_path_lazy_b2.py --benchmark data/benchmarks/gsm8k_eval.jsonl \
    --prompt-id gsm8k_000 --output "$OUT/reject_path.json"
  python3 scripts/audit_activationvec_fallback_kv_closed_loop.py --benchmark data/benchmarks/gsm8k_eval.jsonl \
    --prompt-ids gsm8k_000 gsm8k_001 --max-new-tokens 64 --output "$OUT/kv_closed_loop.json"
  python3 scripts/benchmark_b2_lazy_lm_head.py --benchmark data/benchmarks/gsm8k_eval.jsonl \
    --prompt-ids gsm8k_000 gsm8k_001 gsm8k_002 --timing-rounds 2 --output "$OUT/lazy_benchmark.json"
  echo "Done. B2 checks in $OUT"
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
