#!/usr/bin/env bash
# One-command artifact reproduction (RTX 4050 6GB reference; CPU steps need no GPU).
# Usage: bash reproduce.sh [--quick]
#   --quick : CPU-only (roofline + router sims + stats + unit tests), ~2 min.
#   default : above + GPU standard suite N=10 (~30-60 min on RTX 4050).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"

echo "=== [1/4] unit tests (CPU) ==="
pytest tests/test_roofline.py tests/test_hybrid_router.py -q

echo "=== [2/4] roofline calibration + cross-GPU sweep (CPU) ==="
python3 scripts/run_roofline_analysis.py

echo "=== [3/4] hybrid router simulation + margin/robustness sweep (CPU) ==="
python3 scripts/run_hybrid_router_simulation.py
python3 scripts/sweep_router_margin.py
python3 scripts/analyze_benchmark_stats.py

if [[ "${1:-}" == "--quick" ]]; then
  echo "Quick mode: skipping GPU benchmark."
  exit 0
fi

echo "=== [4/4] GPU standard benchmark suite (N=10) ==="
python3 scripts/run_standard_benchmark_suite.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --samples-per-bench 10 --max-new-tokens 48 \
  --output experiments/16_hybrid_roofline/standard_suite_repro.json
echo "ALL DONE. Key artifacts in experiments/16_hybrid_roofline/"
