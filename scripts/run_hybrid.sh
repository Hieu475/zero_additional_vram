#!/usr/bin/env bash
# Hybrid (in-graph PLD + learned drafter) runs, K=3. Resumable. usage: run_hybrid.sh "<bench> <min_n>" ...
cd "$(dirname "$0")/.."; export PYTHONPATH=src:scripts; mkdir -p logs results/drafter_runs
for cfg in "$@"; do set -- $cfg; b=$1; mn=$2
  out=results/drafter_runs/${b}_hyb_n${mn}.json
  [ -e $out ] || python3 scripts/wp10_hybrid_graph.py --benchmark data/benchmarks/$b.jsonl --limit 100 --ckpt results/drafter/d2_k5.pt --k 3 --ref-every 10 --pld-min-n $mn --output $out > logs/hyb_${b}_n${mn}.log 2>&1
done
