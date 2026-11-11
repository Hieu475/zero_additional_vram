#!/usr/bin/env bash
# Interleaved extra reps (2,3): 3B hybrid on three sets, then all three methods for Qwen2.5-1.5B. Resumable.
cd "$(dirname "$0")/../.."; export PYTHONPATH=src:scripts/lib; mkdir -p logs results/drafter_runs
for rep in 2 3; do
  for b in gsm8k_wp5_100 humaneval_wp5_100 cnndm_wp5_100; do
    out=results/drafter_runs/${b}_hyb_n2_rep${rep}.json
    [ -e $out ] || python3 scripts/drafter/wp10_hybrid_graph.py --benchmark data/benchmarks/$b.jsonl --limit 100 --ckpt results/drafter/d2_k5.pt --k 3 --ref-every 10 --pld-min-n 2 --output $out > logs/hyb_${b}_n2_rep${rep}.log 2>&1
  done
  scripts/drafter/run_q15_eval.sh $rep
done
