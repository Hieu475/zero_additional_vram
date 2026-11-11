#!/usr/bin/env bash
# Step 1 of docs/DRAFTER_2X_PLAN.md: interleaved reps of learned drafter (wp9, K=3) vs PLD K=3 (wp5, same invariant kernels). Resumable.
cd "$(dirname "$0")/../.."
PY=${PY:-python3}; export PYTHONPATH=src:scripts/lib
Q3=aa8e72537993ba99e69dfaafa59ed015b17504d1
mkdir -p results/drafter_runs logs
for b in gsm8k_wp5_100 humaneval_wp5_100 cnndm_wp5_100; do
  B=data/benchmarks/$b.jsonl
  IDS=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('$B') if l.strip()))")
  for rep in 1 2 3; do
    out=results/drafter_runs/${b}_d2k3_rep${rep}.json
    [ -e $out ] || $PY scripts/drafter/wp9_drafter_graph.py --benchmark $B --limit 100 --ckpt results/drafter/d2_k5.pt --k 3 --ref-every 10 --output $out > logs/wp9_${b}_rep${rep}.log 2>&1
    out=results/drafter_runs/pld_${b}_k3_rep${rep}.json
    [ -e $out ] || $PY scripts/kernels/wp5_invariant_graph_pld.py --model Qwen/Qwen2.5-3B-Instruct --revision $Q3 --prompt-ids $IDS --benchmark $B --max-cache-len 1024 --all-m --k 3 --ops lm_head,rmsnorm,sdpa,nf4 --output $out > logs/pld_${b}_rep${rep}.log 2>&1
  done
done
