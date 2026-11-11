#!/usr/bin/env bash
# Second model (Qwen2.5-1.5B): PLD K=3 vs learned drafter K=3 vs hybrid, per set, interleaved per rep. Resumable. usage: run_q15_eval.sh <rep>
cd "$(dirname "$0")/../.."; export PYTHONPATH=src:scripts/lib; PY=python3
R=989aa7980e4cf806f80c7fef2b1adb7bc71aa306; M=Qwen/Qwen2.5-1.5B-Instruct; rep=${1:-1}; O=results/drafter_runs
for b in gsm8k_wp5_100 humaneval_wp5_100 cnndm_wp5_100; do
  B=data/benchmarks/$b.jsonl
  IDS=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('$B') if l.strip()))")
  [ -e $O/q15_pld_${b}_rep$rep.json ] || $PY scripts/kernels/wp5_invariant_graph_pld.py --model $M --revision $R --prompt-ids $IDS --benchmark $B --max-cache-len 1024 --all-m --k 3 --ops lm_head,rmsnorm,sdpa,nf4 --output $O/q15_pld_${b}_rep$rep.json > logs/q15_pld_${b}_$rep.log 2>&1
  [ -e $O/q15_drafter_${b}_rep$rep.json ] || $PY scripts/drafter/wp9_drafter_graph.py --model $M --revision $R --benchmark $B --limit 100 --ckpt results/drafter/q15_k5.pt --k 3 --ref-every 10 --output $O/q15_drafter_${b}_rep$rep.json > logs/q15_drafter_${b}_$rep.log 2>&1
  [ -e $O/q15_hybrid_${b}_rep$rep.json ] || $PY scripts/drafter/wp10_hybrid_graph.py --model $M --revision $R --benchmark $B --limit 100 --ckpt results/drafter/q15_k5.pt --k 3 --ref-every 10 --pld-min-n 2 --output $O/q15_hybrid_${b}_rep$rep.json > logs/q15_hybrid_${b}_$rep.log 2>&1
done
