#!/usr/bin/env bash
# HumanEval 100 prompts: invariant PLD loop vs old row-wise exact loop, run back to back (A3 first, then WP5, then A3 again for drift check).
cd "$(dirname "$0")/../.."
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
export PYTHONPATH=src:scripts/lib
B=data/benchmarks/humaneval_wp5_100.jsonl
I=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('$B') if l.strip()))")
for tag in a3_1 inv a3_2; do
  out=results/kernels/wp5_he100_$tag.json
  [ -e $out ] && continue
  if [ $tag = inv ]; then
    $PY scripts/kernels/wp5_invariant_graph_pld.py --prompt-ids $I --benchmark $B --max-cache-len 1024 --use-m2 --ops lm_head,rmsnorm,sdpa,nf4 --output $out 2>&1 | tail -1 | cut -c1-400
  else
    $PY scripts/audit/a3_graph_pld_loop.py --prompt-ids $I --benchmark $B --max-cache-len 1024 --use-m2 --output $out 2>&1 | tail -1 | cut -c1-400
  fi
done
echo HE_ROWWISE_DONE
