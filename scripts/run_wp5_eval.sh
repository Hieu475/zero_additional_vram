#!/usr/bin/env bash
# WP5 end-to-end evaluation: exact PLD decode loop with M-invariant kernels vs (i) exact baseline graph with the same kernels and
# (ii) the earlier row-wise exact loop (A3, stock kernels). Same prompts, same session.
cd "$(dirname "$0")/.."
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
export PYTHONPATH=src:scripts
ids() { $PY -c "import json,sys;print(' '.join(json.loads(l)['id'] for l in open('$1') if l.strip())[:100000])" | tr ' ' '\n' | head -$2 | tr '\n' ' '; }
for cfg in "gsm8k_eval 50 448" "cnndm_eval 20 1024" "humaneval_eval 20 1024"; do
  set -- $cfg; ds=$1; n=$2; cl=$3
  I=$(ids data/benchmarks/$ds.jsonl $n)
  out=results/kernels/wp5_final_${ds}_k2.json
  [ -e $out ] || $PY scripts/wp5_invariant_graph_pld.py --prompt-ids $I --benchmark data/benchmarks/$ds.jsonl --max-cache-len $cl --use-m2 --ops lm_head,rmsnorm,sdpa,nf4 --output $out 2>&1 | tail -1 | cut -c1-600
  out=results/kernels/wp5_final_${ds}_a3_rowwise.json
  [ -e $out ] || $PY scripts/a3_graph_pld_loop.py --prompt-ids $I --benchmark data/benchmarks/$ds.jsonl --max-cache-len $cl --use-m2 --output $out 2>&1 | tail -1 | cut -c1-600
done
echo WP5_EVAL_DONE
