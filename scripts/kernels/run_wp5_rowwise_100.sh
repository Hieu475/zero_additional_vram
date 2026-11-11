#!/usr/bin/env bash
# GSM8K and CNN/DM, 100 prompts: old row-wise exact loop vs invariant PLD loop, back to back (a3, inv, a3 drift check).
cd "$(dirname "$0")/../.."
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
export PYTHONPATH=src:scripts/lib
for cfg in "gsm8k 512" "cnndm 1024"; do
  set -- $cfg; ds=$1; cl=$2
  B=data/benchmarks/${ds}_wp5_100.jsonl
  I=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('$B') if l.strip()))")
  for tag in a3_1 inv a3_2; do
    out=results/kernels/wp5_${ds}100_$tag.json
    [ -e $out ] && continue
    if [ $tag = inv ]; then
      $PY scripts/kernels/wp5_invariant_graph_pld.py --prompt-ids $I --benchmark $B --max-cache-len $cl --use-m2 --ops lm_head,rmsnorm,sdpa,nf4 --output $out 2>&1 | tail -1 | cut -c1-300
    else
      $PY scripts/audit/a3_graph_pld_loop.py --prompt-ids $I --benchmark $B --max-cache-len $cl --use-m2 --output $out 2>&1 | tail -1 | cut -c1-300
    fi
  done
done
echo ROWWISE100_DONE
