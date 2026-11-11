#!/usr/bin/env bash
# Mistral-7B-Instruct-v0.3 (pre-quantized bnb NF4, unsloth mirror) fits on the 6 GB GPU: exact PLD loop, GSM8K and HumanEval, 100 prompts each.
cd "$(dirname "$0")/../.."
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
export PYTHONPATH=src:scripts/lib
for cfg in "gsm8k_wp5_100 640" "humaneval_wp5_100 1024"; do
  set -- $cfg; ds=$1; cl=$2; B=data/benchmarks/$ds.jsonl
  out=results/kernels/wp8_mistral7b_${ds%%_*}.json
  [ -e $out ] && continue
  I=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('$B') if l.strip()))")
  $PY scripts/kernels/wp5_invariant_graph_pld.py --model unsloth/mistral-7b-instruct-v0.3-bnb-4bit --revision d5f623888f1415cf89b5c208d09cb620694618ee --prompt-ids $I --benchmark $B --max-cache-len $cl --use-m2 --ops lm_head,rmsnorm,sdpa,nf4 --output $out 2>&1 | tail -1 | cut -c1-500
done
echo MISTRAL_DONE
