#!/usr/bin/env bash
# Phi-3.5-mini (head_dim 96, MHA, fused qkv/gate_up, LongRoPE short-factor regime): exact PLD graph loop, GSM8K 100 prompts.
cd "$(dirname "$0")/.."
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
export PYTHONPATH=src:scripts
B=data/benchmarks/gsm8k_wp5_100.jsonl
I=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('$B') if l.strip()))")
out=results/kernels/wp5_scale_phi35_gsm8k100.json
[ -e $out ] || $PY scripts/wp5_invariant_graph_pld.py --model microsoft/Phi-3.5-mini-instruct --revision 2fe192450127e6a83f7441aef6e3ca586c338b77 --prompt-ids $I --benchmark $B --max-cache-len 640 --use-m2 --ops lm_head,rmsnorm,sdpa,nf4 --output $out 2>&1 | tail -1 | cut -c1-400
echo PHI_DONE
