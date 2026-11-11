#!/usr/bin/env bash
# WP5 at scale: exact PLD graph loop with M-invariant kernels, >=100 prompts per set, several models. Resumable.
cd "$(dirname "$0")/../.."
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
export PYTHONPATH=src:scripts/lib
ids() { $PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('$1') if l.strip()))"; }
run() {  # tag model revision benchmark cachelen
  out=results/kernels/wp5_scale_$1.json
  [ -e $out ] && return
  $PY scripts/kernels/wp5_invariant_graph_pld.py --model $2 --revision $3 --prompt-ids $(ids $4) --benchmark $4 --max-cache-len $5 --use-m2 --ops lm_head,rmsnorm,sdpa,nf4 --output $out 2>&1 | tail -1 | cut -c1-700
}
Q3=aa8e72537993ba99e69dfaafa59ed015b17504d1; Q15=989aa7980e4cf806f80c7fef2b1adb7bc71aa306; Q05=7ae557604adf67be50417f59c2c2f167def9a775; OL=48d788eca847d4d7548f375ad03d3c9312f6139e
run qwen3b_gsm8k200 Qwen/Qwen2.5-3B-Instruct $Q3 data/benchmarks/gsm8k_wp5_200.jsonl 640
run qwen3b_humaneval100 Qwen/Qwen2.5-3B-Instruct $Q3 data/benchmarks/humaneval_wp5_100.jsonl 1024
run qwen3b_cnndm100 Qwen/Qwen2.5-3B-Instruct $Q3 data/benchmarks/cnndm_wp5_100.jsonl 1024
run qwen15b_gsm8k100 Qwen/Qwen2.5-1.5B-Instruct $Q15 data/benchmarks/gsm8k_wp5_100.jsonl 640
run qwen05b_gsm8k100 Qwen/Qwen2.5-0.5B-Instruct $Q05 data/benchmarks/gsm8k_wp5_100.jsonl 640
run olmo2_1b_gsm8k100 allenai/OLMo-2-0425-1B-Instruct $OL data/benchmarks/gsm8k_wp5_100.jsonl 640
echo WP5_SCALE_DONE
