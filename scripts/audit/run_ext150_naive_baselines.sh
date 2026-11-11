#!/usr/bin/env bash
# Naive-PLD baselines on cnndm_ext150 (first 60) for the configs that only had the planned run.
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
PYQ=${PYQ:?venv python with torchao}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=src:scripts/lib
run() { # py name backend model rev
  local f=$OUT/ext150_cnndm_$2_k2_none_n60_v1.json
  [ -e $f ] && return 0
  $1 scripts/lib/audit_pld_trajectory_divergence.py --benchmark data/benchmarks/cnndm_ext150.jsonl --backend $3 --model $4 --revision $5 --k 2 --n-prompts 60 --rowwise none --output $f 2>&1 | tail -1 | sed "s/^/[$2] /"
}
run $PY qwen15b hf_bnb_nf4 Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
run $PY qwen05b hf_bnb_nf4 Qwen/Qwen2.5-0.5B-Instruct 7ae557604adf67be50417f59c2c2f167def9a775
run $PYQ torchao_qwen3b torchao_int4_wo Qwen/Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1
echo BASE_DONE
