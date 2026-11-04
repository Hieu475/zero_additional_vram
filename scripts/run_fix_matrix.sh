#!/usr/bin/env bash
# Audit-driven fix validation: naive PLD with selected operator classes run row-by-row; trajectory divergence vs M=1 greedy.
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=scripts:src
Q3=aa8e72537993ba99e69dfaafa59ed015b17504d1
run() { # name backend model rev k ops
  local tag=$(echo $6 | tr ',' '+')
  local f=$OUT/fix_$1_k$5_${tag}_gsm8k50_v1.json
  [ -e $f ] && return 0
  $PY scripts/audit_pld_trajectory_divergence.py --backend $2 --model $3 --revision $4 --k $5 --n-prompts 50 --rowwise $6 --output $f 2>&1 | tail -1 | sed "s/^/[$6] /"
}
MIN=rmsnorm,sdpa,lm_head
run qwen3b hf_bnb_nf4 Qwen/Qwen2.5-3B-Instruct $Q3 2 $MIN
run qwen3b hf_bnb_nf4 Qwen/Qwen2.5-3B-Instruct $Q3 2 sdpa,lm_head
run qwen3b hf_bnb_nf4 Qwen/Qwen2.5-3B-Instruct $Q3 2 sdpa
run qwen3b hf_bnb_nf4 Qwen/Qwen2.5-3B-Instruct $Q3 3 $MIN
run qwen3b hf_bnb_nf4 Qwen/Qwen2.5-3B-Instruct $Q3 3 all
run qwen3b_hqq hqq_gemlite_w4a16 Qwen/Qwen2.5-3B-Instruct $Q3 2 $MIN
run qwen3b_hqq hqq_gemlite_w4a16 Qwen/Qwen2.5-3B-Instruct $Q3 2 all
run qwen15b hf_bnb_nf4 Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 2 $MIN
run qwen05b hf_bnb_nf4 Qwen/Qwen2.5-0.5B-Instruct 7ae557604adf67be50417f59c2c2f167def9a775 2 $MIN
run olmo2_1b hf_bnb_nf4 allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e 2 $MIN
run phi35 hf_bnb_nf4 microsoft/Phi-3.5-mini-instruct 2fe192450127e6a83f7441aef6e3ca586c338b77 2 $MIN
echo FIX_DONE
