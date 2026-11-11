#!/usr/bin/env bash
# C6 extension: prompts 20..99 (80 new) at K=2 for every config; merged with the first 20 by shape_audit summarize.
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=src:scripts/lib
run() { # name backend model rev
  [ -e $OUT/c6x_$1_k2_gsm8k20to99_v1.json ] && return 0
  $PY scripts/lib/audit_pld_trajectory_divergence.py --backend $2 --model $3 --revision $4 --k 2 --start 20 --n-prompts 80 --output $OUT/c6x_$1_k2_gsm8k20to99_v1.json 2>&1 | tail -1
}
run qwen3b hf_bnb_nf4 Qwen/Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1
run qwen15b hf_bnb_nf4 Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
run qwen05b hf_bnb_nf4 Qwen/Qwen2.5-0.5B-Instruct 7ae557604adf67be50417f59c2c2f167def9a775
run olmo2_1b hf_bnb_nf4 allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e
run phi35 hf_bnb_nf4 microsoft/Phi-3.5-mini-instruct 2fe192450127e6a83f7441aef6e3ca586c338b77
run qwen3b_hqq hqq_gemlite_w4a16 Qwen/Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1
echo C6X_DONE
