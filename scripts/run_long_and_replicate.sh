#!/usr/bin/env bash
# (1) PLD on 511-3072-token prompts (long_context_dev.jsonl); (2) independent replicate on cnndm_ext150 prompts 60..119.
# Plan lists are the ones produced from the GSM8K audits (unchanged).
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
PYQ=${PYQ:?venv python with torchao}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=scripts:src
Q3=aa8e72537993ba99e69dfaafa59ed015b17504d1
run() { # py tag bench start n backend model rev ops
  local tag=$(echo $9 | tr ',' '+'); local f=$OUT/$2_k2_${tag}_v1.json
  [ -e $f ] && return 0
  $1 scripts/audit_pld_trajectory_divergence.py --benchmark $3 --start $4 --n-prompts $5 --backend $6 --model $7 --revision $8 --k 2 --rowwise $9 --output $f 2>&1 | tail -1 | sed "s/^/[$2 $9] /"
}
LC=data/benchmarks/long_context_dev.jsonl
for r in none lm_head,rmsnorm,sdpa; do run $PY longctx_qwen3b $LC 0 12 hf_bnb_nf4 Qwen/Qwen2.5-3B-Instruct $Q3 $r; done
for r in none lm_head,rmsnorm,sdpa; do run $PY longctx_olmo2_1b $LC 0 12 hf_bnb_nf4 allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e $r; done
E=data/benchmarks/cnndm_ext150.jsonl
for r in none lm_head,rmsnorm,sdpa; do run $PY ext150b_cnndm_qwen3b $E 60 60 hf_bnb_nf4 Qwen/Qwen2.5-3B-Instruct $Q3 $r; done
run $PYQ ext150b_cnndm_torchao_qwen3b $E 60 60 torchao_int4_wo Qwen/Qwen2.5-3B-Instruct $Q3 lm_head,rmsnorm,sdpa
echo LONG_DONE
