#!/usr/bin/env bash
# cnndm_ext150: OLMo-2-1B and Phi-3.5 (no GQA), naive vs plan list (plan lists from the GSM8K audits).
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=scripts:src
N=${N:-60}
run() { # name model rev auditjson
  local plan=$($PY scripts/shape_audit.py plan $OUT/$4)
  for r in none $plan; do
    local tag=$(echo $r | tr ',' '+'); local f=$OUT/ext150_cnndm_$1_k2_${tag}_n${N}_v1.json
    [ -e $f ] && continue
    $PY scripts/audit_pld_trajectory_divergence.py --benchmark data/benchmarks/cnndm_ext150.jsonl --model $2 --revision $3 --k 2 --n-prompts $N --rowwise $r --output $f 2>&1 | tail -1 | sed "s/^/[$r] /"
  done
}
run olmo2_1b allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e c4_olmo2_1b_m3_gsm8k20_v1.json
run phi35 microsoft/Phi-3.5-mini-instruct 2fe192450127e6a83f7441aef6e3ca586c338b77 c4_phi35_m3_gsm8k20_v1.json
echo MHA_DONE
