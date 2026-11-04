#!/usr/bin/env bash
# Independent validation: operator list from `shape_audit.py plan` (GSM8K audits) applied to HumanEval prompts.
# For each model: naive PLD vs planned row-wise PLD (K=2, 50 prompts, 128 tokens) against M=1 greedy.
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=scripts:src
run() { # name model rev auditjson
  local ops=$($PY scripts/shape_audit.py plan $OUT/$4)
  echo "== $1 plan=$ops"
  for r in none $ops; do
    local tag=$(echo $r | tr ',' '+')
    local f=$OUT/heldout_humaneval_$1_k2_${tag}_v1.json
    [ -e $f ] && continue
    $PY scripts/audit_pld_trajectory_divergence.py --benchmark data/benchmarks/humaneval_eval.jsonl --model $2 --revision $3 --k 2 --n-prompts 50 --rowwise $r --output $f 2>&1 | tail -1 | sed "s/^/[$r] /"
  done
}
run qwen3b Qwen/Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1 op_shape_sensitivity_gsm8k20_v1.json
run qwen15b Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 c3_qwen15b_m3_gsm8k20_v1.json
run qwen05b Qwen/Qwen2.5-0.5B-Instruct 7ae557604adf67be50417f59c2c2f167def9a775 c3_qwen05b_m3_gsm8k20_v1.json
run olmo2_1b allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e c4_olmo2_1b_m3_gsm8k20_v1.json
run phi35 microsoft/Phi-3.5-mini-instruct 2fe192450127e6a83f7441aef6e3ca586c338b77 c4_phi35_m3_gsm8k20_v1.json
echo HELDOUT_DONE
