#!/usr/bin/env bash
# C3/C4: other model families (OLMo-2, Phi-3.5) on bnb NF4 and HQQ+GemLite on Qwen2.5-3B.
# usage: run_c4_families.sh <prompts> <steps> <outdir-suffix>   (smoke: 1 prompt, 12 steps)
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
OUT=${OUT:-experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles}
N=${1:-20}; STEPS=${2:-48}; TAG=${3:-v1}; MS=${MS:-"3 4"}
PROMPTS=$(for i in $(seq 0 $((N-1))); do printf "gsm8k_%03d " $i; done)
export PYTHONPATH=src:scripts/lib
run() { # name backend model rev M
  $PY scripts/audit/audit_op_shape_sensitivity.py --backend $2 --model $3 --revision $4 --prompt-ids $PROMPTS --block-size $5 --steps $STEPS \
     --output $OUT/c4_$1_m$5_gsm8k${N}_$TAG.json 2>&1 | grep -v "Loading weights" | tail -15 | cut -c1-300
}
for M in $MS; do
  run olmo2_1b hf_bnb_nf4 allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e $M
  run phi35 hf_bnb_nf4 microsoft/Phi-3.5-mini-instruct 2fe192450127e6a83f7441aef6e3ca586c338b77 $M
  run qwen3b_hqq hqq_gemlite_w4a16 Qwen/Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1 $M
done
echo C4_DONE
