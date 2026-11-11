#!/usr/bin/env bash
# C2 sweep: per-operator shape sensitivity vs block size M and context length (bitsandbytes NF4, Qwen2.5-3B).
set -euo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
D=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
P20=$(for i in $(seq -w 0 19); do printf "gsm8k_0%s " "$i"; done)
P10=$(for i in $(seq -w 0 9); do printf "gsm8k_00%s " "$i"; done)
export PYTHONPATH=src:scripts/lib
for M in 2 4 5 8 16; do
  $PY scripts/audit/audit_op_shape_sensitivity.py --prompt-ids $P20 --block-size $M --steps 48 --output $D/c2_m${M}_gsm8k20_v1.json 2>&1 | grep -v "Loading weights"
done
for C in 512 1024 2048 3072; do
  $PY scripts/audit/audit_op_shape_sensitivity.py --prompt-ids $P10 --block-size 3 --steps 48 --context-len $C --output $D/c2_m3_ctx${C}_gsm8k10_v1.json 2>&1 | grep -v "Loading weights"
done
echo SWEEP_DONE
