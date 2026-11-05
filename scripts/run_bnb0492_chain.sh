#!/usr/bin/env bash
# Compare bitsandbytes 0.49.2 (before the new 4-bit GEMM kernels, PR #1949) with 0.50.0 on Qwen2.5-3B NF4.
set -uo pipefail
PYB=${PYB:?venv python with bitsandbytes==0.49.2}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=scripts:src
M=Qwen/Qwen2.5-3B-Instruct; R=aa8e72537993ba99e69dfaafa59ed015b17504d1
P20=$(for i in $(seq 0 19); do printf "gsm8k_%03d " $i; done)
for BS in 3 4 8; do
  f=$OUT/c10_bnb0492_qwen3b_m${BS}_gsm8k20_v1.json
  [ -e $f ] || $PYB scripts/audit_op_shape_sensitivity.py --model $M --revision $R --prompt-ids $P20 --block-size $BS --steps 48 --output $f 2>&1 | grep -v "Loading weights" | tail -14 | cut -c1-200
done
PLAN3=$($PYB scripts/shape_audit.py plan $OUT/c10_bnb0492_qwen3b_m3_gsm8k20_v1.json)
PLAN4=$($PYB scripts/shape_audit.py plan $OUT/c10_bnb0492_qwen3b_m4_gsm8k20_v1.json)
echo "PLAN3=$PLAN3 PLAN4=$PLAN4"
run() { # k ops
  local tag=$(echo $2 | tr ',' '+'); local f=$OUT/c10_bnb0492_qwen3b_k$1_${tag}_gsm8k50_v1.json
  [ -e $f ] || $PYB scripts/audit_pld_trajectory_divergence.py --model $M --revision $R --k $1 --n-prompts 50 --rowwise $2 --output $f 2>&1 | tail -1 | sed "s/^/[k=$1 $2] /"
}
run 2 none; run 2 $PLAN3; run 3 none; run 3 $PLAN4
echo B49_DONE
