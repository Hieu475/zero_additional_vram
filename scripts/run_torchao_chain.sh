#!/usr/bin/env bash
# Second 4-bit backend: torchao Int4WeightOnly (tile-packed tinygemm), Qwen2.5-3B. Needs a venv with torchao (see notes).
set -uo pipefail
PY=${PY:?set PY to a python with torchao installed}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=scripts:src
M=Qwen/Qwen2.5-3B-Instruct; R=aa8e72537993ba99e69dfaafa59ed015b17504d1; B=torchao_int4_wo
P20=$(for i in $(seq 0 19); do printf "gsm8k_%03d " $i; done)
for BS in 3 4; do
  f=$OUT/c8_torchao_qwen3b_m${BS}_gsm8k20_v1.json
  [ -e $f ] || $PY scripts/audit_op_shape_sensitivity.py --backend $B --model $M --revision $R --prompt-ids $P20 --block-size $BS --steps 48 --output $f 2>&1 | grep -v "Loading weights" | tail -14 | cut -c1-200
done
PLAN=$($PY scripts/shape_audit.py plan $OUT/c8_torchao_qwen3b_m3_gsm8k20_v1.json)
echo "PLAN(M=3)=$PLAN"
for r in none $PLAN; do
  tag=$(echo $r | tr ',' '+')
  f=$OUT/c8_torchao_qwen3b_k2_${tag}_gsm8k50_v1.json
  [ -e $f ] || $PY scripts/audit_pld_trajectory_divergence.py --backend $B --model $M --revision $R --k 2 --n-prompts 50 --rowwise $r --output $f 2>&1 | tail -1 | sed "s/^/[$r] /"
done
echo TORCHAO_DONE
