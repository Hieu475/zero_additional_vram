#!/usr/bin/env bash
# C3: per-operator shape sensitivity on other Qwen2.5 sizes (M=3 and M=4).
set -euo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
D=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
P20=$(for i in $(seq -w 0 19); do printf "gsm8k_0%s " "$i"; done)
export PYTHONPATH=src:scripts/lib
run() { # name model rev M
  $PY scripts/tools/run_with_model.py "$2" "$3" scripts/audit/audit_op_shape_sensitivity.py --prompt-ids $P20 --block-size $4 --steps 48 --output $D/c3_$1_m$4_gsm8k20_v1.json 2>&1 | grep -v "Loading weights"
}
run qwen05b Qwen/Qwen2.5-0.5B-Instruct "$REV05" 3
run qwen05b Qwen/Qwen2.5-0.5B-Instruct "$REV05" 4
run qwen15b Qwen/Qwen2.5-1.5B-Instruct "$REV15" 3
run qwen15b Qwen/Qwen2.5-1.5B-Instruct "$REV15" 4
echo C3_DONE
