#!/usr/bin/env bash
# D: Qwen2.5-3B (bnb NF4) target with a Qwen2.5-0.5B fp16 draft; K in {2,4,6}; naive vs row-wise lists; GSM8K and CNN/DM (50 prompts).
# K=2 uses the audit list for M=3 (lm_head,rmsnorm,sdpa); K>=4 (M>=5) uses all classes because bnb 0.50.0 projections differ from M=4.
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=src:scripts/lib
run() { # dataset bench k ops
  local tag=$(echo $4 | tr ',' '+'); local f=$OUT/d_draft_qwen3b_$1_k$3_${tag}_v1.json
  [ -e $f ] && return 0
  $PY scripts/audit/audit_draftmodel_sd_divergence.py --benchmark $2 --n-prompts 50 --k $3 --rowwise $4 --output $f 2>&1 | tail -1 | cut -c1-200
}
for ds in "gsm8k data/benchmarks/gsm8k_eval.jsonl" "cnndm data/benchmarks/cnndm_eval.jsonl"; do set -- $ds
  run $1 $2 2 none; run $1 $2 2 lm_head,rmsnorm,sdpa
  run $1 $2 4 none; run $1 $2 4 all
  run $1 $2 6 none; run $1 $2 6 all
done
echo D_DONE
