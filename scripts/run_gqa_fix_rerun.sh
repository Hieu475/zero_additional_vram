#!/usr/bin/env bash
# Re-run after the GQA row-wise SDPA fix (compact K/V + enable_gqa, as in the real M=1 step). Only GQA models are affected.
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
PYQ=${PYQ:?venv python with torchao}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=scripts:src
P20=$(for i in $(seq 0 19); do printf "gsm8k_%03d " $i; done)
declare -A REV=([qwen3b]=aa8e72537993ba99e69dfaafa59ed015b17504d1 [qwen15b]=989aa7980e4cf806f80c7fef2b1adb7bc71aa306 [qwen05b]=7ae557604adf67be50417f59c2c2f167def9a775)
declare -A MOD=([qwen3b]=Qwen/Qwen2.5-3B-Instruct [qwen15b]=Qwen/Qwen2.5-1.5B-Instruct [qwen05b]=Qwen/Qwen2.5-0.5B-Instruct)
declare -A PLAN=([qwen3b]=lm_head,rmsnorm,sdpa [qwen15b]=lm_head,sdpa [qwen05b]=sdpa)
for n in qwen3b qwen15b qwen05b; do   # 1. corrected op audit (M=3) -> v2
  f=$OUT/c9_${n}_m3_gsm8k20_gqafix_v2.json
  [ -e $f ] || $PY scripts/audit_op_shape_sensitivity.py --model ${MOD[$n]} --revision ${REV[$n]} --prompt-ids $P20 --block-size 3 --steps 48 --output $f 2>&1 | tail -3 | cut -c1-200
done
for n in qwen3b qwen15b qwen05b; do   # 2. CNN/DM with the (unchanged) plan lists, fixed row-wise SDPA
  tag=$(echo ${PLAN[$n]} | tr ',' '+'); f=$OUT/heldout_cnndm_${n}_k2_${tag}_gqafix_v2.json
  [ -e $f ] || $PY scripts/audit_pld_trajectory_divergence.py --benchmark data/benchmarks/cnndm_eval.jsonl --model ${MOD[$n]} --revision ${REV[$n]} --k 2 --n-prompts 50 --rowwise ${PLAN[$n]} --output $f 2>&1 | tail -1 | sed "s/^/[$n ${PLAN[$n]}] /"
done
for r in none lm_head,rmsnorm,sdpa; do   # 3. torchao on CNN/DM
  tag=$(echo $r | tr ',' '+'); f=$OUT/heldout_cnndm_torchao_qwen3b_k2_${tag}_gqafix_v2.json
  [ -e $f ] || $PYQ scripts/audit_pld_trajectory_divergence.py --backend torchao_int4_wo --benchmark data/benchmarks/cnndm_eval.jsonl --model Qwen/Qwen2.5-3B-Instruct --revision ${REV[qwen3b]} --k 2 --n-prompts 50 --rowwise $r --output $f 2>&1 | tail -1 | sed "s/^/[torchao $r] /"
done
echo GQAFIX_DONE
