#!/usr/bin/env bash
# Data for the divergence prediction model (B) and the sampling analysis (G):
#  1. per-row noise audit (gap shift, TV, KL), M=3, 20 GSM8K prompts, six configs
#  2. naive PLD with per-decision margins, six configs x four prompt sets
set -uo pipefail
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
PYQ=${PYQ:?venv python with torchao}
OUT=experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles
export PYTHONPATH=src:scripts/lib
P20=$(for i in $(seq 0 19); do printf "gsm8k_%03d " $i; done)
Q3=aa8e72537993ba99e69dfaafa59ed015b17504d1
CFG=(
 "qwen05b hf_bnb_nf4 Qwen/Qwen2.5-0.5B-Instruct 7ae557604adf67be50417f59c2c2f167def9a775"
 "qwen15b hf_bnb_nf4 Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
 "qwen3b hf_bnb_nf4 Qwen/Qwen2.5-3B-Instruct $Q3"
 "olmo2_1b hf_bnb_nf4 allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e"
 "phi35 hf_bnb_nf4 microsoft/Phi-3.5-mini-instruct 2fe192450127e6a83f7441aef6e3ca586c338b77"
 "torchao_qwen3b torchao_int4_wo Qwen/Qwen2.5-3B-Instruct $Q3"
)
pick() { [ "$1" = torchao_int4_wo ] && echo $PYQ || echo $PY; }
for c in "${CFG[@]}"; do set -- $c
  f=$OUT/b_noise_$1_m3_gsm8k20_v1.json
  [ -e $f ] || $(pick $2) scripts/audit/audit_op_shape_sensitivity.py --backend $2 --model $3 --revision $4 --prompt-ids $P20 --block-size 3 --steps 48 --output $f 2>&1 | tail -1 | cut -c1-100
done
echo NOISE_DONE
DS=( "gsm8k data/benchmarks/gsm8k_eval.jsonl 0 50" "humaneval data/benchmarks/humaneval_eval.jsonl 0 50" "cnndm data/benchmarks/cnndm_eval.jsonl 0 50" "ext150 data/benchmarks/cnndm_ext150.jsonl 0 60" )
for d in "${DS[@]}"; do set -- $d; dn=$1; bench=$2; st=$3; n=$4
  for c in "${CFG[@]}"; do set -- $c
    f=$OUT/b_margins_$1_${dn}_naive_k2_v1.json
    [ -e $f ] || $(pick $2) scripts/lib/audit_pld_trajectory_divergence.py --backend $2 --model $3 --revision $4 --benchmark $bench --start $st --n-prompts $n --k 2 --rowwise none --output $f 2>&1 | tail -1 | sed "s/^/[$dn] /" | cut -c1-140
  done
done
echo B_DONE
