#!/usr/bin/env bash
# WP8: re-run the exact PLD loop on every cell recording the first EOS position, so speedups can be reported on prompts whose 128 generated
# tokens contain no EOS (post-EOS text is degenerate/repetitive and flatters PLD).
cd "$(dirname "$0")/../.."
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
export PYTHONPATH=src:scripts/lib
Q3=Qwen/Qwen2.5-3B-Instruct; Q3R=aa8e72537993ba99e69dfaafa59ed015b17504d1
cells=(
"qwen3b_gsm8k gsm8k_wp5_100 $Q3 $Q3R 640"
"qwen3b_humaneval humaneval_wp5_100 $Q3 $Q3R 1024"
"qwen3b_cnndm cnndm_wp5_100 $Q3 $Q3R 1024"
"llama3b_gsm8k gsm8k_wp5_100 unsloth/Llama-3.2-3B-Instruct 006f5dcd1393c3add266de40994ba96225e9689d 640"
"llama1b_gsm8k gsm8k_wp5_100 unsloth/Llama-3.2-1B-Instruct 5a8abab4a5d6f164389b1079fb721cfab8d7126c 640"
"phi35_gsm8k gsm8k_wp5_100 microsoft/Phi-3.5-mini-instruct 2fe192450127e6a83f7441aef6e3ca586c338b77 640"
)
for cfg in "${cells[@]}"; do
  set -- $cfg; name=$1; ds=$2; model=$3; rev=$4; cl=$5
  B=data/benchmarks/$ds.jsonl
  out=results/kernels/wp8_eos_$name.json
  [ -e $out ] && continue
  I=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('$B') if l.strip()))")
  $PY scripts/kernels/wp5_invariant_graph_pld.py --model $model --revision $rev --prompt-ids $I --benchmark $B --max-cache-len $cl --use-m2 --ops lm_head,rmsnorm,sdpa,nf4 --output $out 2>&1 | tail -1 | cut -c1-500
done
echo WP8_DONE
