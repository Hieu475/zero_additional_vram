#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
export PYTHONPATH=src:scripts
[ -e results/kernels/wp6_agreement_gsm8k_eval50.json ] || $PY scripts/wp6_reference_agreement.py --benchmark data/benchmarks/gsm8k_eval.jsonl --n-prompts 50 --output results/kernels/wp6_agreement_gsm8k_eval50.json 2>&1 | tail -1
[ -e results/kernels/wp6_agreement_gsm8k_ext150.json ] || $PY scripts/wp6_reference_agreement.py --benchmark data/benchmarks/gsm8k_ext150.jsonl --n-prompts 150 --output results/kernels/wp6_agreement_gsm8k_ext150.json 2>&1 | tail -1
echo WP6_DONE
# WP6b: HumanEval pass@1, stock vs invariant
[ -e results/kernels/wp6_humaneval_pass1.json ] || $PY scripts/wp6_humaneval_passk.py --output results/kernels/wp6_humaneval_pass1.json 2>&1 | tail -1 | cut -c1-500
