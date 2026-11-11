#!/usr/bin/env bash
# Self-distillation data for Qwen2.5-1.5B (same recipe as the 3B drafter). Resumable.
cd "$(dirname "$0")/../.."; export PYTHONPATH=src:scripts/lib
R=989aa7980e4cf806f80c7fef2b1adb7bc71aa306; M=Qwen/Qwen2.5-1.5B-Instruct; O=results/drafter/q15
g() { [ -e $O/$1.npz ] || python3 scripts/drafter/drafter_gen_data.py --model $M --revision $R --source $2 --n $3 --start $4 --output $O/$1.npz > logs/gen_q15_$1.log 2>&1; }
g gsm8k_a gsm8k 3000 0; g code code 3000 0; g cnndm cnndm 2000 0; g gsm8k_b gsm8k 3000 3000
