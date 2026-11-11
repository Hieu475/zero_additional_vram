#!/usr/bin/env bash
cd "$(dirname "$0")/../.."; export PYTHONPATH=src
D=results/drafter/q15
python3 scripts/drafter/drafter_train.py --data $D/gsm8k_a.npz $D/gsm8k_b.npz $D/cnndm.npz $D/code.npz --k 5 --epochs 8 --model-id Qwen/Qwen2.5-1.5B-Instruct --merged $D/_merged_h.npy --output results/drafter/q15_k5.pt > logs/q15_train.log 2>&1
