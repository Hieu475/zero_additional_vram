## Per-operator bitwise mismatch vs row-by-row (block of M rows)

| Config | M | ctx | proj elems mismatched | RMSNorm elems | LM head | SDPA | argmax flips / rows | file |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 | 3 | nat. | 0.000% | 111 | 0.048% | 17.4% | 2/960 | c4_olmo2_1b_m3_gsm8k20_v1.json |
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 | 4 | nat. | 15.422% | 189 | 0.048% | 17.4% | 1/960 | c4_olmo2_1b_m4_gsm8k20_v1.json |
| Phi-3.5-mini-instruct / hf_bnb_nf4 | 3 | nat. | 0.000% | 358 | 0.083% | 19.2% | 3/960 | c4_phi35_m3_gsm8k20_v1.json |
| Phi-3.5-mini-instruct / hf_bnb_nf4 | 4 | nat. | 14.789% | 459 | 0.084% | 19.3% | 4/960 | c4_phi35_m4_gsm8k20_v1.json |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 | 3 | nat. | 0.000% | 0 | 0.000% | 3.6% | 6/960 | c3_qwen05b_m3_gsm8k20_v1.json |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 | 3 | nat. | 0.000% | 0 | 0.000% | 3.6% | 6/960 | c9_qwen05b_m3_gsm8k20_gqafix_v2.json |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 | 4 | nat. | 14.273% | 0 | 0.000% | 3.6% | 9/960 | c3_qwen05b_m4_gsm8k20_v1.json |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 | 3 | nat. | 0.000% | 0 | 0.109% | 18.3% | 7/960 | c3_qwen15b_m3_gsm8k20_v1.json |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 | 3 | nat. | 0.000% | 0 | 0.109% | 18.3% | 7/960 | c9_qwen15b_m3_gsm8k20_gqafix_v2.json |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 | 4 | nat. | 13.853% | 92 | 0.110% | 18.3% | 9/960 | c3_qwen15b_m4_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 2 | nat. | 0.000% | 60 | 0.193% | 18.6% | 4/960 | c2_m2_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 2 | nat. | 0.000% | 60 | 0.193% | 18.6% | 4/960 | c2_m2_gsm8k20_v2_recheck.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 3 | nat. | 0.000% | 87 | 0.193% | 18.6% | 4/960 | c9_qwen3b_m3_gsm8k20_gqafix_v2.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 3 | nat. | 0.000% | 87 | 0.193% | 18.6% | 4/960 | op_shape_sensitivity_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 4 | nat. | 12.963% | 244 | 0.193% | 18.6% | 4/960 | c2_m4_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 5 | nat. | 12.960% | 191 | 0.194% | 18.7% | 4/900 | c2_m5_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 8 | nat. | 12.962% | 199 | 0.193% | 18.6% | 5/960 | c2_m8_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 16 | nat. | 12.963% | 399 | 0.193% | 18.6% | 3/960 | c2_m16_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 3 | 512 | 0.000% | 20 | 0.173% | 19.4% | 4/480 | c2_m3_ctx512_gsm8k10_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 3 | 512 | 0.000% | 20 | 0.173% | 19.4% | 4/480 | c9_m3_ctx512_gsm8k10_gqafix_v2.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 3 | 1024 | 0.000% | 206 | 0.168% | 19.1% | 3/480 | c2_m3_ctx1024_gsm8k10_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 3 | 2048 | 0.000% | 61 | 0.180% | 18.6% | 3/480 | c2_m3_ctx2048_gsm8k10_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 3 | 2048 | 0.000% | 61 | 0.180% | 18.4% | 3/480 | c9_m3_ctx2048_gsm8k10_gqafix_v2.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 3 | 3072 | 0.000% | 84 | 0.175% | 19.3% | 1/480 | c2_m3_ctx3072_gsm8k10_v1.json |
| Qwen2.5-3B-Instruct / hqq_gemlite_w4a16 [NONDETERMINISTIC run-to-run: not a valid shape-sensitivity measurement] | 3 | nat. | 49.637% | 1704 | 1.571% | 18.2% | 0/960 | c4_qwen3b_hqq_m3_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hqq_gemlite_w4a16 [NONDETERMINISTIC run-to-run: not a valid shape-sensitivity measurement] | 4 | nat. | 49.059% | 1827 | 1.570% | 18.2% | 0/960 | c4_qwen3b_hqq_m4_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / torchao_int4_wo | 3 | nat. | 0.000% | 65 | 0.164% | 17.4% | 9/960 | c8_torchao_qwen3b_m3_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / torchao_int4_wo | 4 | nat. | 0.000% | 83 | 0.164% | 17.4% | 10/960 | c8_torchao_qwen3b_m4_gsm8k20_v1.json |

## Naive PLD trajectory divergence vs sequential M=1 greedy

| Config | K | prompts | max new | diverged | median first divergence | file |
|---|---:|---:|---:|---:|---:|---|
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 | 2 | 20 | 128 | 6/20 | 60 | c6_olmo2_1b_k2_gsm8k20_v1.json |
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 | 3 | 20 | 128 | 6/20 | 52 | c6_olmo2_1b_k3_gsm8k20_v1.json |
| Phi-3.5-mini-instruct / hf_bnb_nf4 | 2 | 20 | 128 | 11/20 | 64 | c6_phi35_k2_gsm8k20_v1.json |
| Phi-3.5-mini-instruct / hf_bnb_nf4 | 3 | 20 | 128 | 10/20 | 53 | c6_phi35_k3_gsm8k20_v1.json |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 | 2 | 20 | 128 | 14/20 | 43 | c6_qwen05b_k2_gsm8k20_v1.json |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 | 3 | 20 | 128 | 14/20 | 44 | c6_qwen05b_k3_gsm8k20_v1.json |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 | 2 | 20 | 128 | 9/20 | 57 | c6_qwen15b_k2_gsm8k20_v1.json |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 | 3 | 20 | 128 | 9/20 | 57 | c6_qwen15b_k3_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 2 | 20 | 128 | 8/20 | 94 | c6_qwen3b_k2_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 | 3 | 20 | 128 | 6/20 | 78 | c6_qwen3b_k3_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hqq_gemlite_w4a16 [NONDETERMINISTIC run-to-run: not a valid shape-sensitivity measurement] | 2 | 20 | 128 | 2/20 | 36 | c6_qwen3b_hqq_k2_gsm8k20_v1.json |
| Qwen2.5-3B-Instruct / hqq_gemlite_w4a16 [NONDETERMINISTIC run-to-run: not a valid shape-sensitivity measurement] | 3 | 20 | 128 | 2/20 | 105 | c6_qwen3b_hqq_k3_gsm8k20_v1.json |

## PLD trajectory divergence vs sequential M=1 greedy, prompt ranges merged (Wilson 95% CI)

Row-wise = operator classes run one row at a time during block verification ('none' = naive PLD).

| Config | K | row-wise ops | prompts | diverged | rate | 95% CI |
|---|---:|---|---:|---:|---:|---|
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 [CNN/DM] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 [CNN/DM] | 2 | none | 50 | 27 | 54% | [40%, 67%] |
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | none | 50 | 17 | 34% | [22%, 48%] |
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 [GSM8K] | 3 | none | 20 | 6 | 30% | [15%, 52%] |
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 [HumanEval] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| OLMo-2-0425-1B-Instruct / hf_bnb_nf4 [HumanEval] | 2 | none | 50 | 17 | 34% | [22%, 48%] |
| Phi-3.5-mini-instruct / hf_bnb_nf4 [CNN/DM] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Phi-3.5-mini-instruct / hf_bnb_nf4 [CNN/DM] | 2 | none | 50 | 41 | 82% | [69%, 90%] |
| Phi-3.5-mini-instruct / hf_bnb_nf4 [GSM8K] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Phi-3.5-mini-instruct / hf_bnb_nf4 [GSM8K] | 2 | none | 50 | 23 | 46% | [33%, 60%] |
| Phi-3.5-mini-instruct / hf_bnb_nf4 [GSM8K] | 3 | none | 20 | 10 | 50% | [30%, 70%] |
| Phi-3.5-mini-instruct / hf_bnb_nf4 [HumanEval] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Phi-3.5-mini-instruct / hf_bnb_nf4 [HumanEval] | 2 | none | 50 | 31 | 62% | [48%, 74%] |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 [CNN/DM, PRE-GQA-FIX row-wise SDPA (invalid for GQA models)] | 2 | sdpa | 50 | 10 | 20% | [11%, 33%] |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 [CNN/DM] | 2 | none | 50 | 38 | 76% | [63%, 86%] |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 [CNN/DM] | 2 | sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | none | 50 | 33 | 66% | [52%, 78%] |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 [GSM8K] | 3 | none | 20 | 14 | 70% | [48%, 85%] |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 [HumanEval] | 2 | none | 50 | 26 | 52% | [39%, 65%] |
| Qwen2.5-0.5B-Instruct / hf_bnb_nf4 [HumanEval] | 2 | sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 [CNN/DM, PRE-GQA-FIX row-wise SDPA (invalid for GQA models)] | 2 | lm_head+sdpa | 50 | 33 | 66% | [52%, 78%] |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 [CNN/DM] | 2 | lm_head+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 [CNN/DM] | 2 | none | 50 | 36 | 72% | [58%, 83%] |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | none | 50 | 25 | 50% | [37%, 63%] |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 [GSM8K] | 3 | none | 20 | 9 | 45% | [26%, 66%] |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 [HumanEval] | 2 | lm_head+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-1.5B-Instruct / hf_bnb_nf4 [HumanEval] | 2 | none | 50 | 13 | 26% | [16%, 40%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [CNN/DM, PRE-GQA-FIX row-wise SDPA (invalid for GQA models)] | 2 | lm_head+rmsnorm+sdpa | 50 | 36 | 72% | [58%, 83%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [CNN/DM] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [CNN/DM] | 2 | none | 50 | 33 | 66% | [52%, 78%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | lm_head+sdpa | 50 | 3 | 6% | [2%, 16%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | none | 50 | 18 | 36% | [24%, 50%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [GSM8K] | 2 | sdpa | 50 | 3 | 6% | [2%, 16%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [GSM8K] | 3 | linear+lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [GSM8K] | 3 | lm_head+rmsnorm+sdpa | 50 | 18 | 36% | [24%, 50%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [GSM8K] | 3 | none | 20 | 6 | 30% | [15%, 52%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [HumanEval] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-3B-Instruct / hf_bnb_nf4 [HumanEval] | 2 | none | 50 | 19 | 38% | [26%, 52%] |
| Qwen2.5-3B-Instruct / hqq_gemlite_w4a16 [NONDETERMINISTIC run-to-run: not a valid shape-sensitivity measurement] [GSM8K] | 2 | linear+lm_head+rmsnorm+sdpa | 50 | 3 | 6% | [2%, 16%] |
| Qwen2.5-3B-Instruct / hqq_gemlite_w4a16 [NONDETERMINISTIC run-to-run: not a valid shape-sensitivity measurement] [GSM8K] | 2 | lm_head+rmsnorm+sdpa | 50 | 7 | 14% | [7%, 26%] |
| Qwen2.5-3B-Instruct / hqq_gemlite_w4a16 [NONDETERMINISTIC run-to-run: not a valid shape-sensitivity measurement] [GSM8K] | 2 | none | 50 | 3 | 6% | [2%, 16%] |
| Qwen2.5-3B-Instruct / hqq_gemlite_w4a16 [NONDETERMINISTIC run-to-run: not a valid shape-sensitivity measurement] [GSM8K] | 3 | none | 20 | 2 | 10% | [3%, 30%] |
| Qwen2.5-3B-Instruct / torchao_int4_wo [CNN/DM] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-3B-Instruct / torchao_int4_wo [CNN/DM] | 2 | none | 50 | 38 | 76% | [63%, 86%] |
| Qwen2.5-3B-Instruct / torchao_int4_wo [GSM8K] | 2 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-3B-Instruct / torchao_int4_wo [GSM8K] | 2 | none | 50 | 19 | 38% | [26%, 52%] |
| Qwen2.5-3B-Instruct / torchao_int4_wo [GSM8K] | 3 | lm_head+rmsnorm+sdpa | 50 | 0 | 0% | [0%, 7%] |
| Qwen2.5-3B-Instruct / torchao_int4_wo [GSM8K] | 3 | none | 50 | 20 | 40% | [28%, 54%] |
