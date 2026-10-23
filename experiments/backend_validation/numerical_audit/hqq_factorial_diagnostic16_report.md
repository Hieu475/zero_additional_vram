# KV × verification factorial intervention

## Protocol

- Execution commit: `3578a0c21f1a6b1e9cd24682bfa51cb30f05d4b0`; model `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Backend `hqq_gemlite_w4a16` with metadata `{"attention": "sdpa", "hqq_version": "0.2.8.post1", "quantization": "HQQ W4A16, group_size=64, axis=1, GemLite"}`; SDPA; static KV.
- Diagnostic cohort: 16 prompts; each PLD output mismatch vs the matched-capacity canonical sequential reference: 0/16.
- For each selected prompt prefix and fixed candidate sequence, four isolated cache clones were evaluated: A active KV + batch, B canonical KV + batch, C active KV + sequential, D canonical KV + sequential.
- Every condition began at the same logical prefix length; input token positions are `[prefix_length, prefix_length + candidate_length)`. Attention mask was omitted consistently so the model/cache constructs the causal mask from logical cache length.
- Cache snapshots were fingerprinted before intervention; source snapshots remained unchanged. All cache length and clone integrity checks passed: `True`.
- The selected row is the first active target argmax that differs from the same-prefix canonical vanilla continuation. Logit row `j` follows candidate input token `j` and predicts the next token. Each row logs argmax, top-1 margin, MAE, max absolute/relative error, L2 norm, and the factorial interaction norm. KV errors are recorded layer-wise.

## Attribution

Actual selected-cycle classification counts: `{"no_argmax_divergence_in_selected_row": 16}`. Actual cycle K counts: `{"1": 1, "4": 15}`. Verification modes: `{"batched": 16}`. Actual cycle input source: `{"actual_pld_candidate_sequence": 16}`. Counterfactual vanilla continuation extension was used in 1 prompts.

The main attribution uses the actual K at the first mismatch. For cycles that fell back with K=0, K=1/2/4 probes extend the fixed candidate sequence using the vanilla continuation; those are counterfactual execution-shape controls, not candidates PLD actually drafted. They are labeled in the raw JSONL. Three prompts had real speculative candidates at the selected cycle.

| Prompt | Actual K | Mode | Cycle | Attribution | A argmax | D argmax | Vanilla expected | Candidate source |
|---|---:|---|---:|---|---:|---:|---:|---|
| qasper_119_ctx384 | 4 | batched | 7 | no_argmax_divergence_in_selected_row | 2197 | 2197 | 2197 | actual_pld_candidate_sequence |
| multi_news_189_ctx384 | 4 | batched | 10 | no_argmax_divergence_in_selected_row | 879 | 879 | 879 | actual_pld_candidate_sequence |
| qasper_109_ctx384 | 4 | batched | 8 | no_argmax_divergence_in_selected_row | 12818 | 12818 | 12818 | actual_pld_candidate_sequence |
| gov_report_98_ctx384 | 4 | batched | 0 | no_argmax_divergence_in_selected_row | 2197 | 2197 | 2197 | actual_pld_candidate_sequence |
| gov_report_134_ctx384 | 4 | batched | 0 | no_argmax_divergence_in_selected_row | 2197 | 2197 | 2197 | actual_pld_candidate_sequence |
| gov_report_80_ctx384 | 4 | batched | 0 | no_argmax_divergence_in_selected_row | 11426 | 11426 | 11426 | actual_pld_candidate_sequence |
| qmsum_2_ctx384 | 1 | batched | 17 | no_argmax_divergence_in_selected_row | 572 | 572 | 572 | actual_pld_candidate_sequence |
| narrativeqa_6_ctx384 | 4 | batched | 12 | no_argmax_divergence_in_selected_row | 2753 | 2753 | 2753 | actual_pld_candidate_sequence |
| multi_news_143_ctx384 | 4 | batched | 0 | no_argmax_divergence_in_selected_row | 2197 | 2197 | 2197 | actual_pld_candidate_sequence |
| qmsum_4_ctx384 | 4 | batched | 15 | no_argmax_divergence_in_selected_row | 3383 | 3383 | 3383 | actual_pld_candidate_sequence |
| qmsum_196_ctx384 | 4 | batched | 2 | no_argmax_divergence_in_selected_row | 10435 | 10435 | 10435 | actual_pld_candidate_sequence |
| qmsum_54_ctx384 | 4 | batched | 15 | no_argmax_divergence_in_selected_row | 70703 | 70703 | 70703 | actual_pld_candidate_sequence |
| narrativeqa_155_ctx384 | 4 | batched | 10 | no_argmax_divergence_in_selected_row | 1486 | 1486 | 1486 | actual_pld_candidate_sequence |
| qasper_163_ctx384 | 4 | batched | 2 | no_argmax_divergence_in_selected_row | 2661 | 2661 | 2661 | actual_pld_candidate_sequence |
| narrativeqa_139_ctx384 | 4 | batched | 6 | no_argmax_divergence_in_selected_row | 975 | 975 | 975 | actual_pld_candidate_sequence |
| narrativeqa_55_ctx384 | 4 | batched | 2 | no_argmax_divergence_in_selected_row | 3383 | 3383 | 3383 | actual_pld_candidate_sequence |

## Interpretation

Classifications describe selected-row greedy argmax decisions, not sequence-level guarantees. A nonzero logit norm alone is not treated as a token change. The actual-cycle results attribute selected mismatches to KV history for 15 prompts and execution shape for 1 prompt. The K=1/2/4 counterfactuals show how batch sensitivity changes with candidate length; they should not be reported as actual PLD decisions where the cycle used K=0.
