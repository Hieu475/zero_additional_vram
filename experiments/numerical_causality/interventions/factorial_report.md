# KV × verification factorial intervention

## Protocol

- Execution commit: `19bcc9d7827b2cc47969b8ad867c6c22e51dddeb`; model `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- NF4, FP16 compute, double quant enabled; SDPA; static KV; RTX 4050.
- Diagnostic cohort: 16 prompts; each PLD output mismatch vs the matched-capacity canonical sequential reference: 16/16.
- For each selected prompt prefix and fixed candidate sequence, four isolated cache clones were evaluated: A active KV + batch, B canonical KV + batch, C active KV + sequential, D canonical KV + sequential.
- Every condition began at the same logical prefix length; input token positions are `[prefix_length, prefix_length + candidate_length)`. Attention mask was omitted consistently so the model/cache constructs the causal mask from logical cache length.
- Cache snapshots were fingerprinted before intervention; source snapshots remained unchanged. All cache length and clone integrity checks passed: `True`.
- The selected row is the first active target argmax that differs from the same-prefix canonical vanilla continuation. Logit row `j` follows candidate input token `j` and predicts the next token. Each row logs argmax, top-1 margin, MAE, max absolute/relative error, L2 norm, and the factorial interaction norm. KV errors are recorded layer-wise.

## Attribution

Actual selected-cycle classification counts: `{"execution_sensitive": 1, "kv_history_sensitive": 15}`. Actual cycle K counts: `{"0": 13, "1": 1, "4": 2}`. Verification modes: `{"batched": 3, "vanilla_fallback": 13}`. Actual cycle input source: `{"actual_pld_candidate_sequence": 16}`. Counterfactual vanilla continuation extension was used in 14 prompts.

The main attribution uses the actual K at the first mismatch. For cycles that fell back with K=0, K=1/2/4 probes extend the fixed candidate sequence using the vanilla continuation; those are counterfactual execution-shape controls, not candidates PLD actually drafted. They are labeled in the raw JSONL. Three prompts had real speculative candidates at the selected cycle.

| Prompt | Actual K | Mode | Cycle | Attribution | A argmax | D argmax | Vanilla expected | Candidate source |
|---|---:|---|---:|---|---:|---:|---:|---|
| qasper_119_ctx384 | 0 | vanilla_fallback | 20 | kv_history_sensitive | 1084 | 576 | 576 | actual_pld_candidate_sequence |
| multi_news_189_ctx384 | 0 | vanilla_fallback | 24 | kv_history_sensitive | 1084 | 13612 | 13612 | actual_pld_candidate_sequence |
| qasper_109_ctx384 | 0 | vanilla_fallback | 13 | kv_history_sensitive | 2661 | 2197 | 2197 | actual_pld_candidate_sequence |
| gov_report_98_ctx384 | 0 | vanilla_fallback | 16 | kv_history_sensitive | 33845 | 5646 | 5646 | actual_pld_candidate_sequence |
| gov_report_134_ctx384 | 0 | vanilla_fallback | 16 | kv_history_sensitive | 311 | 10953 | 10953 | actual_pld_candidate_sequence |
| gov_report_80_ctx384 | 0 | vanilla_fallback | 8 | kv_history_sensitive | 1824 | 7789 | 7789 | actual_pld_candidate_sequence |
| qmsum_2_ctx384 | 1 | batched | 10 | kv_history_sensitive | 304 | 572 | 572 | actual_pld_candidate_sequence |
| narrativeqa_6_ctx384 | 0 | vanilla_fallback | 29 | kv_history_sensitive | 6109 | 4008 | 4008 | actual_pld_candidate_sequence |
| multi_news_143_ctx384 | 0 | vanilla_fallback | 29 | kv_history_sensitive | 2337 | 304 | 304 | actual_pld_candidate_sequence |
| qmsum_4_ctx384 | 4 | batched | 10 | execution_sensitive | 279 | 432 | 432 | actual_pld_candidate_sequence |
| qmsum_196_ctx384 | 0 | vanilla_fallback | 24 | kv_history_sensitive | 15723 | 279 | 279 | actual_pld_candidate_sequence |
| qmsum_54_ctx384 | 0 | vanilla_fallback | 20 | kv_history_sensitive | 11 | 13 | 13 | actual_pld_candidate_sequence |
| narrativeqa_155_ctx384 | 0 | vanilla_fallback | 14 | kv_history_sensitive | 1221 | 1246 | 1246 | actual_pld_candidate_sequence |
| qasper_163_ctx384 | 4 | batched | 13 | kv_history_sensitive | 2924 | 525 | 525 | actual_pld_candidate_sequence |
| narrativeqa_139_ctx384 | 0 | vanilla_fallback | 25 | kv_history_sensitive | 8747 | 17845 | 17845 | actual_pld_candidate_sequence |
| narrativeqa_55_ctx384 | 0 | vanilla_fallback | 19 | kv_history_sensitive | 374 | 16555 | 16555 | actual_pld_candidate_sequence |

## Interpretation

Classifications describe selected-row greedy argmax decisions, not sequence-level guarantees. A nonzero logit norm alone is not treated as a token change. The actual-cycle results attribute selected mismatches to KV history for 15 prompts and execution shape for 1 prompt. The K=1/2/4 counterfactuals show how batch sensitivity changes with candidate length; they should not be reported as actual PLD decisions where the cycle used K=0.
