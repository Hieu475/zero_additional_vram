# Benchmark–trace reproduction audit

## Scope

This audit compares saved artifacts and includes controlled live replays. The benchmark has raw measurements and prepared prompt IDs; the causal trace has cycle snapshots. The benchmark artifact itself does not save candidate IDs, acceptance decisions, or per-cycle cache lengths, so its historical first execution difference is reconstructed by replaying the same prompt cohort under each artifact's quantization settings.

## Results

- Matched trace prompts: 16; benchmark prepared prompts: 60.
- Prompt token IDs are present in both benchmark and trace source files and compare equal: 16/16. The trace itself does not persist the consumed input IDs, so this verifies the referenced input artifact, not the live model call.
- Vanilla output first-difference present: 12/16.
- Benchmark PLD K=4 differs from its paired vanilla output: 16/16; saved trace PLD output differs from trace vanilla: 8/16.
- The eight trace-exact cases are `multi_news_189_ctx384, gov_report_80_ctx384, narrativeqa_6_ctx384, multi_news_143_ctx384, qmsum_4_ctx384, qmsum_196_ctx384, qmsum_54_ctx384, narrativeqa_55_ctx384`; trace mismatch cases are `qasper_119_ctx384, qasper_109_ctx384, gov_report_98_ctx384, gov_report_134_ctx384, qmsum_2_ctx384, narrativeqa_155_ctx384, qasper_163_ctx384, narrativeqa_139_ctx384`.
- Benchmark-versus-trace vanilla token IDs differ for 12/16 prompts: `qasper_119_ctx384, gov_report_98_ctx384, gov_report_134_ctx384, gov_report_80_ctx384, narrativeqa_6_ctx384, multi_news_143_ctx384, qmsum_4_ctx384, qmsum_196_ctx384, narrativeqa_155_ctx384, qasper_163_ctx384, narrativeqa_139_ctx384, narrativeqa_55_ctx384`.
- Benchmark run commit: `1e7e300e0766f68970a642fe8bd575d977ad5d26`; trace run commit is not recorded exactly; current checkout is `19bcc9d7`.
- Trace provenance records base commit `6149a670b983a3870fa88769da1bfb19dd94305d` with uncommitted research changes; there is no exact trace execution commit. The benchmark session records `1e7e300e0766f68970a642fe8bd575d977ad5d26`.
- The captured hashes for the trace runner and PLD generator match the tracked source at current `HEAD`; the trace artifact still lacks a clean-tree commit stamp.
- Shared recorded settings: model revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, NF4, FP16 compute, SDPA, static KV, K=4, 32-token budget.
- **Quantization mismatch confirmed:** the benchmark explicitly sets `bnb_4bit_use_double_quant=True`; the tracer's model loader omits that setting, whose installed BitsAndBytesConfig default is `False`. The two runs therefore used different quantized weights. The controlled no-double-quant replay reproduces the archived trace outputs and its 8/16 result; the benchmark-quant replay reproduces all 16 benchmark PLD outputs and its 16/16 mismatch result.
- PLD cache capacity is prompt length + budget + K + 16 in both runners (436 for 384-token prompts). However, the benchmark vanilla reference uses `max(2048, prompt_length + budget + 8)` (2048), while the tracer's vanilla reference uses the PLD capacity (436). This is a concrete cache-shape mismatch in the vanilla comparison and a plausible numerical confound, not yet proven to cause any specific token change.
- Tokenizer revision is pinned to the same model revision; both greedy paths stop after emitting EOS or reaching the 32-token budget. Random/order seeds are not evidence of execution equivalence.

## Live replay

Replay artifact: [`live_replay.json`](live_replay.json), execution commit `19bcc9d7827b2cc47969b8ad867c6c22e51dddeb`, quantization double-quant `True`.

- Production vs lightweight trace output: 16/16 equal.
- Lightweight vs full numerical trace output: 16/16 equal; recorded cycle states equal: 16/16.
- Earliest current-vs-archived trace state differences (selected target token):
- `qasper_119_ctx384`: {'cycle_index': 20, 'field': 'selected_next_token_id', 'left': 1084, 'right': 576}
- `multi_news_189_ctx384`: {'cycle_index': 24, 'field': 'selected_next_token_id', 'left': 1084, 'right': 13612}
- `qasper_109_ctx384`: {'cycle_index': 9, 'field': 'selected_next_token_id', 'left': 9760, 'right': 7982}
- `gov_report_98_ctx384`: {'cycle_index': 16, 'field': 'selected_next_token_id', 'left': 33845, 'right': 5646}
- `gov_report_134_ctx384`: {'cycle_index': 16, 'field': 'selected_next_token_id', 'left': 311, 'right': 10953}
- `gov_report_80_ctx384`: {'cycle_index': 8, 'field': 'selected_next_token_id', 'left': 1824, 'right': 7789}
- `qmsum_2_ctx384`: {'cycle_index': 6, 'field': 'selected_next_token_id', 'left': 5267, 'right': 30}
- `narrativeqa_6_ctx384`: {'cycle_index': 3, 'field': 'selected_next_token_id', 'left': 3042, 'right': 264}
- `multi_news_143_ctx384`: {'cycle_index': 5, 'field': 'selected_next_token_id', 'left': 12126, 'right': 2692}
- `qmsum_54_ctx384`: {'cycle_index': 20, 'field': 'selected_next_token_id', 'left': 11, 'right': 13}
- `narrativeqa_155_ctx384`: {'cycle_index': 14, 'field': 'selected_next_token_id', 'left': 1221, 'right': 1246}
- `qasper_163_ctx384`: {'cycle_index': 4, 'field': 'selected_next_token_id', 'left': 11376, 'right': 279}
- `narrativeqa_55_ctx384`: {'cycle_index': 15, 'field': 'selected_next_token_id', 'left': 26573, 'right': 785}

The live replay used the benchmark's double-quant setting. These results show tracing itself did not change the current-run outputs. The archived trace used a different quantization setting, so its selected-token differences are not evidence of instrumentation effects.

## Matched-to-tracer quantization replay

Replay artifact: [`no_double_quant_replay.json`](no_double_quant_replay.json), identical 16 prompts and runtime protocol, with `bnb_4bit_use_double_quant=False` to match the archived tracer loader.

- Replay PLD output matches the archived tracer output: 16/16.
- Replay PLD mismatch against the archived tracer's vanilla sequence: 8/16 (the archived 8/16 result).
- First selected-state differences between benchmark quantization and tracer quantization:
- `qasper_119_ctx384`: cycle 20, `selected_next_token_id`; double-quant `1084`, single-quant `576`.
- `multi_news_189_ctx384`: cycle 24, `selected_next_token_id`; double-quant `1084`, single-quant `13612`.
- `qasper_109_ctx384`: cycle 9, `selected_next_token_id`; double-quant `9760`, single-quant `7982`.
- `gov_report_98_ctx384`: cycle 16, `selected_next_token_id`; double-quant `33845`, single-quant `5646`.
- `gov_report_134_ctx384`: cycle 16, `selected_next_token_id`; double-quant `311`, single-quant `10953`.
- `gov_report_80_ctx384`: cycle 8, `selected_next_token_id`; double-quant `1824`, single-quant `7789`.
- `qmsum_2_ctx384`: cycle 6, `selected_next_token_id`; double-quant `5267`, single-quant `30`.
- `narrativeqa_6_ctx384`: cycle 3, `selected_next_token_id`; double-quant `3042`, single-quant `264`.
- `multi_news_143_ctx384`: cycle 5, `selected_next_token_id`; double-quant `12126`, single-quant `2692`.
- `qmsum_54_ctx384`: cycle 20, `selected_next_token_id`; double-quant `11`, single-quant `13`.
- `narrativeqa_155_ctx384`: cycle 14, `selected_next_token_id`; double-quant `1221`, single-quant `1246`.
- `qasper_163_ctx384`: cycle 4, `selected_next_token_id`; double-quant `11376`, single-quant `279`.
- `narrativeqa_55_ctx384`: cycle 15, `selected_next_token_id`; double-quant `26573`, single-quant `785`.

This controlled toggle reproduces each archived tracer output and its 8/16 exactness result, while the benchmark quantization replay matches the saved benchmark PLD outputs 16/16. The quantization configuration discrepancy explains the reported 8/16 versus 16/16 observation for this cohort.

Repeat probe: 4/4 selected prompts reproduced identical output and lightweight trajectory in a separate process.


## Interpretation

Prompt IDs compare exactly in the common referenced prepared-input artifact. The trace does not store the consumed prompt token array, limiting this to source-artifact equality. Trace-exactness is 8/16 in its saved artifact, while independent benchmark PLD K=4 is 0/16 exact. The quantization setting mismatch (double quant on benchmark, off in trace) means the traces were not using the same quantized weights. The matched setting replay reproduces all archived trace outputs and the 8/16 exactness rate; the benchmark setting replay reproduces all saved benchmark PLD outputs and the 16/16 mismatch rate. Thus WP1's discrepancy is explained by the quantization configuration, with instrumentation showing no output change in the controlled current replays. Vanilla cache capacities still differ and remain a confound for comparisons between vanilla references.

## Required next evidence

WP1 discrepancy is resolved for this cohort. For future runs, record `bnb_4bit_use_double_quant` and exact commit directly in every run config. The vanilla reference comparison still needs a capacity-controlled replay (2048 vs 436) before attributing its 12/16 output differences. Expand beyond this diagnostic cohort and retain EOS and token-ID cycle evidence when collecting new data.
