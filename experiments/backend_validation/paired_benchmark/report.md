# Paired backend pilot

This smoke uses the same prepared 384-token prompt, pinned Qwen2.5-3B revision, static KV capacity 436, SDPA, greedy decoding, and 8-token output budget on both backends. Each backend was loaded once, all three methods received an 8-token warmup, and one measured repetition used the recorded randomized method order. This is harness and compatibility evidence, not a publishable performance result or confidence interval.

## Within-backend results

| Backend | Method | Decode TPS | Internal speedup | E2E (s) | Exact vs own vanilla | Peak allocated (MB) |
|---|---|---:|---:|---:|---|---:|
| hf_bnb_nf4 | vanilla | 38.45 | 1.000× | 0.349 | True | 2107.3 |
| hf_bnb_nf4 | pld_k4 | 32.90 | 0.856× | 0.385 | True | 2107.3 |
| hf_bnb_nf4 | cka90_k2 | 35.10 | 0.913× | 0.365 | True | 2110.8 |
| hqq_gemlite_w4a16 | vanilla | 28.44 | 1.000× | 0.414 | True | 2224.4 |
| hqq_gemlite_w4a16 | pld_k4 | 21.67 | 0.762× | 0.495 | True | 2224.4 |
| hqq_gemlite_w4a16 | cka90_k2 | 22.35 | 0.786× | 0.482 | True | 2228.0 |

The direct vanilla token sequences first differ across backends at generated position 3; this is cross-backend quantization fidelity, not speculative decoding correctness.

In this one-prompt pilot, both PLD K4 and CKA90/K2 were slower than vanilla within each backend, while all measured speculative outputs matched that backend's vanilla output for these eight tokens. Do not generalize this result: only one prompt and one measured repetition were used, and the short output can miss later divergence and context-dependent performance.

The first HQQ file (`hqq_smoke_paired.jsonl`) used only one warmup token. CKA's warmup did not exercise the full speculative path, so its large measured verification latency included cold-path work; that file is retained as exploratory evidence but excluded from this comparison.

Next: use a full-path warmup (at least one complete 32-token speculative run), run 3+ sessions over the matched prompt cohort, and keep the within-backend exactness record for every prompt. Expand contexts/output budgets only after that pilot is stable.
