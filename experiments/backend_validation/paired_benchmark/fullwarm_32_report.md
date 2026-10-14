# Paired backend full-warm pilot

Same prepared prompt (384 tokens and identical token-ID hash), model revision, KV capacity 436, SDPA, greedy decoding, and 32-token output budget. Each backend was loaded once in a fresh process. Vanilla, PLD K4, and CKA90/K2 each received a separate 32-token warmup before one measured repetition. This is a diagnostic pilot, not a confidence interval or a publication result.

| Backend | Method | Decode TPS | Internal speedup | E2E (s) | Exact vs own vanilla | First divergence | Peak allocated MB |
|---|---|---:|---:|---:|---|---:|---:|
| hf_bnb_nf4 | vanilla | 37.16 | 1.000× | 1.042 | True | None | 2107.3 |
| hf_bnb_nf4 | pld_k4 | 33.45 | 0.900× | 1.138 | False | 23 | 2111.1 |
| hf_bnb_nf4 | cka90_k2 | 33.63 | 0.905× | 1.121 | False | 23 | 2110.9 |
| hqq_gemlite_w4a16 | vanilla | 23.66 | 1.000× | 1.568 | True | None | 2224.4 |
| hqq_gemlite_w4a16 | pld_k4 | 20.92 | 0.884× | 1.726 | True | None | 2228.3 |
| hqq_gemlite_w4a16 | cka90_k2 | 20.18 | 0.853× | 1.786 | True | None | 2228.0 |

Direct vanilla outputs differ across backends at generated position 3. This records cross-backend quantization fidelity; it is not speculative correctness.

On this prompt, both speculative methods are slower than the vanilla baseline within each backend. HQQ+GemLite matches its own vanilla for all 32 output tokens; BNB PLD and CKA outputs both first diverge from BNB vanilla at token position 23. The candidate stack's vanilla decode throughput is lower in this single session, and its peak allocated VRAM is higher. These results do not show a backend advantage and cannot be generalized beyond this prompt/session.

The earlier `hqq_smoke_paired.jsonl` used only one warmup token and showed a cold CKA verification path; exclude its latency values. The 8-token warm pilot is recorded separately in `pilot_summary.json`.

Next validation: at least 3 independent sessions across the diagnostic cohort, then held-out prompts and the longer context/output grid. Keep within-backend exactness separate from cross-backend token fidelity.
