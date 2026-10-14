# Three-session paired backend benchmark

Cohort: 16 diagnostic prompts at 384 tokens, 32-token generation budget, 32-token full-path warmup, capacity 436, Qwen2.5-3B pinned revision, greedy SDPA. Each backend was loaded in three separate processes, with distinct order seeds. All timings are uninstrumented generation timings; session/prompt pairs are the unit for internal speedups. This is the diagnostic mismatch cohort, not held-out evaluation.

Decode TPS, E2E, and peak allocated VRAM are means across valid session/prompt runs; brackets show two-way bootstrap 95% intervals resampling independent sessions and prompts. Internal speedup is paired within the same backend, session, and prompt; its interval uses the same two-way resampling.

| Backend | Method | Valid / 48 | Decode TPS (95% CI) | Internal speedup (cluster CI) | Slower than vanilla | Exact vs own vanilla | Peak allocated MB |
|---|---|---:|---:|---:|---:|---:|---:|
| hf_bnb_nf4 | vanilla | 48/48 | 33.26 [32.52, 34.06] | — | — | 48/48 (100.0%) | 2107.3 |
| hf_bnb_nf4 | pld_k4 | 48/48 | 31.52 [29.94, 33.29] | 0.949 [0.905, 1.001] | 75.0% | 0/48 (0.0%) | 2111.0 |
| hf_bnb_nf4 | cka90_k2 | 48/48 | 27.96 [26.64, 29.14] | 0.841 [0.807, 0.870] | 100.0% | 12/48 (25.0%) | 2111.4 |
| hqq_gemlite_w4a16 | vanilla | 48/48 | 23.05 [22.10, 23.79] | — | — | 48/48 (100.0%) | 2224.4 |
| hqq_gemlite_w4a16 | pld_k4 | 48/48 | 22.84 [20.20, 26.95] | 0.989 [0.885, 1.152] | 77.1% | 48/48 (100.0%) | 2228.1 |
| hqq_gemlite_w4a16 | cka90_k2 | 48/48 | 17.99 [17.07, 18.87] | 0.783 [0.746, 0.820] | 100.0% | 47/48 (97.9%) | 2228.5 |

Cross-backend vanilla exactness: 0/48 paired outputs. First-token difference positions are retained in the JSON; they measure quantization/backend fidelity, not speculative correctness.
Recorded failed/OOM method runs: 0.

The comparison estimates backend stack performance on this one diagnostic prompt cohort. Quantization formats differ, so cross-backend raw token equality is not the correctness criterion. Within-backend exactness is reported separately. No claim of lossless acceleration follows unless every intended evaluation condition matches its own backend's vanilla output.

Raw per-session JSONL, prompt cohort, and analysis code are kept beside this report. Interpret confidence intervals with care: there are only three independent sessions and prompts are a fixed diagnostic cohort.
