# Held-out 128-token backend cohort

Three independent process sessions per backend; 20 prompts at 384 input tokens, 128 output tokens, full 128-token warmup, static KV capacity 520, greedy SDPA. The cohort was excluded from backend selection but may have appeared in earlier ZASSD studies. Input token hashes match across all paired methods and backends.

Intervals are two-way bootstrap intervals resampling sessions and prompts. Prompt cohort is fixed and sessions=3, so treat intervals as exploratory.

| Backend | Method | Valid / 60 | Decode TPS (two-way CI) | Paired speedup (two-way CI) | Slower fraction | Exact vs own vanilla |
|---|---|---:|---:|---:|---:|---:|
| hf_bnb_nf4 | vanilla | 60/60 | 38.95 [31.47, 43.28] | — | — | 60/60 |
| hf_bnb_nf4 | pld_k4 | 60/60 | 44.09 [35.98, 51.08] | 1.133 [1.056, 1.220] | 30.0% | 21/60 |
| hf_bnb_nf4 | cka90_k2 | 60/60 | 34.32 [27.29, 38.59] | 0.881 [0.857, 0.905] | 96.7% | 15/60 |
| hqq_gemlite_w4a16 | vanilla | 60/60 | 26.49 [25.27, 27.49] | — | — | 60/60 |
| hqq_gemlite_w4a16 | pld_k4 | 60/60 | 31.50 [28.60, 34.98] | 1.189 [1.089, 1.306] | 20.0% | 43/60 |
| hqq_gemlite_w4a16 | cka90_k2 | 60/60 | 21.53 [19.93, 22.81] | 0.811 [0.767, 0.842] | 100.0% | 43/60 |

Cross-backend vanilla exactness: 0/60 paired outputs. Failed/OOM or failed-warmup method runs: 0.

| Method | HQQ minus BNB internal speedup (two-way CI) |
|---|---:|
| pld_k4 | 0.056 [-0.004, 0.118] |
| cka90_k2 | -0.070 [-0.113, -0.034] |

The paired speedup is within-backend relative to that backend's vanilla. Raw cross-backend output equality measures quantization/backend fidelity, not speculative correctness. This cohort was held out from backend selection, but it may have appeared in earlier ZASSD studies.
