# Held-out 128-token backend pilot

One independent process/session per backend; 20 prompts at 384 input tokens, 128 output tokens, full 128-token warmup, static KV capacity 520, greedy SDPA. The cohort was excluded from backend selection but may have appeared in earlier ZASSD studies. Input token hashes match across all paired methods and backends.

Intervals are prompt bootstrap intervals over this one session. They do not estimate session-to-session variance and are exploratory.

| Backend | Method | Valid / 20 | Decode TPS (prompt CI) | Paired speedup (prompt CI) | Slower fraction | Exact vs own vanilla |
|---|---|---:|---:|---:|---:|---:|
| hf_bnb_nf4 | vanilla | 20/20 | 31.27 [30.74, 31.92] | — | — | 20/20 |
| hf_bnb_nf4 | pld_k4 | 20/20 | 35.14 [32.84, 37.59] | 1.126 [1.049, 1.214] | 30.0% | 7/20 |
| hf_bnb_nf4 | cka90_k2 | 20/20 | 27.10 [26.59, 27.61] | 0.869 [0.841, 0.891] | 100.0% | 5/20 |
| hqq_gemlite_w4a16 | vanilla | 20/20 | 25.17 [24.51, 25.79] | — | — | 20/20 |
| hqq_gemlite_w4a16 | pld_k4 | 20/20 | 30.44 [27.57, 33.71] | 1.207 [1.107, 1.323] | 10.0% | 14/20 |
| hqq_gemlite_w4a16 | cka90_k2 | 20/20 | 20.11 [18.66, 21.15] | 0.796 [0.742, 0.833] | 100.0% | 14/20 |

Cross-backend vanilla exactness: 0/20. Failed/OOM or failed-warmup method runs: 0.

This is a one-session pilot, not the planned repeated evaluation. The paired speedup is within-backend relative to that backend's vanilla, while raw cross-backend output equality is only a quantization/backend fidelity observation.
