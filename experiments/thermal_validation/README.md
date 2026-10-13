# RTX 4050 repeated PLD thermal validation

## Protocol

Three paired sessions per condition use the same 12 prepared 384-token prompts, Qwen2.5-3B NF4 with FP16 compute, SDPA, static KV, and 32 generated tokens. Each prompt runs vanilla, forced-miss PLD K=4, PLD K=4, and PLD K=8 in randomized method order. Telemetry is sampled once per second. The three sessions are independent process runs but were not collected on three separate days; condition remains partly confounded with run occasion.

## Session-level results

Values are the mean of each session’s 12 paired prompt speedups, then the mean and Student-t 95% interval across the three session means (df=2). Exactness is exact output token identity against vanilla.

| Condition | Method | Session ratios | Mean (session t 95% CI) | Exact per session |
|---|---|---|---:|---:|
| cool | pld_k4 | 1.055, 1.028, 1.048 | 1.043 [1.009, 1.078] | 7/7/7 (of 12) |
| cool | pld_k8 | 1.089, 1.063, 1.074 | 1.075 [1.044, 1.107] | 7/7/7 (of 12) |
| cool | pld_forced_miss_k4 | 0.968, 0.954, 0.960 | 0.961 [0.944, 0.977] | 12/12/12 (of 12) |
| hot | pld_k4 | 1.038, 1.056, 1.057 | 1.050 [1.024, 1.076] | 7/7/7 (of 12) |
| hot | pld_k8 | 1.089, 1.097, 1.092 | 1.093 [1.083, 1.103] | 7/7/7 (of 12) |
| hot | pld_forced_miss_k4 | 0.970, 0.968, 0.961 | 0.966 [0.955, 0.977] | 12/12/12 (of 12) |

## What the data supports

PLD K=4/K=8 produced similar speedup ranges in cool-start and heat-soaked sessions, and the session means were repeatable across these three runs. The 12-prompt set is small, and the session-level intervals are based on only three runs; treat them as exploratory. The hot session records reached 82–84°C in later measurements, while cool-start runs naturally warmed into the 70s. The telemetry records GPU temperature, clock, utilization, and sampled power.

Exactness remained 7/12 for K=4 and K=8 in every session. The same five prompts differed from vanilla in all six sessions: `gov_report_80_ctx384`, `multi_news_143_ctx384`, `narrativeqa_55_ctx384`, `qmsum_196_ctx384`, and `qmsum_54_ctx384`. Vanilla token histories were identical across sessions. Their speedups therefore do not support a lossless-acceleration claim, and the repeated prompt-level mismatches are stable under both thermal conditions. Forced-miss PLD matched all 12 prompts and ran below vanilla speed in each session, as expected for added speculative overhead.

Temperature was observed rather than controlled during measured prompts, and each condition was collected in a separate session. No claim of causal thermal invariance is warranted. Full session data and analysis are in `rtx4050_session*/` and `thermal_repeated_analysis.json`.
