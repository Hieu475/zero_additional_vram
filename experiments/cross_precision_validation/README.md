# Cross-precision and backend mechanism screen

This matrix used Qwen2.5-0.5B-Instruct on the RTX 4050, four held-out prompt IDs, K=4, and a 16-token output budget. It is mechanism evidence for this model and setup only; it does not directly establish Qwen2.5-3B behavior. All 14 configured cells completed. Raw traces, per-cell configs/environments, attributions, and commands are stored under this directory.

| Precision | Attention | Cache | Verification | KV drift | Logit drift | Token divergence | Exact |
|---|---|---|---|---:|---:|---:|---:|
| NF4 | sdpa | static | batched | 4/4 | 4/4 | 1/4 | 0.75 |
| NF4 | sdpa | static | sequential | 0/4 | 0/4 | 0/4 | 1.00 |
| NF4 | eager | static | batched | 4/4 | 4/4 | 2/4 | 0.50 |
| NF4 | eager | static | sequential | 0/4 | 0/4 | 0/4 | 1.00 |
| NF4 | sdpa | dynamic | batched | 4/4 | 4/4 | 1/4 | 0.75 |
| NF4 | sdpa | dynamic | sequential | 0/4 | 0/4 | 0/4 | 1.00 |
| FP16 | sdpa | static | batched | 4/4 | 4/4 | 1/4 | 0.75 |
| FP16 | sdpa | static | sequential | 0/4 | 0/4 | 0/4 | 1.00 |
| BF16 | sdpa | static | batched | 4/4 | 4/4 | 0/4 | 1.00 |
| BF16 | sdpa | static | sequential | 0/4 | 0/4 | 0/4 | 1.00 |
| FP32 | eager | static | batched | 4/4 | 4/4 | 0/4 | 1.00 |
| FP32 | eager | static | sequential | 0/4 | 0/4 | 0/4 | 1.00 |
| FP32 | sdpa | static | batched | 4/4 | 4/4 | 0/4 | 1.00 |
| FP32 | sdpa | static | sequential | 0/4 | 0/4 | 0/4 | 1.00 |

## Interpretation

Every batched cell had numerical KV and logit differences on all four prompts. Token divergence varied by precision/backend/cache and was absent in several cells despite numerical drift. Every sequential cell had no observed KV/logit difference and exact token agreement for this four-prompt sample. This pattern is consistent with differences arising from batched execution and its interaction with numeric format and cache/backend, rather than NF4 alone; the sample is too small to establish that causal generalization.

The 16-token horizon limits sequence-divergence observations. The matrix does not include Qwen2.5-3B FP16/BF16, larger prompt coverage, thermal repetitions, or speed measurements. Do not report the 0.5B results as direct validation of the target 3B model.

See `matrix_summary.json` for machine-readable results and per-cell configs/environments for run settings.
