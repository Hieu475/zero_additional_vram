# Qwen2.5-3B NF4 precision/backend validation

Execution commit: `19bcc9d7827b2cc47969b8ad867c6c22e51dddeb`. Model revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
Hardware: `NVIDIA GeForce RTX 4050 Laptop GPU`; cohort: `{'diagnostic': 16, 'heldout': 14}`.
NF4 double quant, FP16 compute, greedy decode, 32 token limit, K=4.

## Cohort results

| Config | Cohort | N | Exact match | KV drift | Logit drift | Token divergence | Mean argmax agreement | TPS | VRAM MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sdpa_static | diagnostic | 16 | 0.000 | 16 | 16 | 16 | 0.7031 | 25.14 | 2111 |
| sdpa_static | heldout | 14 | 1.000 | 14 | 14 | 0 | 1.0000 | 26.46 | 2111 |

WP2 intervention labels linked for the 16 diagnostic prompts in sdpa_static: `{'kv_history_sensitive': 15, 'execution_sensitive': 1}`. Held-out prompts have no WP2 label.

| eager_static | diagnostic | 16 | 0.500 | 16 | 16 | 8 | 0.7520 | 23.36 | 3058 |
| eager_static | heldout | 14 | 1.000 | 14 | 14 | 0 | 1.0000 | 23.37 | 2112 |

WP2 intervention labels linked for the 16 diagnostic prompts in eager_static: `{'kv_history_sensitive': 15, 'execution_sensitive': 1}`. Held-out prompts have no WP2 label.

| sdpa_dynamic | diagnostic | 16 | 0.000 | 16 | 16 | 16 | 0.7031 | 25.34 | 2798 |
| sdpa_dynamic | heldout | 14 | 1.000 | 14 | 14 | 0 | 1.0000 | 26.85 | 2110 |

WP2 intervention labels linked for the 16 diagnostic prompts in sdpa_dynamic: `{'kv_history_sensitive': 15, 'execution_sensitive': 1}`. Held-out prompts have no WP2 label.

| sdpa_static_sequential | diagnostic | 16 | 1.000 | 0 | 0 | 0 | 1.0000 | 22.90 | 2454 |
| sdpa_static_sequential | heldout | 14 | 1.000 | 0 | 0 | 0 | 1.0000 | 23.51 | 2113 |

WP2 intervention labels linked for the 16 diagnostic prompts in sdpa_static_sequential: `{'kv_history_sensitive': 15, 'execution_sensitive': 1}`. Held-out prompts have no WP2 label.

## Interpretation and limits

KV/logit drift, argmax changes and sequence mismatch are counted separately. A numerical difference does not imply a changed token. Exact-match applies only to this 30-prompt prepared-input cohort and greedy 32-token run; it is not a lossless guarantee.
The held-out rows are unselected rows from the same prepared-input artifact, not an external dataset. FP16/BF16 unquantized and FP32 controls were not run on this 6 GB GPU; they remain open.
For diagnostic prompts, WP2 attribution is an intervention result from the matched SDPA/static protocol. It is linked for context and is not re-estimated independently for the other backend configurations.
TPS is measured with numerical tracing enabled and includes trace snapshot overhead; treat it as diagnostic instrumentation timing, not a production speed benchmark. Verification latency measures the target verification forward pass. Max-relative error can become very large when reference logits are near zero; interpret it with MAE, max-absolute error and top-1 margin.
Per-prompt token IDs, cycle-level KV/logit errors, margins, EOS positions and environment manifest are stored in the adjacent JSONL and manifest files.
