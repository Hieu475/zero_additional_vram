# Qwen2.5-3B NF4 precision/backend validation

Execution commit: `19bcc9d7827b2cc47969b8ad867c6c22e51dddeb`. Model revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`. 
Hardware: `NVIDIA GeForce RTX 4050 Laptop GPU`; cohort: {'diagnostic': 16, 'heldout': 14}.
NF4 double quant, FP16 compute, greedy decode, 32 token limit, K=4.

## Results

| Config | N | Exact match | Mean argmax agreement | First token divergence | TPS | Verify latency (s) | Peak VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdpa_static | 30 | 0.467 | 0.8417 | 16 | 26.40 | 0.8091 | 2111 |
| eager_static | 30 | 0.733 | 0.8677 | 8 | 23.52 | 0.9418 | 2617 |
| sdpa_dynamic | 30 | 0.467 | 0.8417 | 16 | 26.58 | 0.8343 | 2477 |
| sdpa_static_sequential | 30 | 1.000 | 1.0000 | 0 | 23.73 | 0.9133 | 2295 |

## Interpretation and limits

Numerical drift is recorded independently from argmax and sequence divergence. Any nonzero logit/KV error is not itself evidence of changed output; exact-match is only for this labeled cohort and 32-token greedy run. It does not establish a lossless guarantee.
The held-out rows are unselected rows from the same prepared-input artifact, not an external dataset. FP16/BF16 unquantized and FP32 controls are not measured by this NF4 GPU matrix; they remain open because this run targets the 6 GB RTX 4050.
Raw per-prompt measurements and software/hardware manifest are stored beside this report.
