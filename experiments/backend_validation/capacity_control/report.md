# KV capacity × NF4 double-quantization control

Cohort: 16 diagnostic prompts, 384 prompt tokens, 32-token greedy budget.
The reference protocol uses Qwen2.5-3B-Instruct, SDPA, FP16 compute, NF4 and a static cache.
Correctness capture and timing were separate passes; timing used direct model forwards without trace instrumentation.

## Paired factorial contrasts

| Contrast (right vs left) | Sequences equal | Mean decode TPS ratio [bootstrap 95% CI] | Prefill KV exact | Mean max logit abs error | Allocated VRAM delta |
|---|---:|---:|---:|---:|---:|
| capacity 436→2048, double_quant=True | 16/16 | 1.020 [0.999, 1.043] | 16/16 | 0 | +56.7 MB |
| capacity 436→2048, double_quant=False | 16/16 | 1.019 [1.000, 1.038] | 16/16 | 0 | +56.7 MB |
| double_quant True→False, capacity=436 | 4/16 | 1.013 [0.988, 1.041] | 0/16 | 0.656006 | +1502.4 MB |
| double_quant True→False, capacity=2048 | 4/16 | 1.012 [0.989, 1.036] | 0/16 | 0.656006 | +1502.4 MB |

## Reading the result

The capacity contrast is direct within each quantization setting. Prefill cache and logit comparisons use the same prompt token prefix; generated-prefix logits are compared only through the first position where the predicted token differs.
Quantization contrasts compare outputs from the same capacity and therefore do not mix in the cache allocation difference.
Latency and VRAM are descriptive for one run on one RTX 4050. Quantization order was fixed (double-quant on before off), so timing deltas are not treated as causal performance evidence. The unusually high peak allocated memory with double quantization disabled requires kernel-level profiling before assigning a cause.
Capacity changes reserve additional static KV backing memory but do not alter the logical cache length. This cohort is diagnostic and does not support population-level generalization.
