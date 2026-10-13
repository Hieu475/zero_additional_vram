# Reduced-sync repeated screening (RTX 4050)

## Protocol

Uninstrumented paired benchmark of `reference` and `reduced_sync_conservative`, with Qwen2.5-3B NF4, K=2, 32 generated tokens, and 30 prompts. CKA90 and CKA67 each ran in three independent processes/sessions with distinct seeds and one repeat per prompt. The prompt set was held constant. Each session includes its own raw measurements, environment, and prompt-bootstrap analysis. See `screening_manifest.json` for commands and `screening_summary.json` for extracted results.

## Results

| Config | Session | E2E speedup (prompt mean; 95% prompt bootstrap CI) | Decode speedup (prompt mean; 95% prompt bootstrap CI) | Exact match |
|---|---:|---:|---:|---:|
| CKA90 | 1 | 1.007 [0.997, 1.019] | 1.003 [0.995, 1.010] | 30/30 |
| CKA90 | 2 | 0.995 [0.984, 1.005] | 1.000 [0.991, 1.008] | 30/30 |
| CKA90 | 3 | 0.996 [0.979, 1.014] | 0.992 [0.977, 1.007] | 30/30 |
| CKA67 | 1 | 1.012 [0.993, 1.032] | 1.016 [0.997, 1.035] | 30/30 |
| CKA67 | 2 | 0.991 [0.976, 1.006] | 0.986 [0.972, 1.001] | 30/30 |
| CKA67 | 3 | 1.004 [0.985, 1.024] | 1.009 [0.990, 1.028] | 30/30 |

All per-session confidence intervals include 1.0. The descriptive across-session means are CKA90 E2E 0.999 [0.985, 1.013], decode 0.998 [0.984, 1.012]; CKA67 E2E 1.003 [0.977, 1.029], decode 1.004 [0.967, 1.041]. These intervals use only three session means and a t interval, so they are imprecise.

## Interpretation

This screening does not establish a throughput improvement. It supports retaining reduced-sync as an engineering cleanup or secondary ablation, not as a primary performance contribution. All measured output sequences matched exactly. Profiling instrumentation was disabled for these throughput measurements; the prior synchronization-count result remains a separate profiling observation.
