# HQQ + GemLite targeted Nsight Compute samples

Qwen2.5-3B-Instruct, RTX 4050 Laptop GPU (SM 8.9), 384-token prompt, 32-token output budget, SDPA/static KV=436. Three first matching launches per kernel family, captured after a 32-token warmup using Nsight Compute basic set. NCU replays kernels and perturbs timing; samples are not production latency.

| Kernel | Launch | Grid | Block | Duration (µs) | DRAM throughput (% peak) | SM throughput (% peak) | Registers/thread |
|---|---:|---|---|---:|---:|---:|---:|
| `gemv_INT_revsplitK_kernel` | 1 | (16, 32, 1) | (32, 1, 1) | 16.54 | 75.8 | 34.4 | 80 |
| `gemv_INT_revsplitK_kernel` | 2 | (4, 64, 1) | (32, 1, 1) | 4.54 | 35.8 | 18.2 | 79 |
| `gemv_INT_revsplitK_kernel` | 3 | (4, 64, 1) | (32, 1, 1) | 4.54 | 36.1 | 18.5 | 79 |
| `gemm_INT_kernel` | 1 | (96, 1, 1) | (256, 1, 1) | 259.01 | 7.9 | 26.9 | 100 |
| `gemm_INT_kernel` | 2 | (12, 1, 1) | (256, 1, 1) | 52.48 | 18.7 | 16.6 | 100 |
| `gemm_INT_kernel` | 3 | (12, 1, 1) | (256, 1, 1) | 52.54 | 18.7 | 16.4 | 100 |

The sampled HQQ GEMV launches range from about 4.5 to 16.5 µs; the three GEMM launches range from about 52.5 to 259.0 µs. The launch-to-launch spread indicates that shape/work phase matters and the first three launches should not be generalized to the whole model. These basic counters show moderate DRAM utilization for sampled GEMV and low DRAM utilization for GEMM, but do not provide matched FLOP/byte, arithmetic intensity, or a roofline classification.

Raw reports: `hqq_vanilla_gemv.ncu-rep` and `hqq_vanilla_gemm.ncu-rep`.
