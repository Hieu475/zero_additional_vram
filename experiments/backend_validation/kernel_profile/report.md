# Nsight Systems baseline profile

These are instrumented diagnostic captures for one fixed prompt (384 input tokens, 32-token budget), NF4 double-quant, FP16 compute, SDPA, static KV capacity 436. They are not production performance measurements. Each method was captured in a separate process after warmup.

GPU kernel time is the sum of reported kernel durations and may include overlap; CUDA API host durations are not additive to GPU time. Nsight Systems does not identify dequantization as causal merely because its kernel appears in the timeline.

| Method | Total GPU kernel time (ms) | NF4 GEMM (ms; share) | BNB dequant (ms; share) | Synchronize API (ms; calls) | Kernel launch API (ms; calls) |
|---|---:|---:|---:|---:|---:|
| vanilla | 828.2 | 312.8 (37.8%) | 66.5 (8.0%) | 378.1 (65) | 252.0 (64784) |
| pld_k4 | 811.1 | 400.0 (49.3%) | 54.3 (6.7%) | 270.6 (91) | 280.2 (59926) |
| cka90_k2 | 951.7 | 451.7 (47.5%) | 40.0 (4.2%) | 376.8 (95) | 233.5 (64589) |

## Interpretation

- BNB 4-bit GEMM and blockwise dequantization are measurable contributors. Their shares vary by method; the profile alone does not establish that either is the limiting critical path.
- Synchronization API totals are host-side API durations, not pure GPU idle time. Pair with an uninstrumented production benchmark and a timeline/critical-path inspection before attributing slowdown.
- Memory-transfer summaries are reported in results.json. Their small aggregate duration in these captures does not characterize DRAM bandwidth or arithmetic intensity.
- Targeted NCU throughput samples are summarized below. They do not provide a matched FLOP/byte estimate, so no arithmetic-intensity claim is made.
- Environment, capture configuration, raw capture hashes and exported summary paths are stored in manifest.json.

## Targeted Nsight Compute samples

Nsight Compute used the `basic` set, restricted to the CUDA profiler start/stop window (`--profile-from-start off`) and the first three matching launches per run. Model loading and the four-token warmup were excluded. Each row below lists the three distinct launch samples in order; kernel timing/counters are perturbed by NCU replay and are not end-to-end performance measurements.

| Workload / kernel | Duration samples (µs) | DRAM throughput samples (% peak) | SM throughput samples (% peak) | Registers/thread |
|---|---|---|---|---:|
| vanilla_gemm | 20.35, 6.85, 6.24 | 62.6, 33.1, 24.9 | 40.8, 15.6, 16.1 | 46 / 46 / 46 |
| vanilla_dequant | 2.66, 38.88, 2.40 | 35.6, 91.0, 4.4 | 25.6, 92.0, 3.5 | 40 / 44 / 40 |
| cka90_k2_gemm | 23.14, 6.43, 6.43 | 66.2, 26.9, 28.7 | 36.6, 15.4, 15.5 | 46 / 46 / 46 |
| cka90_k2_dequant | 2.88, 38.59, 2.37 | 52.5, 90.8, 4.8 | 23.2, 84.4, 3.6 | 40 / 44 / 40 |

The blockwise dequant sample set contains two short launches with low DRAM utilization and one larger launch near 91% of peak DRAM throughput in both vanilla and CKA90/K2. This shows strong shape dependence: a subset of dequant launches can be memory intensive, while the kernel family as a whole should not be labeled the bottleneck from presence alone. The sampled NF4 GEMMs show lower SM throughput than peak and moderate DRAM utilization. These few samples do not establish a roofline location, overall critical-path share, or method-level causal performance difference.

No arithmetic-intensity claim is made: the collected basic counters provide throughput and occupancy context but not a matched FLOP/byte count for these kernels. Extending to a roofline section or source-level counters should be done only after choosing representative launch shapes from the full trace.

Raw NCU reports: `vanilla_4bit_gemm.ncu-rep`, `vanilla_dequant.ncu-rep`, `cka90_4bit_gemm.ncu-rep`, and `cka90_dequant.ncu-rep`. `ncu_manifest.json` records hashes, software versions, scope, and filter protocol.
