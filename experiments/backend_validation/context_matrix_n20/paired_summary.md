# Output-128 context matrix: paired summary

Only structurally complete cells are included (20 prompts × 3 methods, successful warmups, no failed method rows, expected KV capacity). Each context has four prompts from each of five LongBench tasks: NarrativeQA, QASPER, GovReport, MultiNews, and QMSum. The 14 fresh clean reruns override the historical dirty/mixed files for analysis; those originals remain unchanged. The other 10 cells use prior clean records.

- Complete cells: **24/24**.
- Cells with fully clean working-tree and source-tree provenance: **24**.
- TPS definition: `(generated tokens - 1) / decode time`; speedup is paired against vanilla for the same backend, prompt, and session.
- Within-cell 95% intervals resample sessions and prompt documents stratified by LongBench task. Backend/context contrasts pair source documents and resample sessions independently within each condition; session numbers are not assumed to identify synchronized runs.
- Cross-backend speedup deltas compare inference stacks with different quantized weights. Replacement sessions ran in blocks (BNB before HQQ), not interleaved, so absolute cross-backend TPS is descriptive; this does not isolate a kernel-only effect.

## Clean-provenance cells

| Backend | Context | Method | Session IDs | Decode speedup | 95% interval | Decode slowdown | E2E speedup | Exact vs own vanilla | Token disagreement | First mismatch (median zero-based index) | Mean decode TPS | Peak allocated (MB) |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hf_bnb_nf4 | 128 | pld_k4 | 1,2,3 | 1.140× | [1.102, 1.180] | 13.3% | 1.135× | 18/60 | 28.8% | 67 | 48.58 | 2030 |
| hf_bnb_nf4 | 128 | cka90_k2 | 1,2,3 | 0.882× | [0.868, 0.898] | 98.3% | 0.885× | 15/60 | 25.7% | 66 | 37.60 | 2030 |
| hf_bnb_nf4 | 384 | pld_k4 | 1,2,3 | 1.104× | [1.052, 1.162] | 30.0% | 1.096× | 21/60 | 37.4% | 49 | 46.49 | 2115 |
| hf_bnb_nf4 | 384 | cka90_k2 | 1,2,3 | 0.887× | [0.872, 0.902] | 98.3% | 0.892× | 9/60 | 45.9% | 56 | 37.36 | 2115 |
| hf_bnb_nf4 | 1024 | pld_k4 | 1,2,3 | 1.039× | [0.972, 1.110] | 45.0% | 1.032× | 12/60 | 40.7% | 59 | 42.06 | 2327 |
| hf_bnb_nf4 | 1024 | cka90_k2 | 1,2,3 | 0.870× | [0.852, 0.884] | 100.0% | 0.885× | 6/60 | 59.2% | 26 | 35.28 | 2327 |
| hf_bnb_nf4 | 2048 | pld_k4 | 1,2,3 | 1.073× | [0.968, 1.213] | 55.0% | 1.041× | 27/60 | 30.4% | 49 | 44.04 | 2676 |
| hf_bnb_nf4 | 2048 | cka90_k2 | 1,2,3 | 0.844× | [0.822, 0.863] | 100.0% | 0.874× | 12/60 | 41.0% | 54 | 34.68 | 2677 |
| hqq_gemlite_w4a16 | 128 | pld_k4 | 1,2,3 | 1.148× | [1.099, 1.199] | 13.3% | 1.146× | 46/60 | 10.4% | 63 | 30.90 | 2146 |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | 1,2,3 | 0.825× | [0.811, 0.838] | 100.0% | 0.827× | 47/60 | 8.9% | 65 | 22.21 | 2147 |
| hqq_gemlite_w4a16 | 384 | pld_k4 | 1,2,3 | 1.171× | [1.079, 1.279] | 28.3% | 1.163× | 41/60 | 14.9% | 94 | 32.08 | 2232 |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | 1,2,3 | 0.807× | [0.785, 0.827] | 100.0% | 0.812× | 45/60 | 10.7% | 98 | 22.13 | 2232 |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | 1,2,3 | 1.143× | [1.057, 1.223] | 30.0% | 1.126× | 44/60 | 10.0% | 78 | 31.35 | 2453 |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | 1,2,3 | 0.824× | [0.809, 0.840] | 100.0% | 0.836× | 46/60 | 10.4% | 59 | 22.61 | 2445 |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | 1,2,3 | 1.077× | [1.011, 1.136] | 40.0% | 1.059× | 42/60 | 19.7% | 36 | 28.17 | 2808 |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | 1,2,3 | 0.822× | [0.806, 0.837] | 100.0% | 0.845× | 44/60 | 16.8% | 47 | 21.48 | 2799 |

## Complete cells including exploratory provenance

This section includes every complete selected cell. With the clean rerun overlay present, it currently uses the same 24 clean cells as the primary section; it remains as a cross-check if incomplete/dirty cells are later added.

| Backend | Context | Method | Session count | Decode speedup | 95% interval | Decode slowdown | E2E speedup | Exact vs own vanilla | Token disagreement | First mismatch (median zero-based index) | Mean decode TPS | Peak allocated (MB) |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hf_bnb_nf4 | 128 | pld_k4 | 3 | 1.140× | [1.102, 1.180] | 13.3% | 1.135× | 18/60 | 28.8% | 67 | 48.58 | 2030 |
| hf_bnb_nf4 | 128 | cka90_k2 | 3 | 0.882× | [0.868, 0.898] | 98.3% | 0.885× | 15/60 | 25.7% | 66 | 37.60 | 2030 |
| hf_bnb_nf4 | 384 | pld_k4 | 3 | 1.104× | [1.052, 1.162] | 30.0% | 1.096× | 21/60 | 37.4% | 49 | 46.49 | 2115 |
| hf_bnb_nf4 | 384 | cka90_k2 | 3 | 0.887× | [0.872, 0.902] | 98.3% | 0.892× | 9/60 | 45.9% | 56 | 37.36 | 2115 |
| hf_bnb_nf4 | 1024 | pld_k4 | 3 | 1.039× | [0.972, 1.110] | 45.0% | 1.032× | 12/60 | 40.7% | 59 | 42.06 | 2327 |
| hf_bnb_nf4 | 1024 | cka90_k2 | 3 | 0.870× | [0.852, 0.884] | 100.0% | 0.885× | 6/60 | 59.2% | 26 | 35.28 | 2327 |
| hf_bnb_nf4 | 2048 | pld_k4 | 3 | 1.073× | [0.968, 1.213] | 55.0% | 1.041× | 27/60 | 30.4% | 49 | 44.04 | 2676 |
| hf_bnb_nf4 | 2048 | cka90_k2 | 3 | 0.844× | [0.822, 0.863] | 100.0% | 0.874× | 12/60 | 41.0% | 54 | 34.68 | 2677 |
| hqq_gemlite_w4a16 | 128 | pld_k4 | 3 | 1.148× | [1.099, 1.199] | 13.3% | 1.146× | 46/60 | 10.4% | 63 | 30.90 | 2146 |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | 3 | 0.825× | [0.811, 0.838] | 100.0% | 0.827× | 47/60 | 8.9% | 65 | 22.21 | 2147 |
| hqq_gemlite_w4a16 | 384 | pld_k4 | 3 | 1.171× | [1.079, 1.279] | 28.3% | 1.163× | 41/60 | 14.9% | 94 | 32.08 | 2232 |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | 3 | 0.807× | [0.785, 0.827] | 100.0% | 0.812× | 45/60 | 10.7% | 98 | 22.13 | 2232 |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | 3 | 1.143× | [1.057, 1.223] | 30.0% | 1.126× | 44/60 | 10.0% | 78 | 31.35 | 2453 |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | 3 | 0.824× | [0.809, 0.840] | 100.0% | 0.836× | 46/60 | 10.4% | 59 | 22.61 | 2445 |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | 3 | 1.077× | [1.011, 1.136] | 40.0% | 1.059× | 42/60 | 19.7% | 36 | 28.17 | 2808 |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | 3 | 0.822× | [0.806, 0.837] | 100.0% | 0.845× | 44/60 | 16.8% | 47 | 21.48 | 2799 |

## Paired backend deltas

Positive values mean HQQ gains more internal speedup than BNB. Rows pair by prompt; replicate sessions are resampled independently within backend. They do not compare cross-backend token identity.

| Context | Method | Provenance | BNB session IDs | HQQ session IDs | Mean HQQ−BNB speedup delta | 95% interval |
|---:|---|---|---|---|---:|---|
| 128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.008 | [-0.046, 0.061] |
| 128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.057 | [-0.077, -0.038] |
| 128 | pld_k4 | clean | 1,2,3 | 1,2,3 | +0.008 | [-0.046, 0.061] |
| 128 | cka90_k2 | clean | 1,2,3 | 1,2,3 | -0.057 | [-0.077, -0.038] |
| 384 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.067 | [-0.014, 0.157] |
| 384 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.080 | [-0.103, -0.058] |
| 384 | pld_k4 | clean | 1,2,3 | 1,2,3 | +0.067 | [-0.014, 0.157] |
| 384 | cka90_k2 | clean | 1,2,3 | 1,2,3 | -0.080 | [-0.103, -0.058] |
| 1024 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.104 | [0.000, 0.202] |
| 1024 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.046 | [-0.067, -0.023] |
| 1024 | pld_k4 | clean | 1,2,3 | 1,2,3 | +0.104 | [0.000, 0.202] |
| 1024 | cka90_k2 | clean | 1,2,3 | 1,2,3 | -0.046 | [-0.067, -0.023] |
| 2048 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.004 | [-0.137, 0.120] |
| 2048 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.023 | [-0.046, 0.004] |
| 2048 | pld_k4 | clean | 1,2,3 | 1,2,3 | +0.004 | [-0.137, 0.120] |
| 2048 | cka90_k2 | clean | 1,2,3 | 1,2,3 | -0.023 | [-0.046, 0.004] |

## Paired context-length contrasts

Each contrast pairs the same source documents between ctx128 and the target context. Replicate sessions are resampled independently within each context. Positive decode or E2E deltas mean the method's speedup ratio improved at the longer context; negative exact-rate deltas mean exactness declined.

| Backend | Context contrast | Method | Provenance | ctx128 sessions | Target sessions | Decode speedup delta | 95% interval | E2E speedup delta | Exact-rate delta (pp) |
|---|---:|---|---|---|---|---:|---|---:|---:|
| hf_bnb_nf4 | 384-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.036 | [-0.111, 0.038] | -0.039 | +5.0 |
| hf_bnb_nf4 | 384-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | +0.005 | [-0.015, 0.024] | +0.008 | -10.0 |
| hf_bnb_nf4 | 384-minus-128 | pld_k4 | clean | 1,2,3 | 1,2,3 | -0.036 | [-0.111, 0.038] | -0.039 | +5.0 |
| hf_bnb_nf4 | 384-minus-128 | cka90_k2 | clean | 1,2,3 | 1,2,3 | +0.005 | [-0.015, 0.024] | +0.008 | -10.0 |
| hf_bnb_nf4 | 1024-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.101 | [-0.187, -0.015] | -0.104 | -10.0 |
| hf_bnb_nf4 | 1024-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.012 | [-0.032, 0.007] | -0.000 | -15.0 |
| hf_bnb_nf4 | 1024-minus-128 | pld_k4 | clean | 1,2,3 | 1,2,3 | -0.101 | [-0.187, -0.015] | -0.104 | -10.0 |
| hf_bnb_nf4 | 1024-minus-128 | cka90_k2 | clean | 1,2,3 | 1,2,3 | -0.012 | [-0.032, 0.007] | -0.000 | -15.0 |
| hf_bnb_nf4 | 2048-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.067 | [-0.182, 0.080] | -0.094 | +15.0 |
| hf_bnb_nf4 | 2048-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.038 | [-0.063, -0.014] | -0.011 | -5.0 |
| hf_bnb_nf4 | 2048-minus-128 | pld_k4 | clean | 1,2,3 | 1,2,3 | -0.067 | [-0.182, 0.080] | -0.094 | +15.0 |
| hf_bnb_nf4 | 2048-minus-128 | cka90_k2 | clean | 1,2,3 | 1,2,3 | -0.038 | [-0.063, -0.014] | -0.011 | -5.0 |
| hqq_gemlite_w4a16 | 384-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.023 | [-0.074, 0.128] | +0.017 | -8.3 |
| hqq_gemlite_w4a16 | 384-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.018 | [-0.044, 0.007] | -0.015 | -3.3 |
| hqq_gemlite_w4a16 | 384-minus-128 | pld_k4 | clean | 1,2,3 | 1,2,3 | +0.023 | [-0.074, 0.128] | +0.017 | -8.3 |
| hqq_gemlite_w4a16 | 384-minus-128 | cka90_k2 | clean | 1,2,3 | 1,2,3 | -0.018 | [-0.044, 0.007] | -0.015 | -3.3 |
| hqq_gemlite_w4a16 | 1024-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.005 | [-0.100, 0.083] | -0.020 | -3.3 |
| hqq_gemlite_w4a16 | 1024-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.001 | [-0.022, 0.019] | +0.009 | -1.7 |
| hqq_gemlite_w4a16 | 1024-minus-128 | pld_k4 | clean | 1,2,3 | 1,2,3 | -0.005 | [-0.100, 0.083] | -0.020 | -3.3 |
| hqq_gemlite_w4a16 | 1024-minus-128 | cka90_k2 | clean | 1,2,3 | 1,2,3 | -0.001 | [-0.022, 0.019] | +0.009 | -1.7 |
| hqq_gemlite_w4a16 | 2048-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.071 | [-0.151, 0.001] | -0.087 | -6.7 |
| hqq_gemlite_w4a16 | 2048-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.004 | [-0.023, 0.015] | +0.017 | -5.0 |
| hqq_gemlite_w4a16 | 2048-minus-128 | pld_k4 | clean | 1,2,3 | 1,2,3 | -0.071 | [-0.151, 0.001] | -0.087 | -6.7 |
| hqq_gemlite_w4a16 | 2048-minus-128 | cka90_k2 | clean | 1,2,3 | 1,2,3 | -0.004 | [-0.023, 0.015] | +0.017 | -5.0 |

## Acceptance versus speedup

Spearman association uses each prompt's mean acceptance rate and internal speedup over the available sessions. Its interval resamples prompts within task and is conditional on those sessions; it is descriptive, not a causal effect.

| Backend | Context | Method | Provenance | Prompts | Spearman rho | 95% prompt interval |
|---|---:|---|---|---:|---:|---|
| hf_bnb_nf4 | 128 | pld_k4 | exploratory | 20 | +0.944 | [0.861, 0.980] |
| hf_bnb_nf4 | 128 | cka90_k2 | exploratory | 20 | +0.775 | [0.508, 0.922] |
| hf_bnb_nf4 | 128 | pld_k4 | clean | 20 | +0.944 | [0.861, 0.980] |
| hf_bnb_nf4 | 128 | cka90_k2 | clean | 20 | +0.775 | [0.508, 0.922] |
| hf_bnb_nf4 | 384 | pld_k4 | exploratory | 20 | +0.880 | [0.716, 0.962] |
| hf_bnb_nf4 | 384 | cka90_k2 | exploratory | 20 | +0.836 | [0.618, 0.944] |
| hf_bnb_nf4 | 384 | pld_k4 | clean | 20 | +0.880 | [0.716, 0.962] |
| hf_bnb_nf4 | 384 | cka90_k2 | clean | 20 | +0.836 | [0.618, 0.944] |
| hf_bnb_nf4 | 1024 | pld_k4 | exploratory | 20 | +0.911 | [0.712, 0.983] |
| hf_bnb_nf4 | 1024 | cka90_k2 | exploratory | 20 | +0.890 | [0.706, 0.966] |
| hf_bnb_nf4 | 1024 | pld_k4 | clean | 20 | +0.911 | [0.712, 0.983] |
| hf_bnb_nf4 | 1024 | cka90_k2 | clean | 20 | +0.890 | [0.706, 0.966] |
| hf_bnb_nf4 | 2048 | pld_k4 | exploratory | 20 | +0.971 | [0.918, 0.991] |
| hf_bnb_nf4 | 2048 | cka90_k2 | exploratory | 20 | +0.833 | [0.628, 0.964] |
| hf_bnb_nf4 | 2048 | pld_k4 | clean | 20 | +0.971 | [0.918, 0.991] |
| hf_bnb_nf4 | 2048 | cka90_k2 | clean | 20 | +0.833 | [0.628, 0.964] |
| hqq_gemlite_w4a16 | 128 | pld_k4 | exploratory | 20 | +0.851 | [0.659, 0.929] |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | exploratory | 20 | +0.830 | [0.622, 0.936] |
| hqq_gemlite_w4a16 | 128 | pld_k4 | clean | 20 | +0.851 | [0.659, 0.929] |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | clean | 20 | +0.830 | [0.622, 0.936] |
| hqq_gemlite_w4a16 | 384 | pld_k4 | exploratory | 20 | +0.864 | [0.633, 0.959] |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | exploratory | 20 | +0.899 | [0.781, 0.964] |
| hqq_gemlite_w4a16 | 384 | pld_k4 | clean | 20 | +0.864 | [0.633, 0.959] |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | clean | 20 | +0.899 | [0.781, 0.964] |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | exploratory | 20 | +0.672 | [0.200, 0.964] |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | exploratory | 20 | +0.867 | [0.621, 0.967] |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | clean | 20 | +0.672 | [0.200, 0.964] |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | clean | 20 | +0.867 | [0.621, 0.967] |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | exploratory | 20 | +0.629 | [0.156, 0.929] |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | exploratory | 20 | +0.596 | [0.257, 0.821] |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | clean | 20 | +0.629 | [0.156, 0.929] |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | clean | 20 | +0.596 | [0.257, 0.821] |

## Cross-check with existing kernel profile

The separate instrumented BNB NF4 profile at context 384/output 32 reports dequantization kernel shares of 8.0% for vanilla, 6.7% for PLD K4, and 4.2% for CKA90/K2, while NF4 GEMM shares are 37.8%, 49.3%, and 47.5%. CKA90/K2 has the lowest dequant share yet remains slower in the paired context matrix. This supports avoiding a dequantization-only explanation; it does not establish what causes the slowdown because the profile is a single instrumented prompt with a different budget/capacity. See [`kernel_profile/report.md`](../kernel_profile/report.md).


## Evidence from the completed clean matrix

The clean-provenance point estimates show CKA90/K2 below 1× for both decode and end-to-end speedup in every completed cell. PLD K4 is not uniformly profitable: BNB ctx1024 is below 1×, and many prompt-level runs still slow down even when the cell mean exceeds 1×. Neither method reaches 100% exact sequence match in these clean cells, so the current evidence does not support a lossless claim.

The clean matched-source comparison from ctx128 to ctx1024 estimates a PLD speedup change of -0.101 for BNB (95% interval [-0.187, -0.015]) and -0.005 for HQQ (95% interval [-0.100, +0.083]). The BNB interval excludes zero while the HQQ interval crosses zero. This is evidence of different observed context sensitivity in this cohort; it does not isolate the cause because quantization and backend implementation differ.

## Interpretation limits

The matrix contains three clean sessions per backend/context. Session-level intervals are now estimable, but three session clusters remain a small basis for uncertainty estimates. The exploratory profile remains a one-prompt instrumented measurement and is not causal timing evidence.
