# Output-128 context matrix: paired preliminary summary

Only structurally complete cells are included (20 prompts × 3 methods, successful warmups, no failed method rows, expected KV capacity). Each context has four prompts from each of five LongBench tasks: NarrativeQA, QASPER, GovReport, MultiNews, and QMSum. Incomplete files are ignored. Clean-provenance results are shown separately from exploratory runs.

- Complete cells: **24/24**.
- Cells with fully clean working-tree and source-tree provenance: **10**.
- TPS definition: `(generated tokens - 1) / decode time`; speedup is paired against vanilla for the same backend, prompt, and session.
- Within-cell 95% intervals resample sessions and prompt documents stratified by LongBench task. Backend/context contrasts pair source documents and resample sessions independently within each condition; session numbers are not assumed to identify synchronized runs.
- Cross-backend comparison is HQQ internal speedup minus BNB internal speedup. Quantized weights differ, so this is an inference-stack comparison, not a kernel-only effect.

## Clean-provenance cells

| Backend | Context | Method | Session IDs | Decode speedup | 95% interval | Decode slowdown | E2E speedup | Exact vs own vanilla | Token disagreement | First mismatch (median zero-based index) | Mean decode TPS | Peak allocated (MB) |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hf_bnb_nf4 | 128 | pld_k4 | 3 | 1.126× | n/a (<2 sessions) | 15.0% | 1.122× | 6/20 | 28.8% | 67 | 50.37 | 2030 |
| hf_bnb_nf4 | 128 | cka90_k2 | 3 | 0.885× | n/a (<2 sessions) | 95.0% | 0.888× | 5/20 | 25.7% | 66 | 39.55 | 2030 |
| hf_bnb_nf4 | 384 | pld_k4 | 3 | 1.092× | n/a (<2 sessions) | 40.0% | 1.085× | 7/20 | 37.4% | 49 | 47.18 | 2115 |
| hf_bnb_nf4 | 384 | cka90_k2 | 3 | 0.890× | n/a (<2 sessions) | 100.0% | 0.896× | 3/20 | 45.9% | 56 | 38.52 | 2115 |
| hf_bnb_nf4 | 1024 | pld_k4 | 3 | 0.980× | n/a (<2 sessions) | 65.0% | 0.981× | 4/20 | 40.7% | 59 | 44.06 | 2327 |
| hf_bnb_nf4 | 1024 | cka90_k2 | 3 | 0.858× | n/a (<2 sessions) | 100.0% | 0.875× | 2/20 | 59.2% | 26 | 38.55 | 2327 |
| hf_bnb_nf4 | 2048 | pld_k4 | 3 | 1.040× | n/a (<2 sessions) | 60.0% | 1.018× | 9/20 | 30.4% | 49 | 43.87 | 2676 |
| hf_bnb_nf4 | 2048 | cka90_k2 | 3 | 0.827× | n/a (<2 sessions) | 100.0% | 0.860× | 4/20 | 41.0% | 54 | 34.75 | 2677 |
| hqq_gemlite_w4a16 | 128 | pld_k4 | 3 | 1.164× | n/a (<2 sessions) | 10.0% | 1.161× | 15/20 | 11.6% | 65 | 30.38 | 2146 |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | 3 | 0.821× | n/a (<2 sessions) | 100.0% | 0.823× | 17/20 | 4.8% | 90 | 21.42 | 2147 |
| hqq_gemlite_w4a16 | 384 | pld_k4 | 3 | 1.168× | n/a (<2 sessions) | 30.0% | 1.160× | 16/20 | 9.1% | 74 | 31.95 | 2232 |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | 3 | 0.804× | n/a (<2 sessions) | 100.0% | 0.809× | 15/20 | 9.9% | 98 | 22.03 | 2232 |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | 2,3 | 1.135× | [1.034, 1.229] | 30.0% | 1.119× | 29/40 | 10.9% | 68 | 30.52 | 2457 |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | 2,3 | 0.826× | [0.811, 0.841] | 100.0% | 0.838× | 31/40 | 10.0% | 19 | 22.24 | 2445 |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | 2,3 | 1.088× | [1.022, 1.145] | 37.5% | 1.068× | 27/40 | 21.7% | 25 | 29.18 | 2805 |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | 2,3 | 0.822× | [0.803, 0.842] | 100.0% | 0.846× | 28/40 | 20.3% | 36 | 22.04 | 2799 |

## Complete cells including exploratory provenance

These results include complete cells with historical dirty or missing provenance fields. Use them as sensitivity analysis, not as the clean-only result.

| Backend | Context | Method | Session count | Decode speedup | 95% interval | Decode slowdown | E2E speedup | Exact vs own vanilla | Token disagreement | First mismatch (median zero-based index) | Mean decode TPS | Peak allocated (MB) |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| hf_bnb_nf4 | 128 | pld_k4 | 3 | 1.136× | [1.097, 1.179] | 16.7% | 1.132× | 18/60 | 28.8% | 67 | 48.82 | 2030 |
| hf_bnb_nf4 | 128 | cka90_k2 | 3 | 0.885× | [0.868, 0.904] | 98.3% | 0.888× | 15/60 | 25.7% | 66 | 38.04 | 2030 |
| hf_bnb_nf4 | 384 | pld_k4 | 3 | 1.103× | [1.049, 1.162] | 26.7% | 1.096× | 21/60 | 37.4% | 49 | 45.93 | 2115 |
| hf_bnb_nf4 | 384 | cka90_k2 | 3 | 0.886× | [0.869, 0.903] | 100.0% | 0.891× | 9/60 | 45.9% | 56 | 36.95 | 2115 |
| hf_bnb_nf4 | 1024 | pld_k4 | 3 | 1.010× | [0.956, 1.071] | 55.0% | 1.007× | 12/60 | 40.7% | 59 | 44.01 | 2327 |
| hf_bnb_nf4 | 1024 | cka90_k2 | 3 | 0.863× | [0.849, 0.876] | 100.0% | 0.879× | 6/60 | 59.2% | 26 | 37.60 | 2327 |
| hf_bnb_nf4 | 2048 | pld_k4 | 3 | 1.031× | [0.936, 1.158] | 63.3% | 1.011× | 27/60 | 30.4% | 49 | 45.06 | 2676 |
| hf_bnb_nf4 | 2048 | cka90_k2 | 3 | 0.823× | [0.808, 0.838] | 100.0% | 0.857× | 12/60 | 41.0% | 54 | 35.98 | 2677 |
| hqq_gemlite_w4a16 | 128 | pld_k4 | 3 | 1.135× | [1.051, 1.209] | 20.0% | 1.132× | 50/60 | 8.0% | 65 | 30.35 | 2155 |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | 3 | 0.818× | [0.803, 0.834] | 100.0% | 0.820× | 47/60 | 10.6% | 63 | 21.90 | 2147 |
| hqq_gemlite_w4a16 | 384 | pld_k4 | 3 | 1.164× | [1.080, 1.266] | 30.0% | 1.155× | 49/60 | 6.2% | 98 | 31.57 | 2232 |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | 3 | 0.809× | [0.787, 0.828] | 100.0% | 0.815× | 45/60 | 8.2% | 98 | 21.95 | 2232 |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | 3 | 1.157× | [1.060, 1.248] | 30.0% | 1.138× | 43/60 | 12.1% | 68 | 30.98 | 2453 |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | 3 | 0.828× | [0.815, 0.842] | 100.0% | 0.840× | 47/60 | 10.3% | 19 | 22.21 | 2445 |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | 3 | 1.101× | [1.042, 1.158] | 30.0% | 1.080× | 43/60 | 17.1% | 47 | 29.68 | 2803 |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | 3 | 0.817× | [0.785, 0.838] | 100.0% | 0.840× | 41/60 | 21.2% | 47 | 21.97 | 2804 |

## Paired backend deltas

Positive values mean HQQ gains more internal speedup than BNB. Rows pair by prompt; replicate sessions are resampled independently within backend. They do not compare cross-backend token identity.

| Context | Method | Provenance | BNB session IDs | HQQ session IDs | Mean HQQ−BNB speedup delta | 95% interval |
|---:|---|---|---|---|---:|---|
| 128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.001 | [-0.097, 0.087] |
| 128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.067 | [-0.092, -0.041] |
| 128 | pld_k4 | clean | 3 | 3 | +0.037 | n/a (<2 sessions) |
| 128 | cka90_k2 | clean | 3 | 3 | -0.065 | n/a (<2 sessions) |
| 384 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.061 | [-0.017, 0.139] |
| 384 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.077 | [-0.103, -0.054] |
| 384 | pld_k4 | clean | 3 | 3 | +0.076 | n/a (<2 sessions) |
| 384 | cka90_k2 | clean | 3 | 3 | -0.086 | n/a (<2 sessions) |
| 1024 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.146 | [0.040, 0.243] |
| 1024 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.034 | [-0.055, -0.014] |
| 1024 | pld_k4 | clean | 3 | 2,3 | +0.155 | n/a (<2 sessions) |
| 1024 | cka90_k2 | clean | 3 | 2,3 | -0.032 | n/a (<2 sessions) |
| 2048 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.070 | [-0.059, 0.177] |
| 2048 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.007 | [-0.040, 0.023] |
| 2048 | pld_k4 | clean | 3 | 2,3 | +0.048 | n/a (<2 sessions) |
| 2048 | cka90_k2 | clean | 3 | 2,3 | -0.004 | n/a (<2 sessions) |

## Paired context-length contrasts

Each contrast pairs the same source documents between ctx128 and the target context. Replicate sessions are resampled independently within each context. Positive decode or E2E deltas mean the method's speedup ratio improved at the longer context; negative exact-rate deltas mean exactness declined.

| Backend | Context contrast | Method | Provenance | ctx128 sessions | Target sessions | Decode speedup delta | 95% interval | E2E speedup delta | Exact-rate delta (pp) |
|---|---:|---|---|---|---|---:|---|---:|---:|
| hf_bnb_nf4 | 384-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.033 | [-0.111, 0.045] | -0.036 | +5.0 |
| hf_bnb_nf4 | 384-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | +0.001 | [-0.024, 0.025] | +0.004 | -10.0 |
| hf_bnb_nf4 | 384-minus-128 | pld_k4 | clean | 3 | 3 | -0.035 | n/a (<2 sessions) | -0.037 | +5.0 |
| hf_bnb_nf4 | 384-minus-128 | cka90_k2 | clean | 3 | 3 | +0.005 | n/a (<2 sessions) | +0.008 | -10.0 |
| hf_bnb_nf4 | 1024-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.125 | [-0.200, -0.051] | -0.124 | -10.0 |
| hf_bnb_nf4 | 1024-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.023 | [-0.045, 0.001] | -0.009 | -15.0 |
| hf_bnb_nf4 | 1024-minus-128 | pld_k4 | clean | 3 | 3 | -0.146 | n/a (<2 sessions) | -0.141 | -10.0 |
| hf_bnb_nf4 | 1024-minus-128 | cka90_k2 | clean | 3 | 3 | -0.027 | n/a (<2 sessions) | -0.013 | -15.0 |
| hf_bnb_nf4 | 2048-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.105 | [-0.215, 0.033] | -0.121 | +15.0 |
| hf_bnb_nf4 | 2048-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.062 | [-0.085, -0.039] | -0.031 | -5.0 |
| hf_bnb_nf4 | 2048-minus-128 | pld_k4 | clean | 3 | 3 | -0.086 | n/a (<2 sessions) | -0.104 | +15.0 |
| hf_bnb_nf4 | 2048-minus-128 | cka90_k2 | clean | 3 | 3 | -0.059 | n/a (<2 sessions) | -0.028 | -5.0 |
| hqq_gemlite_w4a16 | 384-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.029 | [-0.077, 0.133] | +0.023 | -1.7 |
| hqq_gemlite_w4a16 | 384-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.009 | [-0.037, 0.016] | -0.005 | -3.3 |
| hqq_gemlite_w4a16 | 384-minus-128 | pld_k4 | clean | 3 | 3 | +0.004 | n/a (<2 sessions) | -0.001 | +5.0 |
| hqq_gemlite_w4a16 | 384-minus-128 | cka90_k2 | clean | 3 | 3 | -0.017 | n/a (<2 sessions) | -0.013 | -10.0 |
| hqq_gemlite_w4a16 | 1024-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | +0.022 | [-0.089, 0.136] | +0.006 | -11.7 |
| hqq_gemlite_w4a16 | 1024-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | +0.010 | [-0.011, 0.032] | +0.020 | +0.0 |
| hqq_gemlite_w4a16 | 1024-minus-128 | pld_k4 | clean | 3 | 2,3 | -0.029 | n/a (<2 sessions) | -0.043 | -2.5 |
| hqq_gemlite_w4a16 | 1024-minus-128 | cka90_k2 | clean | 3 | 2,3 | +0.005 | n/a (<2 sessions) | +0.015 | -7.5 |
| hqq_gemlite_w4a16 | 2048-minus-128 | pld_k4 | exploratory | 1,2,3 | 1,2,3 | -0.034 | [-0.134, 0.068] | -0.053 | -11.7 |
| hqq_gemlite_w4a16 | 2048-minus-128 | cka90_k2 | exploratory | 1,2,3 | 1,2,3 | -0.001 | [-0.031, 0.024] | +0.020 | -10.0 |
| hqq_gemlite_w4a16 | 2048-minus-128 | pld_k4 | clean | 3 | 2,3 | -0.076 | n/a (<2 sessions) | -0.094 | -7.5 |
| hqq_gemlite_w4a16 | 2048-minus-128 | cka90_k2 | clean | 3 | 2,3 | +0.002 | n/a (<2 sessions) | +0.023 | -15.0 |

## Acceptance versus speedup

Spearman association uses each prompt's mean acceptance rate and internal speedup over the available sessions. Its interval resamples prompts within task and is conditional on those sessions; it is descriptive, not a causal effect.

| Backend | Context | Method | Provenance | Prompts | Spearman rho | 95% prompt interval |
|---|---:|---|---|---:|---:|---|
| hf_bnb_nf4 | 128 | pld_k4 | exploratory | 20 | +0.964 | [0.868, 0.985] |
| hf_bnb_nf4 | 128 | cka90_k2 | exploratory | 20 | +0.883 | [0.766, 0.958] |
| hf_bnb_nf4 | 128 | pld_k4 | clean | 20 | +0.940 | [0.847, 0.971] |
| hf_bnb_nf4 | 128 | cka90_k2 | clean | 20 | +0.805 | [0.572, 0.946] |
| hf_bnb_nf4 | 384 | pld_k4 | exploratory | 20 | +0.893 | [0.702, 0.968] |
| hf_bnb_nf4 | 384 | cka90_k2 | exploratory | 20 | +0.845 | [0.612, 0.950] |
| hf_bnb_nf4 | 384 | pld_k4 | clean | 20 | +0.821 | [0.599, 0.914] |
| hf_bnb_nf4 | 384 | cka90_k2 | clean | 20 | +0.588 | [0.175, 0.885] |
| hf_bnb_nf4 | 1024 | pld_k4 | exploratory | 20 | +0.926 | [0.765, 0.986] |
| hf_bnb_nf4 | 1024 | cka90_k2 | exploratory | 20 | +0.842 | [0.626, 0.944] |
| hf_bnb_nf4 | 1024 | pld_k4 | clean | 20 | +0.877 | [0.652, 0.967] |
| hf_bnb_nf4 | 1024 | cka90_k2 | clean | 20 | +0.778 | [0.522, 0.928] |
| hf_bnb_nf4 | 2048 | pld_k4 | exploratory | 20 | +0.962 | [0.906, 0.989] |
| hf_bnb_nf4 | 2048 | cka90_k2 | exploratory | 20 | +0.860 | [0.601, 0.968] |
| hf_bnb_nf4 | 2048 | pld_k4 | clean | 20 | +0.895 | [0.731, 0.973] |
| hf_bnb_nf4 | 2048 | cka90_k2 | clean | 20 | +0.754 | [0.434, 0.927] |
| hqq_gemlite_w4a16 | 128 | pld_k4 | exploratory | 20 | +0.836 | [0.641, 0.917] |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | exploratory | 20 | +0.916 | [0.801, 0.956] |
| hqq_gemlite_w4a16 | 128 | pld_k4 | clean | 20 | +0.723 | [0.441, 0.864] |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | clean | 20 | +0.545 | [0.235, 0.786] |
| hqq_gemlite_w4a16 | 384 | pld_k4 | exploratory | 20 | +0.915 | [0.738, 0.979] |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | exploratory | 20 | +0.889 | [0.755, 0.948] |
| hqq_gemlite_w4a16 | 384 | pld_k4 | clean | 20 | +0.908 | [0.708, 0.980] |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | clean | 20 | +0.728 | [0.509, 0.867] |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | exploratory | 20 | +0.683 | [0.201, 0.968] |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | exploratory | 20 | +0.733 | [0.494, 0.892] |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | clean | 20 | +0.677 | [0.187, 0.961] |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | clean | 20 | +0.683 | [0.365, 0.889] |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | exploratory | 20 | +0.689 | [0.248, 0.942] |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | exploratory | 20 | +0.659 | [0.282, 0.913] |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | clean | 20 | +0.635 | [0.160, 0.920] |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | clean | 20 | +0.532 | [0.145, 0.820] |

## Cross-check with existing kernel profile

The separate instrumented BNB NF4 profile at context 384/output 32 reports dequantization kernel shares of 8.0% for vanilla, 6.7% for PLD K4, and 4.2% for CKA90/K2, while NF4 GEMM shares are 37.8%, 49.3%, and 47.5%. CKA90/K2 has the lowest dequant share yet remains slower in the paired context matrix. This supports avoiding a dequantization-only explanation; it does not establish what causes the slowdown because the profile is a single instrumented prompt with a different budget/capacity. See [`kernel_profile/report.md`](../kernel_profile/report.md).


## Evidence visible in this rolling snapshot

The clean-provenance point estimates show CKA90/K2 below 1× for both decode and end-to-end speedup in every completed cell. PLD K4 is not uniformly profitable: BNB ctx1024 is below 1×, and many prompt-level runs still slow down even when the cell mean exceeds 1×. Neither method reaches 100% exact sequence match in these clean cells, so the current evidence does not support a lossless claim.

The exploratory matched-source comparison from ctx128 to ctx1024 estimates a PLD speedup change of -0.125 for BNB (95% interval [-0.200, -0.051]) and +0.022 for HQQ (95% interval [-0.089, +0.136]). This pattern is consistent with backend-dependent context sensitivity, but the BNB/HQQ runs include historical provenance limitations and should be treated as a follow-up hypothesis.

## Interpretation limits

This report is a rolling summary: it changes as queued benchmark cells finish. A clean single-session cell supports a reproducible descriptive result but not a session-level uncertainty interval. Bootstrap intervals based on only two or three session clusters remain fragile; collecting more independent sessions would strengthen them. Exploratory cells are retained in `paired_summary.json` and are not mixed into the clean-only table.
