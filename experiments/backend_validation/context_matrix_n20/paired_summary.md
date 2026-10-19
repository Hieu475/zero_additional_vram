# Output-128 context matrix: paired preliminary summary

Only structurally complete cells are included (20 prompts × 3 methods, successful warmups, no failed method rows, expected KV capacity). Incomplete files are ignored. Clean-provenance results are shown separately from exploratory runs.

- Complete cells: **20/24**.
- Cells with fully clean working-tree and source-tree provenance: **7**.
- TPS definition: `(generated tokens - 1) / decode time`; speedup is paired against vanilla for the same backend, prompt, and session.
- 95% intervals resample complete sessions and prompt IDs as paired units; they are omitted with fewer than two sessions.
- Cross-backend comparison is HQQ internal speedup minus BNB internal speedup. Quantized weights differ, so this is an inference-stack comparison, not a kernel-only effect.

## Clean-provenance cells

| Backend | Context | Method | Session IDs | Decode speedup | 95% interval | Decode slowdown | E2E speedup | Exact vs own vanilla | First mismatch (median zero-based index) | Mean decode TPS | Peak allocated (MB) |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| hf_bnb_nf4 | 128 | pld_k4 | 3 | 1.126× | n/a (<2 sessions) | 15.0% | 1.122× | 6/20 | 67 | 50.37 | 2030 |
| hf_bnb_nf4 | 128 | cka90_k2 | 3 | 0.885× | n/a (<2 sessions) | 95.0% | 0.888× | 5/20 | 66 | 39.55 | 2030 |
| hf_bnb_nf4 | 384 | pld_k4 | 3 | 1.092× | n/a (<2 sessions) | 40.0% | 1.085× | 7/20 | 49 | 47.18 | 2115 |
| hf_bnb_nf4 | 384 | cka90_k2 | 3 | 0.890× | n/a (<2 sessions) | 100.0% | 0.896× | 3/20 | 56 | 38.52 | 2115 |
| hf_bnb_nf4 | 1024 | pld_k4 | 3 | 0.980× | n/a (<2 sessions) | 65.0% | 0.981× | 4/20 | 59 | 44.06 | 2327 |
| hf_bnb_nf4 | 1024 | cka90_k2 | 3 | 0.858× | n/a (<2 sessions) | 100.0% | 0.875× | 2/20 | 26 | 38.55 | 2327 |
| hf_bnb_nf4 | 2048 | pld_k4 | 3 | 1.040× | n/a (<2 sessions) | 60.0% | 1.018× | 9/20 | 49 | 43.87 | 2676 |
| hf_bnb_nf4 | 2048 | cka90_k2 | 3 | 0.827× | n/a (<2 sessions) | 100.0% | 0.860× | 4/20 | 54 | 34.75 | 2677 |
| hqq_gemlite_w4a16 | 128 | pld_k4 | 3 | 1.164× | n/a (<2 sessions) | 10.0% | 1.161× | 15/20 | 65 | 30.38 | 2146 |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | 3 | 0.821× | n/a (<2 sessions) | 100.0% | 0.823× | 17/20 | 90 | 21.42 | 2147 |
| hqq_gemlite_w4a16 | 384 | pld_k4 | 3 | 1.168× | n/a (<2 sessions) | 30.0% | 1.160× | 16/20 | 74 | 31.95 | 2232 |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | 3 | 0.804× | n/a (<2 sessions) | 100.0% | 0.809× | 15/20 | 98 | 22.03 | 2232 |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | 2 | 1.134× | n/a (<2 sessions) | 30.0% | 1.117× | 17/20 | 67 | 31.22 | 2457 |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | 2 | 0.826× | n/a (<2 sessions) | 100.0% | 0.838× | 15/20 | 19 | 22.74 | 2445 |

## Complete cells including exploratory provenance

These results include complete cells with historical dirty or missing provenance fields. Use them as sensitivity analysis, not as the clean-only result.

| Backend | Context | Method | Session count | Decode speedup | 95% interval | Decode slowdown | E2E speedup | Exact vs own vanilla | First mismatch (median zero-based index) | Mean decode TPS | Peak allocated (MB) |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| hf_bnb_nf4 | 128 | pld_k4 | 3 | 1.136× | [1.078, 1.196] | 16.7% | 1.132× | 18/60 | 67 | 48.82 | 2030 |
| hf_bnb_nf4 | 128 | cka90_k2 | 3 | 0.885× | [0.867, 0.906] | 98.3% | 0.888× | 15/60 | 66 | 38.04 | 2030 |
| hf_bnb_nf4 | 384 | pld_k4 | 3 | 1.103× | [1.041, 1.173] | 26.7% | 1.096× | 21/60 | 49 | 45.93 | 2115 |
| hf_bnb_nf4 | 384 | cka90_k2 | 3 | 0.886× | [0.868, 0.904] | 100.0% | 0.891× | 9/60 | 56 | 36.95 | 2115 |
| hf_bnb_nf4 | 1024 | pld_k4 | 3 | 1.010× | [0.949, 1.079] | 55.0% | 1.007× | 12/60 | 59 | 44.01 | 2327 |
| hf_bnb_nf4 | 1024 | cka90_k2 | 3 | 0.863× | [0.842, 0.882] | 100.0% | 0.879× | 6/60 | 26 | 37.60 | 2327 |
| hf_bnb_nf4 | 2048 | pld_k4 | 3 | 1.031× | [0.925, 1.189] | 63.3% | 1.011× | 27/60 | 49 | 45.06 | 2676 |
| hf_bnb_nf4 | 2048 | cka90_k2 | 3 | 0.823× | [0.807, 0.840] | 100.0% | 0.857× | 12/60 | 54 | 35.98 | 2677 |
| hqq_gemlite_w4a16 | 128 | pld_k4 | 3 | 1.135× | [1.037, 1.218] | 20.0% | 1.132× | 50/60 | 65 | 30.35 | 2155 |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | 3 | 0.818× | [0.801, 0.836] | 100.0% | 0.820× | 47/60 | 63 | 21.90 | 2147 |
| hqq_gemlite_w4a16 | 384 | pld_k4 | 2 | 1.167× | [1.077, 1.270] | 30.0% | 1.159× | 32/40 | 78 | 31.65 | 2232 |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | 2 | 0.809× | [0.783, 0.835] | 100.0% | 0.815× | 30/40 | 100 | 21.94 | 2232 |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | 2 | 1.167× | [1.056, 1.272] | 30.0% | 1.147× | 31/40 | 67 | 31.55 | 2451 |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | 2 | 0.830× | [0.810, 0.846] | 100.0% | 0.842× | 31/40 | 19 | 22.45 | 2445 |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | 1 | 1.129× | n/a (<2 sessions) | 15.0% | 1.104× | 16/20 | 79 | 30.68 | 2799 |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | 1 | 0.805× | n/a (<2 sessions) | 100.0% | 0.829× | 13/20 | 47 | 21.84 | 2812 |

## Paired backend deltas

Positive values mean HQQ gains more internal speedup than BNB. The matched rows use the same session and prompt; they do not compare cross-backend token identity.

| Context | Method | Provenance | Common session IDs | Mean HQQ−BNB speedup delta | 95% interval |
|---:|---|---|---|---:|---|
| 128 | pld_k4 | exploratory | 1,2,3 | -0.001 | [-0.110, 0.092] |
| 128 | cka90_k2 | exploratory | 1,2,3 | -0.067 | [-0.091, -0.042] |
| 128 | pld_k4 | clean | 3 | 0.037 | n/a (<2 sessions) |
| 128 | cka90_k2 | clean | 3 | -0.065 | n/a (<2 sessions) |
| 384 | pld_k4 | exploratory | 1,3 | 0.067 | [-0.019, 0.152] |
| 384 | cka90_k2 | exploratory | 1,3 | -0.074 | [-0.108, -0.045] |
| 384 | pld_k4 | clean | 3 | 0.076 | n/a (<2 sessions) |
| 384 | cka90_k2 | clean | 3 | -0.086 | n/a (<2 sessions) |
| 1024 | pld_k4 | exploratory | 1,2 | 0.142 | [0.010, 0.261] |
| 1024 | cka90_k2 | exploratory | 1,2 | -0.035 | [-0.064, -0.007] |
| 2048 | pld_k4 | exploratory | 1 | 0.116 | n/a (<2 sessions) |
| 2048 | cka90_k2 | exploratory | 1 | -0.010 | n/a (<2 sessions) |

## Interpretation limits

This report is a rolling summary: it changes as queued benchmark cells finish. A clean single-session cell supports a reproducible descriptive result but not a session-level uncertainty interval. Exploratory cells are retained in `paired_summary.json` and are not mixed into the clean-only table.
