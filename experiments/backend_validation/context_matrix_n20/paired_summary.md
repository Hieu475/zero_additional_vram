# Output-128 context matrix: paired preliminary summary

Only structurally complete cells are included (20 prompts × 3 methods, successful warmups, no failed method rows, expected KV capacity). Incomplete files are ignored. Clean-provenance results are shown separately from exploratory runs.

- Complete cells: **20/24**.
- Cells with fully clean working-tree and source-tree provenance: **7**.
- TPS definition: `(generated tokens - 1) / decode time`; speedup is paired against vanilla for the same backend, prompt, and session.
- 95% intervals resample complete sessions and prompt IDs as paired units; they are omitted with fewer than two sessions.
- Cross-backend comparison is HQQ internal speedup minus BNB internal speedup. Quantized weights differ, so this is an inference-stack comparison, not a kernel-only effect.

## Clean-provenance cells

| Backend | Context | Method | Session IDs | Mean speedup | 95% interval | Slowdown | Exact vs own vanilla | Mean decode TPS |
|---|---:|---|---|---:|---|---:|---:|---:|
| hf_bnb_nf4 | 128 | pld_k4 | 3 | 1.126× | n/a (<2 sessions) | 15.0% | 6/20 | 50.37 |
| hf_bnb_nf4 | 128 | cka90_k2 | 3 | 0.885× | n/a (<2 sessions) | 95.0% | 5/20 | 39.55 |
| hf_bnb_nf4 | 384 | pld_k4 | 3 | 1.092× | n/a (<2 sessions) | 40.0% | 7/20 | 47.18 |
| hf_bnb_nf4 | 384 | cka90_k2 | 3 | 0.890× | n/a (<2 sessions) | 100.0% | 3/20 | 38.52 |
| hf_bnb_nf4 | 1024 | pld_k4 | 3 | 0.980× | n/a (<2 sessions) | 65.0% | 4/20 | 44.06 |
| hf_bnb_nf4 | 1024 | cka90_k2 | 3 | 0.858× | n/a (<2 sessions) | 100.0% | 2/20 | 38.55 |
| hf_bnb_nf4 | 2048 | pld_k4 | 3 | 1.040× | n/a (<2 sessions) | 60.0% | 9/20 | 43.87 |
| hf_bnb_nf4 | 2048 | cka90_k2 | 3 | 0.827× | n/a (<2 sessions) | 100.0% | 4/20 | 34.75 |
| hqq_gemlite_w4a16 | 128 | pld_k4 | 3 | 1.164× | n/a (<2 sessions) | 10.0% | 15/20 | 30.38 |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | 3 | 0.821× | n/a (<2 sessions) | 100.0% | 17/20 | 21.42 |
| hqq_gemlite_w4a16 | 384 | pld_k4 | 3 | 1.168× | n/a (<2 sessions) | 30.0% | 16/20 | 31.95 |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | 3 | 0.804× | n/a (<2 sessions) | 100.0% | 15/20 | 22.03 |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | 2 | 1.134× | n/a (<2 sessions) | 30.0% | 17/20 | 31.22 |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | 2 | 0.826× | n/a (<2 sessions) | 100.0% | 15/20 | 22.74 |

## Complete cells including exploratory provenance

These results include complete cells with historical dirty or missing provenance fields. Use them as sensitivity analysis, not as the clean-only result.

| Backend | Context | Method | Sessions | Mean speedup | 95% interval | Slowdown | Exact vs own vanilla | Mean decode TPS |
|---|---:|---|---:|---:|---|---:|---:|---:|
| hf_bnb_nf4 | 128 | pld_k4 | 3 | 1.136× | [1.078, 1.196] | 16.7% | 18/60 | 48.82 |
| hf_bnb_nf4 | 128 | cka90_k2 | 3 | 0.885× | [0.867, 0.906] | 98.3% | 15/60 | 38.04 |
| hf_bnb_nf4 | 384 | pld_k4 | 3 | 1.103× | [1.041, 1.173] | 26.7% | 21/60 | 45.93 |
| hf_bnb_nf4 | 384 | cka90_k2 | 3 | 0.886× | [0.868, 0.904] | 100.0% | 9/60 | 36.95 |
| hf_bnb_nf4 | 1024 | pld_k4 | 3 | 1.010× | [0.949, 1.079] | 55.0% | 12/60 | 44.01 |
| hf_bnb_nf4 | 1024 | cka90_k2 | 3 | 0.863× | [0.842, 0.882] | 100.0% | 6/60 | 37.60 |
| hf_bnb_nf4 | 2048 | pld_k4 | 3 | 1.031× | [0.925, 1.189] | 63.3% | 27/60 | 45.06 |
| hf_bnb_nf4 | 2048 | cka90_k2 | 3 | 0.823× | [0.807, 0.840] | 100.0% | 12/60 | 35.98 |
| hqq_gemlite_w4a16 | 128 | pld_k4 | 3 | 1.135× | [1.037, 1.218] | 20.0% | 50/60 | 30.35 |
| hqq_gemlite_w4a16 | 128 | cka90_k2 | 3 | 0.818× | [0.801, 0.836] | 100.0% | 47/60 | 21.90 |
| hqq_gemlite_w4a16 | 384 | pld_k4 | 2 | 1.167× | [1.077, 1.270] | 30.0% | 32/40 | 31.65 |
| hqq_gemlite_w4a16 | 384 | cka90_k2 | 2 | 0.809× | [0.783, 0.835] | 100.0% | 30/40 | 21.94 |
| hqq_gemlite_w4a16 | 1024 | pld_k4 | 2 | 1.167× | [1.056, 1.272] | 30.0% | 31/40 | 31.55 |
| hqq_gemlite_w4a16 | 1024 | cka90_k2 | 2 | 0.830× | [0.810, 0.846] | 100.0% | 31/40 | 22.45 |
| hqq_gemlite_w4a16 | 2048 | pld_k4 | 1 | 1.129× | n/a (<2 sessions) | 15.0% | 16/20 | 30.68 |
| hqq_gemlite_w4a16 | 2048 | cka90_k2 | 1 | 0.805× | n/a (<2 sessions) | 100.0% | 13/20 | 21.84 |

## Paired clean backend deltas

Positive values mean HQQ gains more internal speedup than BNB; this does not compare cross-backend token identity.

| Context | Method | Common clean sessions | Mean HQQ−BNB speedup delta | 95% interval |
|---:|---|---|---:|---|
| 128 | pld_k4 | 3 | 0.037 | n/a (<2 sessions) |
| 128 | cka90_k2 | 3 | -0.065 | n/a (<2 sessions) |
| 384 | pld_k4 | 3 | 0.076 | n/a (<2 sessions) |
| 384 | cka90_k2 | 3 | -0.086 | n/a (<2 sessions) |

## Interpretation limits

This report is a rolling summary: it changes as queued benchmark cells finish. A clean single-session cell supports a reproducible descriptive result but not a session-level uncertainty interval. Exploratory cells are retained in `paired_summary.json` and are not mixed into the clean-only table.
