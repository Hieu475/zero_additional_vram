# Paired context-length backend smoke (32-token output)

The same five LongBench source documents (one each from narrativeqa, qasper, gov_report, multi_news, qmsum) are represented at 128, 384, 1024, and 2048 input tokens. Each session uses a 32-token full-path warmup and output budget, static capacity `context + 36`, greedy SDPA. This is exploratory, not a performance conclusion: 3 sessions and five prompts per cell.

Execution provenance: session 1 used commit `7b8a398` with a clean working tree. Output-32 sessions 2 and 3 used commit `424a083`; output-128 sessions 2 and 3 used commits `ad0e1d2` and `b24e505`. Runtime metadata marks sessions 2 and 3 dirty, so this is exploratory and does not establish clean-tree reproducibility.

| Context | Backend | Cell | Method | Decode TPS mean | Paired speedup | 95% prompt-bootstrap CI | Slower prompts | Exact vs own vanilla | Exact in all sessions | Peak allocated MB |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | hf_bnb_nf4 | complete | vanilla | 39.78 | — | — | — | 15/15 | — | 2023 |
| 128 | hf_bnb_nf4 | complete | pld_k4 | 42.71 | 1.073× | [0.965, 1.197] | 40% | 15/15 obs; 5/5 all runs | 5/5 | 2025 |
| 128 | hf_bnb_nf4 | complete | cka90_k2 | 32.12 | 0.807× | [0.755, 0.858] | 100% | 9/15 obs; 3/5 all runs | 3/5 | 2026 |
| 128 | hqq_gemlite_w4a16 | complete | vanilla | 26.51 | — | — | — | 15/15 | — | 2139 |
| 128 | hqq_gemlite_w4a16 | complete | pld_k4 | 25.06 | 0.946× | [0.860, 1.031] | 60% | 14/15 obs; 4/5 all runs | 4/5 | 2142 |
| 128 | hqq_gemlite_w4a16 | complete | cka90_k2 | 20.11 | 0.760× | [0.695, 0.818] | 100% | 14/15 obs; 4/5 all runs | 4/5 | 2143 |
| 384 | hf_bnb_nf4 | complete | vanilla | 40.55 | — | — | — | 15/15 | — | 2107 |
| 384 | hf_bnb_nf4 | complete | pld_k4 | 41.39 | 1.020× | [0.898, 1.172] | 40% | 15/15 obs; 5/5 all runs | 5/5 | 2110 |
| 384 | hf_bnb_nf4 | complete | cka90_k2 | 34.76 | 0.858× | [0.833, 0.878] | 100% | 9/15 obs; 3/5 all runs | 3/5 | 2111 |
| 384 | hqq_gemlite_w4a16 | complete | vanilla | 26.76 | — | — | — | 15/15 | — | 2224 |
| 384 | hqq_gemlite_w4a16 | complete | pld_k4 | 25.06 | 0.937× | [0.838, 1.035] | 60% | 11/15 obs; 3/5 all runs | 3/5 | 2228 |
| 384 | hqq_gemlite_w4a16 | complete | cka90_k2 | 21.11 | 0.789× | [0.737, 0.817] | 100% | 12/15 obs; 3/5 all runs | 3/5 | 2228 |
| 1024 | hf_bnb_nf4 | complete | vanilla | 40.16 | — | — | — | 15/15 | — | 2316 |
| 1024 | hf_bnb_nf4 | complete | pld_k4 | 38.43 | 0.959× | [0.880, 1.050] | 60% | 9/15 obs; 3/5 all runs | 3/5 | 2323 |
| 1024 | hf_bnb_nf4 | complete | cka90_k2 | 33.70 | 0.842× | [0.819, 0.866] | 100% | 9/15 obs; 3/5 all runs | 3/5 | 2323 |
| 1024 | hqq_gemlite_w4a16 | complete | vanilla | 24.88 | — | — | — | 15/15 | — | 2434 |
| 1024 | hqq_gemlite_w4a16 | complete | pld_k4 | 25.32 | 1.019× | [0.900, 1.158] | 60% | 15/15 obs; 5/5 all runs | 5/5 | 2440 |
| 1024 | hqq_gemlite_w4a16 | complete | cka90_k2 | 20.41 | 0.822× | [0.783, 0.850] | 100% | 15/15 obs; 5/5 all runs | 5/5 | 2441 |
| 2048 | hf_bnb_nf4 | complete | vanilla | 37.37 | — | — | — | 15/15 | — | 2663 |
| 2048 | hf_bnb_nf4 | complete | pld_k4 | 36.48 | 0.978× | [0.892, 1.061] | 60% | 9/15 obs; 3/5 all runs | 3/5 | 2673 |
| 2048 | hf_bnb_nf4 | complete | cka90_k2 | 31.55 | 0.845× | [0.811, 0.878] | 100% | 9/15 obs; 3/5 all runs | 3/5 | 2674 |
| 2048 | hqq_gemlite_w4a16 | complete | vanilla | 26.00 | — | — | — | 15/15 | — | 2788 |
| 2048 | hqq_gemlite_w4a16 | complete | pld_k4 | 26.91 | 1.037× | [0.921, 1.152] | 40% | 15/15 obs; 5/5 all runs | 5/5 | 2798 |
| 2048 | hqq_gemlite_w4a16 | complete | cka90_k2 | 20.01 | 0.771× | [0.729, 0.812] | 100% | 15/15 obs; 5/5 all runs | 5/5 | 2799 |

Cross-backend vanilla exactness by context: 128: 0/15; 384: 0/15; 1024: 0/15; 2048: 0/15

Speedup means and 95% percentile bootstrap intervals resample the five source prompts (10,000 replicates), averaging repeated sessions within each prompt first. The interval is exploratory because only five independent source documents are available per context. ‘Exact vs own vanilla’ counts prompt×session outcomes; ‘Exact in all sessions’ counts prompts exact in every session. Cross-backend vanilla differences reflect stack/quantization fidelity. Do not infer lossless acceleration from speedup alone.
