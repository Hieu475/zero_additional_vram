# Paired context-length backend smoke (32-token output)

The same five LongBench source documents (one each from narrativeqa, qasper, gov_report, multi_news, qmsum) are represented at 128, 384, 1024, and 2048 input tokens. Each session uses a 32-token full-path warmup and output budget, static capacity `context + 36`, greedy SDPA. This is exploratory, not a performance conclusion: 3 sessions and five prompts per cell.

Execution provenance: session 1 used commit `7b8a398` with a clean working tree. Output-32 sessions 2 and 3 used commit `424a083`; output-128 sessions 2 and 3 used commits `ad0e1d2` and `b24e505`. Runtime metadata marks sessions 2 and 3 dirty, so this is exploratory and does not establish clean-tree reproducibility.

| Context | Backend | Cell | Method | Decode TPS mean | Internal speedup vs vanilla | Slower fraction | Exact vs own vanilla | Peak allocated MB |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 128 | hf_bnb_nf4 | complete | vanilla | 39.78 | — | — | 15/15 | 2023 |
| 128 | hf_bnb_nf4 | complete | pld_k4 | 42.71 | 1.073× | 33% | 15/15 | 2025 |
| 128 | hf_bnb_nf4 | complete | cka90_k2 | 32.12 | 0.807× | 100% | 9/15 | 2026 |
| 128 | hqq_gemlite_w4a16 | complete | vanilla | 26.51 | — | — | 15/15 | 2139 |
| 128 | hqq_gemlite_w4a16 | complete | pld_k4 | 25.06 | 0.946× | 60% | 14/15 | 2142 |
| 128 | hqq_gemlite_w4a16 | complete | cka90_k2 | 20.11 | 0.760× | 100% | 14/15 | 2143 |
| 384 | hf_bnb_nf4 | complete | vanilla | 40.55 | — | — | 15/15 | 2107 |
| 384 | hf_bnb_nf4 | complete | pld_k4 | 41.39 | 1.020× | 47% | 15/15 | 2110 |
| 384 | hf_bnb_nf4 | complete | cka90_k2 | 34.76 | 0.858× | 100% | 9/15 | 2111 |
| 384 | hqq_gemlite_w4a16 | complete | vanilla | 26.76 | — | — | 15/15 | 2224 |
| 384 | hqq_gemlite_w4a16 | complete | pld_k4 | 25.06 | 0.937× | 60% | 11/15 | 2228 |
| 384 | hqq_gemlite_w4a16 | complete | cka90_k2 | 21.11 | 0.789× | 100% | 12/15 | 2228 |
| 1024 | hf_bnb_nf4 | complete | vanilla | 40.16 | — | — | 15/15 | 2316 |
| 1024 | hf_bnb_nf4 | complete | pld_k4 | 38.43 | 0.959× | 67% | 9/15 | 2323 |
| 1024 | hf_bnb_nf4 | complete | cka90_k2 | 33.70 | 0.842× | 100% | 9/15 | 2323 |
| 1024 | hqq_gemlite_w4a16 | complete | vanilla | 24.88 | — | — | 15/15 | 2434 |
| 1024 | hqq_gemlite_w4a16 | complete | pld_k4 | 25.32 | 1.019× | 60% | 15/15 | 2440 |
| 1024 | hqq_gemlite_w4a16 | complete | cka90_k2 | 20.41 | 0.822× | 100% | 15/15 | 2441 |
| 2048 | hf_bnb_nf4 | complete | vanilla | 37.37 | — | — | 15/15 | 2663 |
| 2048 | hf_bnb_nf4 | complete | pld_k4 | 36.48 | 0.978× | 60% | 9/15 | 2673 |
| 2048 | hf_bnb_nf4 | complete | cka90_k2 | 31.55 | 0.845× | 100% | 9/15 | 2674 |
| 2048 | hqq_gemlite_w4a16 | complete | vanilla | 26.00 | — | — | 15/15 | 2788 |
| 2048 | hqq_gemlite_w4a16 | complete | pld_k4 | 26.91 | 1.037× | 40% | 15/15 | 2798 |
| 2048 | hqq_gemlite_w4a16 | complete | cka90_k2 | 20.01 | 0.771× | 100% | 15/15 | 2799 |

Cross-backend vanilla exactness by context: 128: 0/15; 384: 0/15; 1024: 0/15; 2048: 0/15

Token exactness is measured against each backend's own vanilla output. Cross-backend output differences reflect stack/quantization fidelity. Do not infer lossless acceleration from speedup alone.
