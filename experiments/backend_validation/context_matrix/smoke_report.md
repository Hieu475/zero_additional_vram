# Paired context-length backend smoke (32-token output)

The same five LongBench source documents (one each from narrativeqa, qasper, gov_report, multi_news, qmsum) are represented at 128, 384, 1024, and 2048 input tokens. Each session uses a 32-token full-path warmup and output budget, static capacity `context + 36`, greedy SDPA. This is exploratory, not a performance conclusion: 1 sessions and five prompts per cell.

Execution provenance: the recorded session used commit `7b8a398` with a clean working tree.

| Context | Backend | Cell | Method | Decode TPS mean | Internal speedup vs vanilla | Slower fraction | Exact vs own vanilla | Peak allocated MB |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 128 | hf_bnb_nf4 | complete | vanilla | 36.43 | — | — | 5/5 | 2023 |
| 128 | hf_bnb_nf4 | complete | pld_k4 | 38.33 | 1.050× | 20% | 5/5 | 2025 |
| 128 | hf_bnb_nf4 | complete | cka90_k2 | 28.61 | 0.785× | 100% | 3/5 | 2026 |
| 128 | hqq_gemlite_w4a16 | complete | vanilla | 28.81 | — | — | 5/5 | 2139 |
| 128 | hqq_gemlite_w4a16 | complete | pld_k4 | 27.22 | 0.945× | 60% | 5/5 | 2142 |
| 128 | hqq_gemlite_w4a16 | complete | cka90_k2 | 21.46 | 0.744× | 100% | 5/5 | 2143 |
| 384 | hf_bnb_nf4 | complete | vanilla | 36.88 | — | — | 5/5 | 2107 |
| 384 | hf_bnb_nf4 | complete | pld_k4 | 37.16 | 1.005× | 60% | 5/5 | 2110 |
| 384 | hf_bnb_nf4 | complete | cka90_k2 | 31.67 | 0.861× | 100% | 3/5 | 2111 |
| 384 | hqq_gemlite_w4a16 | complete | vanilla | 29.30 | — | — | 5/5 | 2224 |
| 384 | hqq_gemlite_w4a16 | complete | pld_k4 | 26.90 | 0.915× | 60% | 4/5 | 2228 |
| 384 | hqq_gemlite_w4a16 | complete | cka90_k2 | 23.08 | 0.789× | 100% | 4/5 | 2228 |
| 1024 | hf_bnb_nf4 | complete | vanilla | 33.97 | — | — | 5/5 | 2316 |
| 1024 | hf_bnb_nf4 | complete | pld_k4 | 32.89 | 0.968× | 60% | 3/5 | 2323 |
| 1024 | hf_bnb_nf4 | complete | cka90_k2 | 29.14 | 0.859× | 100% | 3/5 | 2323 |
| 1024 | hqq_gemlite_w4a16 | complete | vanilla | 24.36 | — | — | 5/5 | 2434 |
| 1024 | hqq_gemlite_w4a16 | complete | pld_k4 | 25.35 | 1.043× | 60% | 5/5 | 2440 |
| 1024 | hqq_gemlite_w4a16 | complete | cka90_k2 | 19.80 | 0.814× | 100% | 5/5 | 2441 |
| 2048 | hf_bnb_nf4 | complete | vanilla | 32.57 | — | — | 5/5 | 2663 |
| 2048 | hf_bnb_nf4 | complete | pld_k4 | 31.34 | 0.963× | 60% | 3/5 | 2673 |
| 2048 | hf_bnb_nf4 | complete | cka90_k2 | 27.51 | 0.844× | 100% | 3/5 | 2674 |
| 2048 | hqq_gemlite_w4a16 | complete | vanilla | 26.32 | — | — | 5/5 | 2788 |
| 2048 | hqq_gemlite_w4a16 | complete | pld_k4 | 27.95 | 1.065× | 40% | 5/5 | 2798 |
| 2048 | hqq_gemlite_w4a16 | complete | cka90_k2 | 20.13 | 0.766× | 100% | 5/5 | 2799 |

Cross-backend vanilla exactness by context: 128: 0/5; 384: 0/5; 1024: 0/5; 2048: 0/5

Token exactness is measured against each backend's own vanilla output. Cross-backend output differences reflect stack/quantization fidelity. Do not infer lossless acceleration from speedup alone.
