# Paired context-length backend evaluation (128-token output)

The same five LongBench source documents (one each from narrativeqa, qasper, gov_report, multi_news, qmsum) are represented at 128, 384, 1024, and 2048 input tokens. Each session uses a 128-token full-path warmup and output budget, static capacity `context + 132`, greedy SDPA. This is exploratory, not a performance conclusion: 2 sessions and five prompts per cell.

Execution provenance: session 1 used commit `7b8a398` with a clean working tree. Session 2 used commit `ad0e1d2`; runtime metadata marks its working tree dirty, so this two-session result is exploratory and does not establish clean-tree reproducibility.

| Context | Backend | Cell | Method | Decode TPS mean | Internal speedup vs vanilla | Slower fraction | Exact vs own vanilla | Peak allocated MB |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 128 | hf_bnb_nf4 | complete | vanilla | 42.18 | — | — | 10/10 | 2026 |
| 128 | hf_bnb_nf4 | complete | pld_k4 | 43.43 | 1.030× | 70% | 6/10 | 2030 |
| 128 | hf_bnb_nf4 | complete | cka90_k2 | 36.06 | 0.856× | 100% | 2/10 | 2030 |
| 128 | hqq_gemlite_w4a16 | complete | vanilla | 25.49 | — | — | 10/10 | 2143 |
| 128 | hqq_gemlite_w4a16 | complete | pld_k4 | 27.22 | 1.071× | 30% | 7/10 | 2146 |
| 128 | hqq_gemlite_w4a16 | complete | cka90_k2 | 18.81 | 0.744× | 100% | 8/10 | 2173 |
| 384 | hf_bnb_nf4 | complete | vanilla | 42.48 | — | — | 10/10 | 2110 |
| 384 | hf_bnb_nf4 | complete | pld_k4 | 48.02 | 1.131× | 20% | 4/10 | 2115 |
| 384 | hf_bnb_nf4 | complete | cka90_k2 | 36.91 | 0.870× | 100% | 4/10 | 2115 |
| 384 | hqq_gemlite_w4a16 | complete | vanilla | 25.62 | — | — | 10/10 | 2227 |
| 384 | hqq_gemlite_w4a16 | complete | pld_k4 | 31.13 | 1.215× | 0% | 8/10 | 2232 |
| 384 | hqq_gemlite_w4a16 | complete | cka90_k2 | 21.52 | 0.841× | 100% | 7/10 | 2232 |
| 1024 | hf_bnb_nf4 | complete | vanilla | 39.71 | — | — | 10/10 | 2320 |
| 1024 | hf_bnb_nf4 | complete | pld_k4 | 46.24 | 1.165× | 30% | 4/10 | 2327 |
| 1024 | hf_bnb_nf4 | complete | cka90_k2 | 34.74 | 0.876× | 100% | 6/10 | 2327 |
| 1024 | hqq_gemlite_w4a16 | complete | vanilla | 24.63 | — | — | 10/10 | 2437 |
| 1024 | hqq_gemlite_w4a16 | complete | pld_k4 | 32.55 | 1.335× | 0% | 8/10 | 2445 |
| 1024 | hqq_gemlite_w4a16 | complete | cka90_k2 | 20.75 | 0.847× | 100% | 8/10 | 2445 |
| 2048 | hf_bnb_nf4 | complete | vanilla | 40.83 | — | — | 10/10 | 2665 |
| 2048 | hf_bnb_nf4 | complete | pld_k4 | 40.09 | 0.982× | 40% | 2/10 | 2676 |
| 2048 | hf_bnb_nf4 | complete | cka90_k2 | 34.27 | 0.839× | 100% | 4/10 | 2677 |
| 2048 | hqq_gemlite_w4a16 | complete | vanilla | 26.45 | — | — | 10/10 | 2788 |
| 2048 | hqq_gemlite_w4a16 | complete | pld_k4 | 30.51 | 1.156× | 10% | 8/10 | 2799 |
| 2048 | hqq_gemlite_w4a16 | complete | cka90_k2 | 21.65 | 0.818× | 100% | 9/10 | 2799 |

Cross-backend vanilla exactness by context: 128: 0/10; 384: 0/10; 1024: 0/10; 2048: 0/10

Token exactness is measured against each backend's own vanilla output. Cross-backend output differences reflect stack/quantization fidelity. Do not infer lossless acceleration from speedup alone.
