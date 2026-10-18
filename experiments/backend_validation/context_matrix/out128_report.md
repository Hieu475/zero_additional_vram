# Paired context-length backend evaluation (128-token output)

The same five LongBench source documents (one each from narrativeqa, qasper, gov_report, multi_news, qmsum) are represented at 128, 384, 1024, and 2048 input tokens. Each session uses a 128-token full-path warmup and output budget, static capacity `context + 132`, greedy SDPA. This is exploratory, not a performance conclusion: 3 sessions and five prompts per cell.

Execution provenance: session 1 used commit `7b8a398` with a clean working tree. Output-32 sessions 2 and 3 used commit `424a083`; output-128 sessions 2 and 3 used commits `ad0e1d2` and `b24e505`. Runtime metadata marks sessions 2 and 3 dirty, so this is exploratory and does not establish clean-tree reproducibility.

| Context | Backend | Cell | Method | Decode TPS mean | Internal speedup vs vanilla | Slower fraction | Exact vs own vanilla | Peak allocated MB |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 128 | hf_bnb_nf4 | complete | vanilla | 42.10 | — | — | 15/15 | 2026 |
| 128 | hf_bnb_nf4 | complete | pld_k4 | 43.35 | 1.030× | 67% | 9/15 | 2030 |
| 128 | hf_bnb_nf4 | complete | cka90_k2 | 36.20 | 0.861× | 100% | 3/15 | 2030 |
| 128 | hqq_gemlite_w4a16 | complete | vanilla | 25.65 | — | — | 15/15 | 2143 |
| 128 | hqq_gemlite_w4a16 | complete | pld_k4 | 27.29 | 1.066× | 33% | 11/15 | 2146 |
| 128 | hqq_gemlite_w4a16 | complete | cka90_k2 | 19.63 | 0.769× | 100% | 13/15 | 2164 |
| 384 | hf_bnb_nf4 | complete | vanilla | 41.90 | — | — | 15/15 | 2110 |
| 384 | hf_bnb_nf4 | complete | pld_k4 | 47.05 | 1.124× | 27% | 6/15 | 2115 |
| 384 | hf_bnb_nf4 | complete | cka90_k2 | 36.37 | 0.869× | 100% | 6/15 | 2115 |
| 384 | hqq_gemlite_w4a16 | complete | vanilla | 26.01 | — | — | 15/15 | 2227 |
| 384 | hqq_gemlite_w4a16 | complete | pld_k4 | 31.50 | 1.212× | 0% | 12/15 | 2232 |
| 384 | hqq_gemlite_w4a16 | complete | cka90_k2 | 21.89 | 0.842× | 100% | 11/15 | 2232 |
| 1024 | hf_bnb_nf4 | complete | vanilla | 39.67 | — | — | 15/15 | 2320 |
| 1024 | hf_bnb_nf4 | complete | pld_k4 | 45.87 | 1.158× | 40% | 6/15 | 2327 |
| 1024 | hf_bnb_nf4 | complete | cka90_k2 | 34.50 | 0.871× | 100% | 9/15 | 2327 |
| 1024 | hqq_gemlite_w4a16 | complete | vanilla | 25.62 | — | — | 15/15 | 2437 |
| 1024 | hqq_gemlite_w4a16 | complete | pld_k4 | 33.41 | 1.313× | 0% | 12/15 | 2445 |
| 1024 | hqq_gemlite_w4a16 | complete | cka90_k2 | 21.38 | 0.839× | 100% | 13/15 | 2445 |
| 2048 | hf_bnb_nf4 | complete | vanilla | 41.00 | — | — | 15/15 | 2665 |
| 2048 | hf_bnb_nf4 | complete | pld_k4 | 40.15 | 0.979× | 47% | 3/15 | 2676 |
| 2048 | hf_bnb_nf4 | complete | cka90_k2 | 34.05 | 0.831× | 100% | 6/15 | 2677 |
| 2048 | hqq_gemlite_w4a16 | complete | vanilla | 26.92 | — | — | 15/15 | 2788 |
| 2048 | hqq_gemlite_w4a16 | complete | pld_k4 | 30.86 | 1.149× | 7% | 11/15 | 2799 |
| 2048 | hqq_gemlite_w4a16 | complete | cka90_k2 | 22.11 | 0.821× | 100% | 13/15 | 2799 |

Cross-backend vanilla exactness by context: 128: 0/15; 384: 0/15; 1024: 0/15; 2048: 0/15

Token exactness is measured against each backend's own vanilla output. Cross-backend output differences reflect stack/quantization fidelity. Do not infer lossless acceleration from speedup alone.
