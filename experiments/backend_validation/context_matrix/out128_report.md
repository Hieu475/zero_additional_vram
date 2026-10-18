# Paired context-length backend evaluation (128-token output)

The same five LongBench source documents (one each from narrativeqa, qasper, gov_report, multi_news, qmsum) are represented at 128, 384, 1024, and 2048 input tokens. Each cell is one process/session with a 128-token full-path warmup and output budget, static capacity `context + 132`, greedy SDPA. This is exploratory, not a performance conclusion: one session and five prompts per cell.

| Context | Backend | Cell | Method | Decode TPS mean | Internal speedup vs vanilla | Slower fraction | Exact vs own vanilla | Peak allocated MB |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 128 | hf_bnb_nf4 | complete | vanilla | 42.86 | — | — | 5/5 | 2026 |
| 128 | hf_bnb_nf4 | complete | pld_k4 | 44.45 | 1.036× | 60% | 3/5 | 2030 |
| 128 | hf_bnb_nf4 | complete | cka90_k2 | 36.32 | 0.848× | 100% | 1/5 | 2030 |
| 128 | hqq_gemlite_w4a16 | complete | vanilla | 25.79 | — | — | 5/5 | 2143 |
| 128 | hqq_gemlite_w4a16 | complete | pld_k4 | 27.27 | 1.060× | 20% | 3/5 | 2146 |
| 128 | hqq_gemlite_w4a16 | complete | cka90_k2 | 17.89 | 0.703× | 100% | 4/5 | 2200 |
| 384 | hf_bnb_nf4 | complete | vanilla | 43.22 | — | — | 5/5 | 2110 |
| 384 | hf_bnb_nf4 | complete | pld_k4 | 49.87 | 1.159× | 20% | 2/5 | 2115 |
| 384 | hf_bnb_nf4 | complete | cka90_k2 | 37.84 | 0.878× | 100% | 2/5 | 2115 |
| 384 | hqq_gemlite_w4a16 | complete | vanilla | 26.04 | — | — | 5/5 | 2227 |
| 384 | hqq_gemlite_w4a16 | complete | pld_k4 | 31.94 | 1.227× | 0% | 3/5 | 2232 |
| 384 | hqq_gemlite_w4a16 | complete | cka90_k2 | 22.00 | 0.845× | 100% | 4/5 | 2232 |
| 1024 | hf_bnb_nf4 | complete | vanilla | 38.90 | — | — | 5/5 | 2320 |
| 1024 | hf_bnb_nf4 | complete | pld_k4 | 47.02 | 1.205× | 20% | 2/5 | 2327 |
| 1024 | hf_bnb_nf4 | complete | cka90_k2 | 34.76 | 0.895× | 100% | 3/5 | 2327 |
| 1024 | hqq_gemlite_w4a16 | complete | vanilla | 23.86 | — | — | 5/5 | 2437 |
| 1024 | hqq_gemlite_w4a16 | complete | pld_k4 | 32.01 | 1.363× | 0% | 4/5 | 2445 |
| 1024 | hqq_gemlite_w4a16 | complete | cka90_k2 | 20.03 | 0.848× | 100% | 4/5 | 2445 |
| 2048 | hf_bnb_nf4 | complete | vanilla | 41.82 | — | — | 5/5 | 2665 |
| 2048 | hf_bnb_nf4 | complete | pld_k4 | 41.16 | 0.987× | 40% | 1/5 | 2676 |
| 2048 | hf_bnb_nf4 | complete | cka90_k2 | 35.63 | 0.853× | 100% | 2/5 | 2677 |
| 2048 | hqq_gemlite_w4a16 | complete | vanilla | 26.11 | — | — | 5/5 | 2788 |
| 2048 | hqq_gemlite_w4a16 | complete | pld_k4 | 30.52 | 1.170× | 0% | 4/5 | 2799 |
| 2048 | hqq_gemlite_w4a16 | complete | cka90_k2 | 21.13 | 0.810× | 100% | 5/5 | 2799 |

Cross-backend vanilla exactness by context: 128: 0/5; 384: 0/5; 1024: 0/5; 2048: 0/5

Token exactness is measured against each backend's own vanilla output. Cross-backend output differences reflect stack/quantization fidelity. Do not infer lossless acceleration from speedup alone.
