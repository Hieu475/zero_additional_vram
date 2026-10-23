# Paired BNB/HQQ kernel profile

Fixed Qwen2.5-3B prompt `qasper_119_ctx384` (384 input tokens), 32 measured output tokens, static KV capacity 436, SDPA, 32-token full warmup. Each capture is one separate process, instrumented by Nsight Systems. These are diagnostic profiles, not production TPS.

| Backend | Method | GPU kernel sum (ms) | Top kernel (share) | Sync API calls | Launch API calls | Profiled E2E (s) | Peak allocated (MB) |
|---|---|---:|---|---:|---:|---:|---:|
| hf_bnb_nf4 | vanilla | 623.9 | `void gemm_4bit_simt<__half, (int)1>(const T1 *, const unsigned char *, const float *, const unsigned char *, const float *, const float *, T1 *, const T1 *, int, int, int, int, int)` (44.1%) | 65 | 64784 | 1.259 | 2107 |
| hf_bnb_nf4 | pld_k4 | 691.2 | `void gemm_4bit_simt<__half, (int)1>(const T1 *, const unsigned char *, const float *, const unsigned char *, const float *, const float *, T1 *, const T1 *, int, int, int, int, int)` (28.2%) | 91 | 59926 | 1.206 | 2111 |
| hf_bnb_nf4 | cka90_k2 | 693.5 | `void gemm_4bit_simt<__half, (int)1>(const T1 *, const unsigned char *, const float *, const unsigned char *, const float *, const float *, T1 *, const T1 *, int, int, int, int, int)` (23.9%) | 95 | 64589 | 1.320 | 2111 |
| hqq_gemlite_w4a16 | vanilla | 610.7 | `gemv_INT_revsplitK_kernel` (45.2%) | 65 | 59168 | 1.788 | 2224 |
| hqq_gemlite_w4a16 | pld_k4 | 635.5 | `gemv_INT_revsplitK_kernel` (40.6%) | 94 | 60091 | 1.985 | 2228 |
| hqq_gemlite_w4a16 | cka90_k2 | 656.9 | `gemv_INT_revsplitK_kernel` (46.1%) | 95 | 59165 | 1.965 | 2228 |

## NVTX phase observations

- `hf_bnb_nf4` / `cka90_k2`: draft forward 715.1 ms; target verify 435.9 ms (summed NVTX durations).
- `hqq_gemlite_w4a16` / `cka90_k2`: draft forward 1113.1 ms; target verify 681.1 ms (summed NVTX durations).

## Limits

- Quantized weights differ between BNB NF4 and HQQ W4A16, so cross-backend kernel differences are stack comparisons, not an isolated kernel treatment.
- Nsight instrumentation changes timings; the TPS fields are not production benchmarks.
- One prompt and one session per cell cannot support general performance claims.
- The host environment disabled CPU context-switch tracing, so host scheduling gaps are not directly observable.
- Nsight Systems captures API and kernel activity; no matched FLOP/byte or arithmetic-intensity measurement is included.

Capture hashes and per-cell metadata are in `paired_backend_profiles.json`.
