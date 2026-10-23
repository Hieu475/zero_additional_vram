# Matched-prefix numerical audit and paired kernel profile

## Protocol and provenance

- Model revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`; GPU: `NVIDIA GeForce RTX 4050 Laptop GPU`; PyTorch/CUDA: `2.11.0+cu128` / `12.8`.
- Numerical cohort: the 16 IDs in `experiments/numerical_causality/config.json`, input token IDs from the prepared prompt file, 32 greedy output tokens, PLD K=4, prompt+generated candidates, SDPA and static KV. BNB is NF4 double-quant/FP16; HQQ is W4A16/GemLite. Quantized weight values differ, so compare within-backend correctness and same-backend interventions.
- HQQ factorial run verified identical logical prefix/candidate cells and cache snapshots; integrity checks passed: `True`.
- Run source at capture time was based on commit `3578a0c21f1a6b1e9cd24682bfa51cb30f05d4b0` with local modifications. Working tree contained 49 pre-existing or current status entries; exact hashes for the scripts used are in `manifest.json`.

## Matched-prefix results

| Measurement | BNB NF4 | HQQ + GemLite |
|---|---:|---:|
| Diagnostic prompts | 16 | 16 |
| PLD-vs-own-vanilla mismatches in recorded run | 16/16 | 2/16 in trace; 0/16 in separate factorial run |
| First active/canonical KV divergence | See prior BNB factorial artifact | 16/16 |
| First logit divergence | See prior BNB factorial artifact | 16/16 |
| First-cycle attribution | {"execution_sensitive": 1, "kv_history_sensitive": 15} | {"no_argmax_divergence_in_selected_row": 16} |

In the first HQQ trace, the two PLD/vanilla sequence differences occurred at generated-token positions 1, 25 with vanilla top-1 margins 0.015625, 0.015625. A separate HQQ factorial execution on the same 16 IDs produced 0/16 sequence mismatches and no selected-row argmax flips. This disagreement is material and triggers the repeatability result below; it should not be averaged away.

## Repeatability check

Five fresh-cache greedy continuations were run per backend and prompt in the same process. BNB matched its first run 5/5 times on both prompts. HQQ matched 3/5 on `gov_report_134_ctx384` and 4/5 on `narrativeqa_55_ctx384`. HQQ run-to-run differences first appeared at token index 1 (top-1 margins 0.0 and 0.015625) and index 25 (margins 0.0 and 0.03125). These are two selected low-margin prompts, not an estimate of whole-cohort instability. Current evidence shows HQQ output instability near close/tied decisions; it does not yet identify whether GemLite kernel nondeterminism, accumulation order, or another execution detail causes it.

This means the observed higher HQQ exactness in the n=20 matrix is promising but cannot yet be attributed solely to lower batched-verification error or larger margins. We need repeatability on a broader held-out set and direct top-2/logit capture around the unstable positions.

## Paired kernel profile

Same prompt, context, static capacity, output budget, SDPA, and full 32-token warmup. One process per backend/method. Nsight Systems profile sums are diagnostic, not production timing.

| Backend | Method | GPU-kernel sum | Profiled E2E | CKA draft forward / cycle | CKA target verify / cycle | Top kernel |
|---|---|---:|---:|---:|---:|---|
| hf_bnb_nf4 | vanilla | 623.9 ms | 1.259 s | — | — | `void gemm_4bit_simt<__half, ` |
| hf_bnb_nf4 | pld_k4 | 691.2 ms | 1.206 s | — | — | `void gemm_4bit_simt<__half, ` |
| hf_bnb_nf4 | cka90_k2 | 693.5 ms | 1.320 s | 65.0 ms | 39.6 ms | `void gemm_4bit_simt<__half, ` |
| hqq_gemlite_w4a16 | vanilla | 610.7 ms | 1.788 s | — | — | `gemv_INT_revsplitK_kernel` |
| hqq_gemlite_w4a16 | pld_k4 | 635.5 ms | 1.985 s | — | — | `gemv_INT_revsplitK_kernel` |
| hqq_gemlite_w4a16 | cka90_k2 | 656.9 ms | 1.965 s | 101.2 ms | 61.9 ms | `gemv_INT_revsplitK_kernel` |

On the measured CKA90/K2 prompt, HQQ summed draft and target verification NVTX time were both about 1.56× the BNB values, despite similar total GPU-kernel sums. This points to meaningful work outside the aggregate kernel-duration total (or different execution/candidate behavior); the trace is not sufficient to identify the cause. For PLD, HQQ took 31 verification cycles and accepted 1 draft token versus 28 cycles and 4 accepted tokens for BNB, so acceptance and weight/stack behavior also differ. Cross-backend results are not a kernel-only treatment.

The largest HQQ kernel family was `gemv_INT_revsplitK_kernel` (about 41–46% of aggregate kernel duration), followed by `gemm_INT_kernel` (about 21–23%). The profile also shows cublas GEMV contribution. This is stronger evidence about which HQQ kernels dominate than inferring a bottleneck from dequantization kernels, but still needs targeted Nsight Compute counters and repeated, uninstrumented paired timing. Detailed rows and capture hashes are in `../kernel_profile/paired_backend_profiles.json`.

## Next measurement

1. Capture top-2 IDs and logits repeatedly at the HQQ low-margin token positions to isolate decision ties from run-to-run backend variation.
2. Run targeted Nsight Compute on HQQ GEMV/GEMM shapes selected from these traces; collect DRAM/SM throughput, occupancy, and register metrics.
3. Only then choose one optimization and re-run production paired benchmarks. Keep backend-specific within-stack exactness separate from cross-backend fidelity.
