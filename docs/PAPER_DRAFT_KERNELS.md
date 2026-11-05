# M-Invariant 4-bit Kernels for Bitwise-Exact Speculative Decoding

DRAFT v0 (2026-11-05). Written for: ML-systems reviewers (target TMLR or a workshop; see "Venue" at the end). Every number comes from `results/kernels/` and `docs/PLAN_SHAPE_INVARIANT_KERNELS.md` sections 13-17. `[VERIFY]` marks claims that need a citation check before submission. `[TODO]` marks missing experiments.

## Abstract

Speculative decoding is supposed to leave the output distribution unchanged. For greedy decoding of 4-bit quantized models it does not leave the output *bits* unchanged. GPU kernels choose their reduction strategy from the number of rows M, so verifying a block of M tokens does not give the same logits as decoding those tokens one at a time. We measured this: prompt-lookup decoding (PLD) leaves the sequential greedy trajectory in 22-82% of sequences, and neural draft models in 34-66%. Running the verification row by row restores exactness but gives up the point of batching, which is to read the weights once.

We build 4-bit kernels (LM head, attention, RMSNorm, NF4 GEMM) whose result for each row does not depend on M or on the other rows in the block. The construction is fixed row tiles with zero padding, fixed reduction order, and no split-K, atomics or autotuning. We check the property bitwise on tens of thousands of rows per kernel. On top of these kernels we implement an exact PLD loop captured in a CUDA graph. Over 700 runs on 4 models and 3 datasets (Qwen2.5-0.5B/1.5B/3B, OLMo-2-1B; GSM8K, HumanEval, CNN/DM) we found 0 mismatches in generated tokens and committed KV cache. The loop is 1.14-1.32x faster than exact M=1 decoding built from the same kernels, and 1.27-1.33x faster than the earlier row-wise exact loop. We also report negative results: a real neural draft model is slower than no speculation (0.66x) on this target, and further GEMM tuning is worth at most about 10%. Finally, exactness is only defined relative to a fixed kernel set: our reference differs from stock HF+bitsandbytes decoding in 56-70% of sequences, with no detectable accuracy change in a 50-prompt GSM8K check.

## 1. Introduction

Speculative decoding (Leviathan et al. 2023; Chen et al. 2023) [VERIFY] verifies a block of drafted tokens in one forward pass. The usual argument says greedy verification reproduces the greedy sequence. That argument assumes the block forward pass produces the same logits as M sequential single-token passes. In floating point on a GPU this fails: libraries pick different tile shapes, split-K factors, or entirely different kernels depending on M. bitsandbytes 0.50.0, for example, switches its 4-bit GEMM kernel at M=4. The sequence the user would have obtained without speculation is therefore not what speculation returns.

Prior work on batch-invariant operations addresses the same shape-dependence for batch size and serving determinism (Thinking Machines 2025; LLM-42) [VERIFY]. Our setting is different in two ways: the changing dimension is the number of *verification rows within one sequence*, and the weights are 4-bit NF4 quantized, where the kernels are custom dequantize-and-multiply and not vendor GEMMs.

Contributions:
1. A set of M-invariant kernels for 4-bit decoding: LM head (Triton), attention (Triton, split-KV with absolute chunks), RMSNorm (Triton), NF4 GEMM (CUDA C++). Each is checked bitwise for independence from M and row position.
2. An exact speculative decoding loop (PLD) that runs as a CUDA graph, with rollback of the KV cache by resetting its length. 700 runs, 0 mismatches, 1.14-1.32x speedup over exact M=1 decoding.
3. A measurement of the problem (per-operator shape sensitivity, trajectory divergence of naive PLD and draft-model SD).
4. Negative results and a clarification of what "exact" means (Section 7).

## 2. What "exact" means here

Let `R` be sequential greedy decoding with a fixed kernel set `K` (M=1 for every step). A speculative decoder is *exact w.r.t. K* if it returns, for every prompt, the same tokens as `R` and leaves the same KV cache bytes. We do not claim equality with stock HF+bnb decoding; that is a different kernel set. Section 6.4 measures how far the two are apart.

## 3. Motivation: operator audit

We run each operator of a 4-bit transformer on the same row inside blocks of different M and compare bits. [TODO: insert the table from `scripts/audit_op_shape_sensitivity.py` / `shape_audit.py summarize`; cite the PLD and draft-model divergence numbers: 22-82% and 34-66% of sequences.] The end-to-end consequence is the divergence rate above.

## 4. Kernel design

**Principle.** Output element (m, n) is computed by the same instruction sequence regardless of M and of the other rows. We get this by: (i) padding the row tile to a fixed MP with zeros, (ii) fixed reduction order over K, (iii) no split-K, no atomics, no autotune, (iv) no cross-row data dependence. Invariance is verified empirically; it is not proved.

**LM head.** Triton `tl.dot` on a 16-row padded tile, K in chunks of 128, fp32 accumulate. Equal bit for bit to cuBLAS GEMM for M>=2, but not to cuBLAS's M=1 gemv path (1.2% of elements differ). That is the reason the reference differs from the stock path. Cost: 3.42 ms vs 3.34 ms for cuBLAS at M=1 (1.02x), constant for M=1..8. An earlier version using `tl.sum` was 7.3x slower and was dropped.

**Attention.** Split-KV with absolute chunks (256 keys, tiles of 64), online softmax per chunk, partial results combined in increasing chunk order. Fully masked chunks are exact no-ops, so a row's result does not depend on how many other rows or later positions are present. For CUDA-graph mode the prefix length is an int32 device scalar. GQA uses compact K/V. Cost: up to 2.1x SDPA at M=1 but 5-25x faster than masked SDPA at M=3.

**RMSNorm.** One program per row.

**NF4 GEMM.** bitsandbytes layout (high nibble first, blocksize 64, double-quantized absmax). One warp handles R output rows and MP activation rows. Weights are read and dequantized once, which is where the saving over row-wise execution comes from. Each lane handles 32 K elements per 1024-element step, fixed `fmaf` order, xor-butterfly reduction. Variants MP in {1,2,4,8} x R in {1,2,4,8} give identical per-row bits. Two Triton versions were 4-20x slower than bnb and were dropped. Full decode-step time inside a CUDA graph: M=1 15.0 ms (bnb 16.3), M=3-4 17.7 ms (bnb 21.2-31.5), M=5-6 23.6 ms (bnb 31).

## 5. Exact speculative loop

PLD proposes K=2 n-gram continuations (n-gram 3). The block of M<=3 rows is verified by the invariant kernels in one CUDA graph replay. Accepted tokens are committed; rejected rows are rolled back by setting the static cache's length (`cumulative_length.fill_(n)`). The attention prefix tensor is updated before every replay. Because every kernel is M-invariant, the committed KV bytes equal those of sequential decoding, which we check directly.

## 6. Evaluation

Setup: one RTX 4050 Laptop GPU (6 GB), Triton 3.6.0, nvcc 12.8, bitsandbytes 0.50.0, greedy decoding, batch 1, 128 new tokens, K=2.

### 6.1 Invariance

0 mismatching rows in 27,000 (LM head, 5 shapes), 6,300 each for Qwen 3B/1.5B/0.5B attention and 2,100 for MHA (context 128-3000, bf16 and fp16), 7,000 (NF4 GEMM). Error vs an fp32 dequantized reference matches bnb's.

### 6.2 End-to-end exactness and speed

Baseline is exact M=1 decoding with the same invariant kernels, in a CUDA graph.

| Cell | Prompts | Exact | Baseline tok/s | PLD loop tok/s | Speedup | Per-prompt median [min, max] |
|---|---|---|---|---|---|---|
| Qwen2.5-3B GSM8K | 200 | 200/200 | 61.2 | 71.8 | 1.17x | 1.18 [1.01, 1.40] |
| Qwen2.5-3B HumanEval | 100 | 100/100 | 58.9 | 77.6 | 1.32x | 1.32 [1.01, 2.22] |
| Qwen2.5-3B CNN/DM | 100 | 100/100 | 58.7 | 70.7 | 1.21x | 1.21 [0.98, 1.54] |
| Qwen2.5-1.5B GSM8K | 100 | 100/100 | 97.1 | 115.6 | 1.19x | 1.18 [1.01, 1.53] |
| Qwen2.5-0.5B GSM8K | 100 | 100/100 | 196.0 | 241.9 | 1.23x | 1.22 [1.05, 1.67] |
| OLMo-2-1B GSM8K | 100 | 100/100 | 136.2 | 155.1 | 1.14x | 1.13 [1.02, 1.37] |

The one-sided 95% Wilson upper bound on the mismatch rate is 1.9% for n=200 and 3.7% for n=100. Only one prompt (CNN/DM) is slower than baseline. Against the earlier row-wise exact loop the gain is 1.27x (GSM8K), 1.33x (CNN/DM); the HumanEval comparison is missing because the old loop crashed there and we did not diagnose it. The invariant M=1 baseline is also faster than the stock-kernel graph baseline (62.4 vs 57.8 tok/s, GSM8K).

### 6.3 Ablations

PLD K=4 is slower than K=2 (1.145x vs 1.206x, 10 prompts). Disabling weight rounding or replacing the LUT dequantization changes the GEMM time by less than 5%; at M=4 the kernel is close to the FMA bound and at M=1 about 20% above the bandwidth bound. We estimate at most 10% of step time is recoverable.

### 6.4 Reference versus stock decoding

Qwen2.5-3B, greedy, invariant vs default mode: identical sequences in 30% (GSM8K, n=50) and 44% (n=150); median first divergence at token 42 and 74.5. GSM8K accuracy is 66% (default) vs 72% (invariant) on 50 prompts; 2 vs 5 discordant pairs. This is within noise. It shows no detected degradation and does not prove equivalence. The answer extractor was written after reading outputs and applied identically to both modes.

## 7. Negative results and a caution

- **Neural draft model.** Qwen2.5-0.5B bf16 drafting for the 3B NF4 target is exact but 0.66x. A draft step costs about 10 ms against 15 ms for a target step. From measured costs, even an NF4 draft gives about 0.87x and a free draft is capped near 1.36x. We could not test larger targets, where the balance may change. [TODO]
- **Kernel tuning** is bounded at about 10% (Section 6.3).
- **A silent bug that "exact" would not catch.** An early NF4 kernel read the fp16 bias as bf16 and produced garbage that was still bitwise exact, because every mode generated the same degenerate text. It was found by reading outputs. Exactness tests need text- or accuracy-level checks.
- **Different valid implementations disagree.** Any two correct implementations with different summation order separate on a large share of sequences (Section 6.4). Exactness is meaningful only against a fixed, named kernel set.

## 8. Threats to validity

- One GPU (RTX 4050 Laptop), models up to 3B, batch 1, greedy only. No >=7B model and no second GPU.
- Invariance is established by construction plus tests, not by proof, and depends on compiler versions (Triton 3.6.0, nvcc 12.8) and on the bitsandbytes layout (0.50.0 only; 0.50.1/0.50.2 untested).
- Phi-3.5 is unsupported (head_dim 96 in the attention kernel).
- Speedup depends on how often PLD finds a draft: HumanEval is highest; the range 1.14-1.32x is modest.
- Accuracy comparison is small (n=50) and one task.
- Related-work comparison to LLM-42 and concurrent batch-invariance work is not done. [TODO]

## 9. Related work [VERIFY all]

Speculative decoding (Leviathan et al.; Chen et al.); prompt lookup decoding; batch-invariant operations for deterministic inference (Thinking Machines "Defeating Nondeterminism in LLM Inference"); LLM-42; Integer Alibi; bitsandbytes NF4 / QLoRA (Dettmers et al.). See `docs/direction_c/LITERATURE_NOTES_DIRECTION_C.md`.

## 10. Reproducibility

`reproduce.sh` runs the tests and the audit summary. Kernels: `src/zassd/kernels/`. Experiments: `scripts/wp1_*` ... `wp6_*`, `run_wp5_scale.sh`, `run_wp6.sh`. Data: `results/kernels/`. Tests: `tests/test_lm_head_invariant.py` (GPU), 16 tests pass.

## Venue

With one small GPU and models <=3B this fits TMLR or a systems/efficient-inference workshop. For ICML/MLSys it needs at least one more GPU and a >=7B target (also where draft models may pay off). Deadlines and formats must be confirmed on the official sites.
