# M-Invariant 4-bit Kernels for Bitwise-Exact Speculative Decoding

DRAFT v0 (2026-11-05). Written for: ML-systems reviewers (target TMLR or a workshop; see "Venue" at the end). Every number comes from `results/kernels/` and `docs/PLAN_SHAPE_INVARIANT_KERNELS.md` sections 13-17. Bibliographic details of the references in Section 11 were checked against search results on 2026-11-05 (author lists, venue, arXiv ids); the content claims about LLM-42, Integer Alibi, Yuan et al. and Chordiya rest on partial readings (see Section 9). `[TODO]` marks missing experiments.

## Abstract

Speculative decoding is supposed to leave the output distribution unchanged. For greedy decoding of 4-bit quantized models it does not leave the output *bits* unchanged. GPU kernels choose their reduction strategy from the number of rows M, so verifying a block of M tokens does not give the same logits as decoding those tokens one at a time. We measured this: prompt-lookup decoding (PLD) leaves the sequential greedy trajectory in 26-82% of sequences (50-prompt cells; up to 92% on an independent CNN/DM set), and a neural draft model in 34-66%. Running the verification row by row restores exactness but gives up the point of batching, which is to read the weights once.

We build 4-bit kernels (LM head, attention, RMSNorm, NF4 GEMM) whose result for each row does not depend on M or on the other rows in the block. The construction is fixed row tiles with zero padding, fixed reduction order, and no split-K, atomics or autotuning. We check the property bitwise on tens of thousands of rows per kernel. On top of these kernels we implement an exact PLD loop captured in a CUDA graph. Over 700 runs on 4 models and 3 datasets (Qwen2.5-0.5B/1.5B/3B, OLMo-2-1B; GSM8K, HumanEval, CNN/DM) we found 0 mismatches in generated tokens and committed KV cache. The loop is 1.14-1.32x faster than exact M=1 decoding built from the same kernels, and 1.27-1.33x faster than the earlier row-wise exact loop. We also report negative results: a real neural draft model is slower than no speculation (0.66x) on this target, and further GEMM tuning is worth at most about 10%. Finally, exactness is only defined relative to a fixed kernel set: our reference differs from stock HF+bitsandbytes decoding in 56-70% of sequences, with no detectable accuracy change in a 50-prompt GSM8K check.

## 1. Introduction

Speculative decoding (Leviathan et al. 2023; Chen et al. 2023) verifies a block of drafted tokens in one forward pass. The usual argument says greedy verification reproduces the greedy sequence. That argument assumes the block forward pass produces the same logits as M sequential single-token passes. In floating point on a GPU this fails: libraries pick different tile shapes, split-K factors, or entirely different kernels depending on M. bitsandbytes 0.50.0, for example, switches its 4-bit GEMM kernel at M=4. The sequence the user would have obtained without speculation is therefore not what speculation returns.

Prior work on batch-invariant operations addresses the same shape-dependence for batch size and serving determinism (He et al. 2025; LLM-42). Our setting is different in two ways: the changing dimension is the number of *verification rows within one sequence*, and the weights are 4-bit NF4 quantized, where the kernels are custom dequantize-and-multiply and not vendor GEMMs.

Contributions:
1. A set of M-invariant kernels for 4-bit decoding: LM head (Triton), attention (Triton, split-KV with absolute chunks), RMSNorm (Triton), NF4 GEMM (CUDA C++). Each is checked bitwise for independence from M and row position.
2. An exact speculative decoding loop (PLD) that runs as a CUDA graph, with rollback of the KV cache by resetting its length. 700 runs, 0 mismatches, 1.14-1.32x speedup over exact M=1 decoding.
3. A measurement of the problem (per-operator shape sensitivity, trajectory divergence of naive PLD and draft-model SD).
4. Negative results and a clarification of what "exact" means (Section 7).

## 2. What "exact" means here

Let `R` be sequential greedy decoding with a fixed kernel set `K` (M=1 for every step). A speculative decoder is *exact w.r.t. K* if it returns, for every prompt, the same tokens as `R` and leaves the same KV cache bytes. We do not claim equality with stock HF+bnb decoding; that is a different kernel set. Section 6.4 measures how far the two are apart.

## 3. Motivation: operator audit

We run each operator of a stock 4-bit transformer (HF + bitsandbytes 0.50.0 NF4, RTX 4050, sm_89) on the same rows inside blocks of M rows and compare to row-by-row execution bitwise (GSM8K prompts, 960 rows per cell, natural contexts). "Proj" is the fraction of mismatching output elements of the NF4 projections; RMSNorm is the count of mismatching elements; LM head and SDPA are fractions of mismatching elements; "flips" is the number of rows whose argmax changes.

| Model | M | Proj | RMSNorm (count) | LM head | SDPA | Argmax flips |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B | 2 | 0.000% | 60 | 0.193% | 18.6% | 4/960 |
| Qwen2.5-3B | 3 | 0.000% | 87 | 0.193% | 18.6% | 4/960 |
| Qwen2.5-3B | 4 | 12.96% | 244 | 0.193% | 18.6% | 4/960 |
| Qwen2.5-3B | 8 | 12.96% | 199 | 0.193% | 18.6% | 5/960 |
| Qwen2.5-1.5B | 3 / 4 | 0.000% / 13.85% | 0 / 92 | 0.109% | 18.3% | 7 / 9 of 960 |
| Qwen2.5-0.5B | 3 / 4 | 0.000% / 14.27% | 0 | 0.000% | 3.6% | 6 / 9 of 960 |
| OLMo-2-1B | 3 / 4 | 0.000% / 15.42% | 111 / 189 | 0.048% | 17.4% | 2 / 1 of 960 |
| Phi-3.5-mini | 3 / 4 | 0.000% / 14.79% | 358 / 459 | 0.083% | 19.2% | 3 / 4 of 960 |

Reading: the NF4 projections are shape-stable up to M=3 and jump to about 13-15% mismatching elements at M>=4, which matches the bitsandbytes kernel switch at M=4 (0.50.0 only, sm_89; the dispatch heuristic changed in 0.50.1/0.50.2, so the threshold is a property of this version and GPU and we make no claim for others). SDPA and the LM head are sensitive at every M, and RMSNorm is sensitive in some models. The end-to-end consequence is trajectory divergence: naive PLD leaves the sequential greedy trajectory in 26-82% of sequences (`paper/figures_c/fig1_trajectory_divergence.csv`: 5 models x GSM8K/HumanEval/CNN-DM, 50 prompts per cell, plus up to 92% on independent CNN/DM sets of 60-120 prompts), while running the verification row by row gives 0 divergent sequences in every cell (Wilson upper bound 3-7%). A real draft model (Qwen2.5-0.5B drafting for Qwen2.5-3B, K=2/4/6, 50 prompts) diverges in 34-38% of GSM8K and 64-66% of CNN/DM sequences; patching all four operators to row-wise execution gives 0 (`scripts/audit_draftmodel_sd_divergence.py`, raw files under `experiments/.../native_bnb_profiles/d_draft_*.json`, which are git-tracked but not in `results/`).

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

The one-sided 95% Wilson upper bound on the mismatch rate is 1.9% for n=200 and 3.7% for n=100. Only one prompt (CNN/DM) is slower than baseline. Against the earlier row-wise exact loop the gain is 1.27x (GSM8K), 1.33x (CNN/DM); HumanEval (100 prompts, 1024-token cache, run back to back: old loop, invariant loop, old loop again) gives 81.9 vs 58.9 and 59.4 tok/s = 1.39x and 1.38x, all three runs exact; the two old-loop runs differ by 0.9%, which bounds the session drift. (An earlier 20-prompt run gave 1.34x.) The invariant M=1 baseline is also faster than the stock-kernel graph baseline (62.4 vs 57.8 tok/s, GSM8K).

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
- Related work was read from page text and PDF extractions; the evaluation sections of LLM-42 and several preprints were only partly read. Our search was narrow and may have missed work.

## 9. Related work

Citations (full list in Section 11) are from our reading notes (`docs/direction_c/LITERATURE_NOTES_DIRECTION_C.md`, sections 4-9). Statements about papers we only saw through summaries are marked (summary). Re-check the partially read ones against the final versions before submission.

**Batch-invariant inference.** He et al. (Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025) trace nondeterminism to kernels whose reduction order depends on batch size (RMSNorm, matmul, attention) and give batch-invariant kernels; their library covers `mm`, `addmm`, `log_softmax` and `mean`, with bf16 examples and no quantization. SGLang and vLLM provide deterministic modes built on the same idea. Our invariance target differs: the varying dimension is the number of verification rows of one sequence, and the matmul is a 4-bit dequantize-and-multiply kernel.

**LLM-42** (arXiv 2601.17768) uses a decode-verify-rollback scheme, inspired by speculative decoding, to make serving deterministic. It verifies over fixed-size windows so kernels always see the same shape, and it reports results for Llama-3.1-8B on H100s. By its own statement it does not integrate with speculative-decoding-based inference, and it does not study weight-quantized models. We use the invariance in the opposite direction: to keep a real speculative decoder exact against the sequential reference.

**Numerical nondeterminism.** Yuan et al. (arXiv 2506.09501) vary GPU type, GPU count, batch size and precision and report up to 9% standard deviation in AIME'24 accuracy; they propose LayerCast and do not examine 4-bit weights or speculative decoding (summary for later sections). "The Integer Alibi" (arXiv 2608.13756, single-author preprint) studies W8A8 INT8 kernels, where an exact INT32 accumulator moves the difference to scaling and rounding. That mechanism does not apply to NF4, whose accumulation is floating point.

**Speculative decoding on consumer hardware.** Chordiya (arXiv 2607.17283, single-author preprint) observes on llama.cpp Metal that scoring the same context in a batch of five versus one at a time shifts logits by up to about 0.1 and can flip the argmax, and concludes that bit-exact greedy equivalence on that backend is unattainable when baseline and verifier use different batch shapes. Our work agrees with the observation (Section 3) and shows that on CUDA, with kernels made M-invariant, the equivalence can be obtained against a fixed reference. The qualitative observation is not new; our measurement per operator, the kernels, and the exact loop are.

**4-bit kernels.** bitsandbytes NF4 / QLoRA (Dettmers et al. 2023); its PR #1949 (in 0.50.0) added the runtime-dispatched 4-bit GEMM kernels we measure against. vLLM is reported to disable Marlin in its batch-invariant mode because Marlin is not batch-invariant (search-result snippet, not verified in the vLLM docs).

**Speculative decoding.** Leviathan et al. 2023; Chen et al. 2023; prompt lookup decoding (Saxena 2023, now in HF transformers as `prompt_lookup_num_tokens`). The standard losslessness argument is about the output distribution under exact arithmetic; it does not cover floating-point shape dependence.

## 10. Reproducibility

`reproduce.sh` runs the tests and the audit summary. Kernels: `src/zassd/kernels/`. Experiments: `scripts/wp1_*` ... `wp6_*`, `run_wp5_scale.sh`, `run_wp6.sh`. Data: `results/kernels/`. Tests: `tests/test_lm_head_invariant.py` (GPU), 16 tests pass.

## Venue

With one small GPU and models <=3B this fits TMLR or a systems/efficient-inference workshop. For ICML/MLSys it needs at least one more GPU and a >=7B target (also where draft models may pay off). Deadlines and formats must be confirmed on the official sites.

## 11. References (bibliographic details verified by search 2026-11-05)

- Leviathan, Kalman, Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023 (oral), PMLR 202. arXiv:2211.17192.
- Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper. Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318 (DeepMind, 2023).
- Dettmers, Pagnoni, Holtzman, Zettlemoyer. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023. arXiv:2305.14314.
- He (Horace) et al. Defeating Nondeterminism in LLM Inference. Thinking Machines Lab blog, 2025-09-10. thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference (we read the post and secondary coverage; the first author's full name should be taken from the post itself).
- Saxena. Prompt Lookup Decoding. GitHub apoorvumang/prompt-lookup-decoding, 2023.
- LLM-42: Enabling Determinism in LLM Inference with Verified Speculation. arXiv:2601.17768 (read from PDF extraction, evaluation section only partly).
- Yuan et al. Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference. arXiv:2506.09501.
- The Integer Alibi. arXiv:2608.13756; follow-up arXiv:2609.00363 (single-author preprints).
- Chordiya. Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware. arXiv:2607.17283.
- bitsandbytes PR #1949 (4-bit GEMM kernels, merged 2026-05-21).
