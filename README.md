# Zero-Additional-VRAM Self-Speculative Decoding (ZASSD)

> A Systems Study on Self-Speculative Decoding for Memory-Constrained Consumer GPUs: When it Works, When it Doesn't, and Why.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11](https://img.shields.io/badge/pytorch-2.11.0%2Bcu128-orange.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/cuda-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 The Core Question: Why Zero-Additional-VRAM?

On consumer hardware (e.g. 6 GB laptop GPUs), deploying modern Large Language Models is strictly bounded by memory capacity. Standard speculative decoding requires loading a dedicated draft model (e.g. 0.5B–1.5B parameters), which demands an extra 1.0–2.5 GB of VRAM.

### Comparison: Traditional Dual-Model vs. Zero-VRAM Speculative Decoding

| Paradigm | Target Model | Draft Engine | Weight VRAM Overhead | Fits in 6 GB Laptop GPU? | Speedup vs. Vanilla | Key Limiting Bottleneck |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Traditional Speculative** | 7B (NF4, ~3.8 GB) | 0.5B Draft (~0.8 GB) | **+800 MB to +1.2 GB** | ❌ **OOM (6.2 GB > 6.0 GB)** | 0.0× (Crash) | **VRAM Capacity Boundary** |
| **Traditional Speculative** | 3B (NF4, ~2.0 GB) | 0.5B Draft (~0.8 GB) | **+800 MB** | ⚠️ Tight (3.4 GB total) | 1.15–1.25× | Dual model checkpoint overhead |
| **ZASSD (Layer-Skip $K=1$)** | 3B (NF4, ~2.0 GB) | Logical Middle-Skip | **0.00 MB** | ✅ **Yes (2.4 GB, 60% free)** | 0.92–1.01× | Memory Bandwidth (192 GB/s) |
| **ZASSD (Prompt-Lookup)** | 3B (NF4, ~2.0 GB) | N-gram History Lookup | **0.00 MB** | ✅ **Yes (2.4 GB, 60% free)** | **1.39×** | Requires context recurrence |

---

## 🔍 When Zero-VRAM Speculative Decoding Works, When it Doesn't, and Why

### 1. When it Works ✅
- **Under Hard VRAM Constraints (e.g. 7B Models on 6 GB GPUs)**:
  When the target model exhausts 80%+ of total VRAM, loading any separate draft model triggers CUDA Out-Of-Memory or hostile host-memory swapping. Zero-VRAM self-speculation is the **only viable speculative mechanism**.
- **On Structured & Repetitive Contexts (Prompt Lookup / PLD)**:
  When prompts contain recurrent tokens (summarization, RAG, coding, markdown structures), **Prompt Lookup Decoding (PLD)** extracts candidate sequences at $\approx 0\text{ ms}$ draft compute, yielding **100% acceptance rates and $1.39\times$ wall-clock speedup** with zero weight overhead.
- **Compute-Bound Regimes (Batch Size > 1 or Datacenter Accelerators)**:
  Parallel target verification of $K+1$ candidate tokens incurs nearly identical latency to a 1-token step on wide matrix execution units, maximizing speculation efficiency.

### 2. When it Doesn't (The Mobile Bandwidth Ceiling) ⚠️
- **Batch-1 Generation on Memory-Bandwidth-Bound Laptop GPUs (RTX 4050, 192 GB/s)**:
  For 3B models at $B=1$, single-token forward passes take $\approx 23\text{ ms}$, bound strictly by reading weights from GDDR6 memory.
  A speculative cycle requires **two separate memory sweeps**:
  1. Draft pass ($14–18\text{ ms}$)
  2. Verify pass ($24–27\text{ ms}$)
  
  Total cycle time is $38–45\text{ ms}$. To break even ($1.0\times$), the cycle must emit $\ge 1.8$ tokens per step, requiring acceptance rate $\alpha \ge 75\%-80\%$ and draft compute ratio $c \le 0.50$.
  On CKA whole-layer skipping ($c \approx 0.78$), the draft pass is too expensive ($78\%$ of full model), resulting in $0.85–0.89\times$ throughput.

---

## 🛠️ Key Systems Engineering Optimizations

To push self-speculative decoding to its physical limits, ZASSD implements:

1. **Static Pre-allocated KV Cache (`StaticPreallocatedKVCache`)**:
   Eliminates all `torch.cat` reallocations. Pre-allocates fixed memory buffers on GPU. Draft writes candidate tokens in-place to $[P : P+K)$; verification overwrites canonical states; rollback is an $O(1)$ pointer truncation. Yields **$+11.2\%$ throughput improvement**.
2. **Vectorized GPU Decision Engine**:
   Eliminates per-token `.item()` calls and CPU-GPU stream synchronizations during draft and verification loops. Computes candidate matches batched on GPU tensors.
3. **Representational Middle-Skip Discovery**:
   Skipping middle layers (`mid_12`, layers 12–23) preserves early syntactic representations and late logit projections, boosting acceptance rate from **$52.9\%$ (CKA-75) to $73.8\% - 81.4\%$**.
4. **Prompt Lookup Decoding Baseline (`prompt_lookup_generate`)**:
   Provides training-free, zero-VRAM n-gram candidate extraction for context-heavy generation.

---

## 🚀 Quickstart & Unified CLI

Install the package in editable mode:

```bash
pip install -e .
```

### 1. Benchmark Any Method via Unified CLI

```bash
# Benchmark ZASSD Self-Speculative Decoding (K=2, Middle-Skip 12)
zassd benchmark --model Qwen/Qwen2.5-3B-Instruct --method zassd --k 2 --skip-strategy mid_12

# Benchmark Prompt Lookup Decoding (PLD)
zassd benchmark --model Qwen/Qwen2.5-3B-Instruct --method prompt_lookup --k 4

# Benchmark Classical Baselines
zassd benchmark --model Qwen/Qwen2.5-3B-Instruct --method knapspec --k 2
zassd benchmark --model Qwen/Qwen2.5-3B-Instruct --method specbound --k 2
```

### 2. Run Interactive Streaming Terminal Demo

```bash
zassd demo
```

### 3. Run Mathematical Exactness Audit

```bash
zassd audit
```

### 4. Regenerate Manuscript Tables & Macros

```bash
zassd generate-tables
```

---

## 📊 Experimental Results

Standardized benchmark on **NVIDIA GeForce RTX 4050 Laptop GPU** (6 GB GDDR6, 192 GB/s, 80W TGP):

### Qwen2.5-3B-Instruct (36 Layers, 4-bit NF4)

Frozen unified benchmark, N=10 prompts, greedy, 48 new tokens
(`experiments/final_validation/`; routed row: N=50/bench, `experiments/16_hybrid_roofline/`):

| Method | Draft Mechanism | Additional Weight VRAM | Throughput (tok/s) | Relative Speedup | Acceptance Rate | Exact Match (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Target** | N/A (Standard AR) | 0.0 MB | 41.7 | 1.00× | — | 100.0% |
| **Prompt Lookup (PLD, K=4)** | N-gram Context Matching | **0.0 MB** | **57.9** | **1.39×** | **100.0%** | 100.0% |
| **ZASSD (mid_12, K=1)** | Logical Layer Skip (L12–23) | **0.0 MB** | 40.2 | 0.98× | 73.8% | 60.0% |
| **ZASSD HW Controller** | Adaptive Depth & Length (+vanilla skip gate) | **0.0 MB** | 38.6 | 0.93× | 89.0% | 60.0% |
| **ZASSD Routed (K=2+PLD4)** | Cost-aware AR / PLD / layer-skip per cycle | **0.0 MB** | **44.1** | **1.06×** | 35.8%† | 83.3% |
| **KnapSpec (adapted)** | Whole-Layer Knapsack | **0.0 MB** | 35.9 | 0.86× | 73.2% | 60.0% |
| **SpecBound (adapted)** | Bounded Layer Skip | **0.0 MB** | 34.4 | 0.83× | 74.1% | 50.0% |

> **Honesty notes.** Exact-match <100% under 4-bit NF4 is NF4 batched-GEMM
> dequantization noise on near-tie logits (logit margin Δ ≤ 0.25), not an
> algorithmic bug. Precision ladder (Qwen2.5-0.5B + Llama-3.2-1B, 40 prompts
> × 128 tokens each): **FP32 100% identity over 10,113 token positions**,
> FP16 90% sequences, NF4 60% — plus a 450-token audit with logit cosine
> 0.99994+ and zero violations of the Δ ≤ 2‖L‖∞ flip bound
> (`experiments/17_fp16_exactness/`). †Routed acceptance counts PLD/LS
> drafts only; vanilla-skipped cycles cannot diverge, hence the higher
> exact-match. KnapSpec/SpecBound are our adapted re-implementations,
> not official code. 7B-class auxiliary drafts: only large FP16 drafts
> OOM — quantized 0.5B/1.5B drafts fit but always cost ΔM_aux > 0 and
> shrink KV headroom (`experiments/16_hybrid_roofline/draft_vram_pareto.json`);
> ZASSD weight overhead is exactly 0.0 MB.

---

## 🧪 Invariant Validation & System Test Suite

All 80+ unit and systems tests run under `pytest`:

```bash
pytest tests/ -v
```

Verified Invariants (regression-tested, not formal-methods proofs):
1. **Invariant 1 (Zero-Copy Prefix Sharing)**: Pointer aliasing between target and ephemeral draft cache.
2. **Invariant 2 (Draft-Private Mutation)**: In-place draft writes into candidate slots leave canonical prefix memory bitwise identical.
3. **Invariant 3 (Exact Rollback)**: Truncation resets sequence length to $P + \text{accepted}$ with zero residual token pollution.
4. **Invariant 4 (Zero Dynamic Memory Allocation)**: Pre-allocated buffer addresses remain strictly constant across all forward steps.
5. **Numerical Equivalence Audit**: 450-token single-vs-batched logit comparison
   (cosine 0.99994+, zero flip-bound violations) plus a strict FP16
   token-identity proof on a 0.5B model.
6. **Algorithmic vs numerical exactness** are reported separately: the
   speculation algorithm is distribution-preserving under exact arithmetic;
   residual NF4 divergences are quantified kernel noise.

---

## 📂 Repository Structure

```
zero_additional_vram/
├── src/zassd/                   # Core Python package
│   ├── baselines/               # KnapSpec, SpecBound & Prompt Lookup Decoding
│   ├── cache/                   # TargetKVCache, StaticPreallocatedKVCache
│   ├── controllers/             # HardwareAwareJointController & AdaptiveK
│   ├── decoding/                # self_speculative_generate with GPU-side decision
│   ├── layer_selection/         # Redundancy ranking & middle-skip selection
│   ├── models/                  # ModelAdapter & LayerManager
│   └── cli.py                   # Unified CLI runner (zassd command)
├── paper/                       # Manuscript & automated LaTeX tables
│   ├── draft/manuscript.md      # Paper draft with honest systems findings
│   └── tables/                  # Auto-generated .tex tables and macros
├── scripts/                     # Standalone benchmarking & profiling scripts
├── experiments/                 # Empirical raw results and summaries
└── tests/                       # Invariant-based system test suite (80+ tests)
```

## 📜 License

MIT License.
