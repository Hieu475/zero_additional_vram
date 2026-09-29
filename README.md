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

| Method | Draft Mechanism | Additional Weight VRAM | Throughput (tok/s) | Relative Speedup | Acceptance Rate | Exact Match (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Target** | N/A (Standard AR) | 0.0 MB | 41.0 | 1.00× | 100.0% | 100.0% |
| **Prompt Lookup (PLD)** | N-gram Context Matching | **0.0 MB** | **59.8** | **1.39×** | **100.0%** | 100.0% |
| **ZASSD (mid_12, K=1)** | Logical Layer Skip (L12–23) | **0.0 MB** | 40.2 | 0.98× | 73.8% | 100.0% |
| **ZASSD HW Controller** | Adaptive Depth & Length | **0.0 MB** | 36.5 | 0.89× | 88.1% | 100.0% |
| **KnapSpec (ICML'24)** | Whole-Layer Knapsack | **0.0 MB** | 35.2 | 0.86× | 73.2% | 100.0% |
| **SpecBound (ACL'24)** | Bounded Layer Skip | **0.0 MB** | 33.8 | 0.83× | 74.1% | 100.0% |

---

## 🧪 Formal Verification & Invariant Test Suite

All 60 unit and systems tests run under `pytest`:

```bash
pytest tests/ -v
```

Verified Invariants:
1. **Invariant 1 (Zero-Copy Prefix Sharing)**: Pointer aliasing between target and ephemeral draft cache.
2. **Invariant 2 (Draft-Private Mutation)**: In-place draft writes into candidate slots leave canonical prefix memory bitwise identical.
3. **Invariant 3 (Exact Rollback)**: Truncation resets sequence length to $P + \text{accepted}$ with zero residual token pollution.
4. **Invariant 4 (Zero Dynamic Memory Allocation)**: Pre-allocated buffer addresses remain strictly constant across all forward steps.
5. **Exactness Equivalence**: Numerical logits and greedy tokens match ground-truth full recomputation.

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
└── tests/                       # Complete formal verification test suite (60 tests)
```

## 📜 License

MIT License.
