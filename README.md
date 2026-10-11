# Zero-Additional-Model-Weight-VRAM Self-Speculative Decoding (ZASSD)

> A Systems Study on Self-Speculative Decoding for Memory-Constrained Consumer GPUs: When it Works, When it Doesn't, and Why.

"Zero additional" refers to auxiliary **model weights**. The decoder still
uses temporary KV and activation memory during drafting and verification.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11](https://img.shields.io/badge/pytorch-2.11.0%2Bcu128-orange.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/cuda-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Core Question: Why Zero-Additional-VRAM?

On consumer hardware (e.g. 6 GB laptop GPUs), deploying modern Large Language Models is strictly bounded by memory capacity. Standard speculative decoding requires loading a dedicated draft model (e.g. 0.5B–1.5B parameters). We measured on-device footprints: 0.5B-FP16 942 MB, 0.5B-NF4 459 MB, 1.5B-FP16 2946 MB, 1.5B-NF4 1182 MB. Next to a 7B NF4 target, large FP16 drafts OOM outright, and even fitting quantized drafts permanently cost auxiliary weight VRAM plus shrunken KV/context headroom. ZASSD costs exactly 0.0 MB of model-weight VRAM with the largest headroom in every configuration we measured.

### Comparison: Traditional Dual-Model vs. Zero-VRAM Speculative Decoding

| Paradigm | Target Model | Draft Engine | Weight VRAM Overhead | Fits in 6 GB Laptop GPU? | Speedup vs. Vanilla | Key Limiting Bottleneck |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Traditional Speculative** | 7B (NF4, ~3.9 GB) | 1.5B Draft FP16 (~2.9 GB) | **+2.9 GB** | **OOM (7.4 GB > 6.0 GB)** | 0.0x (Crash) | **VRAM Capacity Boundary** |
| **Traditional Speculative** | 7B (NF4, ~3.9 GB) | 0.5B Draft NF4 (~0.5 GB) | **+0.5 GB** | Fits, tight headroom | workload-dependent | KV/context headroom |
| **ZASSD (Layer-Skip $K=1$)** | 3B (NF4, ~2.0 GB) | Logical Middle-Skip | **0.00 MB** | **Yes (2.4 GB, 60% free)** | 0.92–1.01x | Memory Bandwidth (192 GB/s) |
| **Prompt Lookup (PLD)** | 3B (NF4, ~2.0 GB) | N-gram History Lookup | **0.00 MB** | **Yes (2.4 GB, 60% free)** | **1.098x** | Requires context recurrence |
| **ZASSD Routed** | 3B (NF4, ~2.0 GB) | Cost-aware AR / PLD / layer-skip | **0.00 MB** | **Yes** | **1.090x** | Slightly below PLD on the paired N=150 benchmark |

---

## When Zero-VRAM Speculative Decoding Works, When it Doesn't, and Why

### 1. When it Works
- **Under Hard VRAM Constraints (e.g. 7B Models on 6 GB GPUs)**:
  When the target model exhausts 80%+ of total VRAM, loading a large auxiliary draft model triggers CUDA Out-Of-Memory or hostile host-memory swapping, and even small quantized drafts (measured 459–1182 MB) eat KV/context headroom. Zero-weight self-speculation costs 0.0 MB of model-weight VRAM with strictly better memory headroom in every configuration we measured.
- **On Structured and Repetitive Contexts (Prompt Lookup / PLD)**:
  When prompts contain recurrent tokens (summarization, RAG, coding, markdown structures), **Prompt Lookup Decoding (PLD)** extracts candidates at low draft cost. The paired N=150 benchmark measures **1.098×** mean speedup with zero auxiliary weight memory.
- **Compute-Bound Regimes (Batch Size > 1 or Datacenter Accelerators)**:
  Parallel target verification of $K+1$ candidate tokens incurs nearly identical latency to a 1-token step on wide matrix execution units, maximizing speculation efficiency.

### 2. When it Doesn't (The Mobile Bandwidth Ceiling)
- **Batch-1 Generation on Memory-Bandwidth-Bound Laptop GPUs (RTX 4050, 192 GB/s)**:
  For 3B models at $B=1$, single-token forward passes take $\approx 23\text{ ms}$, bound strictly by reading weights from GDDR6 memory.
  A speculative cycle requires **two separate memory sweeps**:
  1. Draft pass ($14–22\text{ ms}$, measured per config)
  2. Verify pass ($24–27\text{ ms}$)

  Total cycle time is $38–45\text{ ms}$. To break even ($1.0\times$), the cycle must emit $\ge 1.8$ tokens per step, requiring acceptance rate $\alpha \ge 75\%-80\%$ and draft compute ratio $c \le 0.50$.
  On CKA whole-layer skipping ($c \approx 0.82$ measured), the draft pass is too expensive, resulting in $0.85–0.89\times$ throughput — even the **official KnapSpec code** reproduces this on our hardware (0.86x at 97.8% acceptance, Llama-3.2-1B).

---

## Research Story: Feasibility Boundary, Roofline, Cost-Aware Router

This project is organized around three axes (supporting components below are secondary):

1. **Feasibility Boundary.** High acceptance does not imply speedup. We characterize the $(c, \alpha, K)$ boundary where speculation turns unprofitable and show the controller must be allowed *not* to speculate (VANILLA action with a feasibility gate: +0.083x in ablation, 1.083x gate A/B on GPU).
2. **Speculative Roofline.** Speedup predicted from draft-cost ratio $c$, acceptance $\alpha$, horizon $K$ (in-device calibration MAPE 2.4% on RTX 4050; cross-GPU rows are labeled projections with assumed efficiency factors, pending second-device calibration).
3. **Cost-Aware Hybrid Router.** Per-cycle choice over {Vanilla AR, PLD, Layer-Skip SSD} using measured latencies and online EMA acceptance. In the equal-horizon N=150 comparison it reaches 1.090× versus pure PLD's 1.098×; this is a measured negative result for the router's incremental benefit.

---

## Key Systems Engineering

1. **Static Pre-allocated KV Cache (`StaticPreallocatedKVCache`)**:
   Eliminates all `torch.cat` reallocations. Pre-allocates fixed memory buffers on GPU. Draft writes candidate tokens in-place to $[P : P+K)$; verification overwrites canonical states; rollback is an $O(1)$ pointer truncation.
2. **Vectorized GPU Decision Engine**:
   Eliminates per-token `.item()` calls and CPU-GPU stream synchronizations during draft and verification loops. Computes candidate matches batched on GPU tensors.
3. **Representational Middle-Skip Discovery**:
   Skipping middle layers (`mid_12`, layers 12–23) preserves early syntactic representations and late logit projections, boosting acceptance rate from **$52.9\%$ (CKA-75) to $73.8\% - 81.4\%$**.
4. **Prompt Lookup Decoding Baseline (`prompt_lookup_generate`)**:
   Provides training-free, zero-VRAM n-gram candidate extraction for context-heavy generation.
5. **Feasibility-Gated Controller (`HardwareAwareJointController`)**:
   Action space {VANILLA} U S x K. When every speculative action is predicted slower than vanilla, the controller emits K=0 (plain AR step) instead of knowingly running below 1.0x.

---

## Quickstart, Unified CLI, and One-Command Reproduction

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

### 4. Regenerate Manuscript Tables and Macros

```bash
zassd generate-tables
```

### 5. Reproduce All Results (CPU-Quick or Full GPU)

```bash
bash reproduce.sh --quick   # CPU-only: tests, roofline, router sims, stats (~2 min)
bash reproduce.sh           # + full GPU benchmark suite on RTX 4050
```

A pinned `Dockerfile` (PyTorch 2.11 / CUDA 12.8) is provided for artifact evaluation.

To reproduce the corrected same-GPU comparison and inspect a first divergence:

```bash
python scripts/run_strong_baseline_comparison.py --samples-per-bench 50
python scripts/trace_first_divergence.py --benchmark data/benchmarks/humaneval_eval.jsonl --ids HumanEval/8 --method routed --hash-kv
```

---

## Experimental Results

Standardized benchmark on **NVIDIA GeForce RTX 4050 Laptop GPU** (6 GB GDDR6, 192 GB/s, 80W TGP):

### Qwen2.5-3B-Instruct (36 Layers, 4-bit NF4)

New paired comparison: N=50 prompts each from GSM8K, HumanEval, and CNN/DailyMail;
Qwen2.5-3B-Instruct NF4; 48-token limit; identical PLD horizon K=4; a fresh
router per prompt; rotated method order; exact match by generated token IDs.
Raw per-prompt rows and bootstrap confidence intervals are in
[`experiments/diagnostics/strong_baselines_n50.json`](experiments/diagnostics/strong_baselines_n50.json).

| Method | Mean speedup vs vanilla | 95% bootstrap CI | Exact token-sequence match |
| :--- | ---: | :---: | ---: |
| PLD K=4 (vanilla on lookup miss) | **1.098x** | [1.070, 1.129] | 83.3% |
| PLD cost gate K≤4 | 1.089x | [1.061, 1.118] | 83.3% |
| Routed PLD / layer skip / vanilla | 1.090x | [1.062, 1.118] | 83.3% |

The paired routed/PLD throughput ratio is **0.993** (95% bootstrap CI
[0.988, 0.999]); the router is slower than pure PLD on this workload. PLD
already falls back to vanilla on lookup misses, so the separate cost-gated
baseline tests whether skipping some *hits* helps.

Earlier benchmark files in `experiments/final_validation/` and
`experiments/16_hybrid_roofline/` used mixed sample sizes or unequal PLD
horizons; their speedups and text-based exact-match fields should not be
compared directly with this table. The official KnapSpec implementation on
this GPU (Llama-3.2-1B FP16, AIME24, N=8) measured 0.86x at 97.8%
acceptance (`experiments/18_official_baselines/`).

Under NF4, batched target verification can write KV values that differ
bytewise from single-token vanilla evaluation even when no candidate is
accepted and cache lengths agree. Targeted first-divergence traces for
HumanEval/8 and CNN/DM 001/006 are in `experiments/diagnostics/`. This
implementation does not claim bitwise lossless greedy output under NF4.
An exploratory margin-triggered canonical replay guard selected threshold 1.5
on a 30-prompt dev split and reached 100% exact token match on 30 held-out
prompts, but ran at 0.35x vanilla and 0.33x PLD throughput. It is a measured
correctness/latency tradeoff, not an acceleration result; the full sweep is in
`experiments/diagnostics/nf4_margin_guard_study_n10.json`.

### PLD per-cycle and candidate-source study

Run a dev-tuned K/source ablation and paired test report with:

```bash
python scripts/run_pld_adaptive_study.py --samples-per-split 10 --max-new-tokens 32 --max-k 4
python scripts/analyze_pld_adaptive_study.py
```

The RTX 4050 exploratory run selected K=2 for history, prompt-only, and
prompt-plus-generated on each task. Prompt-plus-generated versus prompt-only
had paired speed ratios of 1.003 (GSM8K), 1.004 (HumanEval), and 1.021
(CNN/DailyMail); only the last small-split interval barely excluded 1.0.
Adaptive K did not consistently beat the source-matched, dev-tuned fixed-K
baseline. The original `history` lookup already searches both prompt and
generated tokens, so this ablation does not make generated-prefix reuse a new
contribution. Cycle costs and per-prompt losses are in
`experiments/diagnostics/pld_adaptive_source_analysis.json`; raw rows are in
`experiments/diagnostics/pld_adaptive_n10_source_matched.json`. These are
exploratory single-GPU results, not publication-strength validation. REST
retrieves draft text from an external corpus, a different method family
([NAACL 2024 paper](https://aclanthology.org/2024.naacl-long.88/)).

---

## Invariant Validation and System Test Suite

All 81 unit and systems tests run under `pytest`:

```bash
pytest tests/ -v
```

Verified Invariants (regression-tested, not formal-methods proofs):
1. **Invariant 1 (Zero-Copy Prefix Sharing)**: Pointer aliasing between target and ephemeral draft cache.
2. **Invariant 2 (Draft-Private Mutation)**: In-place draft writes into candidate slots leave canonical prefix memory bitwise identical.
3. **Invariant 3 (Exact Rollback)**: Truncation resets sequence length to $P + \text{accepted}$ with zero residual token pollution.
4. **Invariant 4 (Zero Dynamic Memory Allocation)**: Pre-allocated buffer addresses remain strictly constant across all forward steps.
5. **Numerical Equivalence Audit**: 4000-position single-vs-batched logit comparison on 3B models
   (99.4–99.8% token agreement, zero flip-bound violations) plus a strict FP32
   token-identity proof (100% over 10,113 positions on 0.5B/1B models).
6. **Algorithmic vs numerical exactness** are reported separately: the
   speculation algorithm is distribution-preserving under exact arithmetic;
   residual NF4 divergences are quantified kernel noise.

---

## Repository Structure

```
zero_additional_vram/
├── src/zassd/                   # Core Python package
│   ├── baselines/               # KnapSpec, SpecBound and Prompt Lookup Decoding
│   ├── cache/                   # TargetKVCache, StaticPreallocatedKVCache
│   ├── controllers/             # HardwareAwareJointController (+vanilla gate) and AdaptiveK
│   ├── decoding/                # self_speculative_generate with GPU-side decision
│   ├── layer_selection/         # Redundancy ranking and middle-skip selection
│   ├── modeling/                # Speculative roofline model
│   ├── models/                  # ModelAdapter and LayerManager
│   ├── profiling/               # Cost models, GPU/energy/latency profilers
│   └── routing/                 # Cost-aware HybridDraftRouter
│   └── cli.py                   # Unified CLI runner (zassd command)
├── paper/                       # Manuscript, figures, and LaTeX tables
│   ├── draft/                   # main.tex (compiles to 8-page PDF) + references
│   ├── figures/                 # Feasibility map, roofline, context sweep, ablations
│   └── tables/                  # Auto-generated .tex tables and macros
├── scripts/                     # Benchmarking, profiling, analysis, and plotting scripts
├── experiments/                 # Empirical raw results and summaries
│   ├── 16_hybrid_roofline/      # Router, roofline, oracle, ablation, sweeps
│   ├── 17_fp16_exactness/       # FP32/FP16 identity proofs and precision ladder
│   ├── 18_official_baselines/   # Official KnapSpec code results on this GPU
│   ├── 19_numerical_audit_scaled/# 4000-position 3B NF4 audit
│   └── final_validation/        # Frozen unified benchmark
├── data/benchmarks/             # GSM8K / HumanEval / CNN-DM splits (incl. extensions)
├── tests/                       # Invariant-based system test suite (81 tests)
├── Dockerfile                   # Pinned artifact container (PyTorch 2.11 / CUDA 12.8)
└── reproduce.sh                 # One-command reproduction (CPU-quick or full GPU)
```

## License

MIT License.
