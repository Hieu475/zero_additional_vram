# Zero-Additional-VRAM Self-Speculative Decoding (ZASSD)

> Hardware-Aware Self-Speculative Decoding for Memory-Constrained LLM Inference on Consumer GPUs.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11](https://img.shields.io/badge/pytorch-2.11.0%2Bcu128-orange.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/cuda-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

Modern Large Language Models (LLMs) deployed on consumer edge hardware (such as laptop GPUs with 6 GB VRAM) face extreme memory and power constraints. Standard speculative decoding requires loading a separate draft model (1B–3B parameters), consuming 1.5–3.0 GB of VRAM—exhausting 30% to 50% of the entire memory budget.

**ZASSD (Zero-Additional-VRAM Self-Speculative Decoding)** solves this by using the base model checkpoint itself to perform self-speculation through logical layer-skipping, requiring **strictly 0.00 MB of additional model weight VRAM**.

### Key Architectural Pillars

1. **Zero-Copy KV Cache Reuse (`TargetKVCache` & `EphemeralDraftKV`)**:
   Maintains a canonical target KV cache and forks ephemeral draft KV caches with zero tensor copying, supporting $O(1)$ speculation cycles and exact rollback.
2. **CKA-Guided Layer Selection**:
   Identifies maximum representational redundancy in middle transformer layers using Centered Kernel Alignment (CKA) to construct draft networks training-free.
3. **Draft Latency Decomposition (Phase 15.2)**:
   Rigorous microprofiling proves that **$96.2\% - 97.0\%$** of draft latency is active transformer GEMM compute. Python patching ($<0.4\%$) and DynamicCache overheads ($<2.9\%$) are mathematically negligible. Layer skipping directly attacks the true hardware bottleneck.
4. **Hardware-Aware Joint Controller (`HardwareAwareJointController`)**:
   Jointly optimizes subnetwork depth $S_t$ and draft length $K_t$ conditioned dynamically on GPU VRAM headroom, operating power, thermal limits, and token entropy with closed-loop telemetry.
5. **Cross-Model Policy Transfer & Model-Isolated Cost Profiles (Phase 15.3)**:
   Empirically validated with dedicated, non-overlapping cost profiles for **Qwen2.5-3B-Instruct** (36 layers) and **Llama-3.2-3B-Instruct** (28 layers).

---

## 🖥️ Target Hardware & Evaluation Environment

All experiments are executed and verified on bare-metal hardware:
* **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6, 80W TGP, 192 GB/s Bandwidth)
* **Driver**: 595.84 | **CUDA**: 12.8 | **PyTorch**: 2.11.0+cu128
* **Quantization**: 4-bit NormalFloat (NF4 via `bitsandbytes`)
* **Precision / Decoding**: FP16 compute, greedy decoding (`temperature = 0.0`)

---

## 🔬 Benchmark Results across Competitive Baselines

Standardized benchmark across 6 decoding methods on RTX 4050 Laptop GPU under identical experimental conditions (`data/benchmarks/prompts.jsonl`):

### 1. Qwen2.5-3B-Instruct (36 Layers, 4-bit NF4)

| Method | Throughput (tok/s) | Speedup | Acceptance (%) | Exact Match (%) | Partial Match (%) | Peak VRAM (MB) | Energy (J/tok) | Draft (ms) | Verify (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla** | **41.0** | **1.00×** | 100.0% | 100.0% | 100.0% | **1983.8** | **1.376** | 0.0 | 24.4 |
| **CKA Fixed (K=2)** | 35.7 | 0.87× | 73.2% | 60.0% | 82.8% | 1991.2 | 1.740 | 38.4 | 26.6 |
| **Adaptive K** | 26.6 | 0.65× | 56.4% | 70.0% | 85.0% | 1992.2 | 2.167 | 68.3 | 38.4 |
| **KnapSpec (ICML'24)** | 35.2 | 0.86× | 73.2% | 60.0% | 82.8% | 1991.4 | 1.818 | 39.0 | 27.1 |
| **SpecBound (ACL'24)** | 33.8 | 0.83× | 74.1% | 50.0% | 79.4% | 1991.6 | 1.853 | 37.4 | 30.5 |
| **ZASSD HW Controller** | **36.5** | **0.89×** | **88.1%** | 60.0% | **82.8%** | 1991.2 | **1.643** | **20.5** | 26.9 |

### 2. Llama-3.2-3B-Instruct (28 Layers, 4-bit NF4)

| Method | Throughput (tok/s) | Speedup | Acceptance (%) | Exact Match (%) | Partial Match (%) | Peak VRAM (MB) | Energy (J/tok) | Draft (ms) | Verify (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla** | **52.0** | **1.00×** | 100.0% | 100.0% | 100.0% | **2158.5** | **1.355** | 0.0 | 19.2 |
| **CKA Fixed (K=2)** | 39.5 | 0.76× | 68.2% | 60.0% | 72.8% | 2171.0 | 1.865 | 30.6 | 24.0 |
| **Adaptive K** | 32.7 | 0.63× | 62.5% | 50.0% | 71.6% | 2173.2 | 2.117 | 41.6 | 35.6 |
| **KnapSpec (ICML'24)** | 39.5 | 0.76× | 68.2% | 60.0% | 72.8% | 2171.0 | 1.879 | 30.4 | 24.1 |
| **SpecBound (ACL'24)** | 41.4 | 0.80× | 76.4% | 50.0% | 70.9% | 2171.6 | 1.741 | 23.1 | 24.5 |
| **ZASSD HW Controller** | **42.6** | **0.82×** | **78.2%** | 60.0% | **72.8%** | 2170.8 | **1.683** | **16.3** | 21.8 |

---

## 🎯 Systems Findings & Scientific Takeaways

1. **Zero Additional Model-Weight VRAM**:
   Standard speculative decoding requires loading a secondary draft model (e.g. 1B-3B parameters), consuming 1.5–3.0 GB of VRAM—exhausting 30% to 50% of the entire 6 GB memory budget on consumer laptop GPUs. ZASSD achieves self-speculation using only the base model checkpoint via dynamic layer bypassing, requiring **strictly 0.00 MB additional weight memory** and only +7.4 to +12.3 MB for dynamic ephemeral cache structures.
2. **Draft Latency Decomposition (Phase 15.2)**:
   Microprofiling on the RTX 4050 GPU directly decomposed:
   $$T_{\text{draft}} = T_{\text{transformer}} + T_{\text{KV}} + T_{\text{layer-mgmt}} + T_{\text{sampling}} + T_{\text{sync}}$$
   Measurements prove that **$96.2\% - 97.0\%$** of draft latency is active transformer GEMM compute over retained layers. Python context manager patching ($<0.1\text{ ms}$, $<0.4\%$) and DynamicCache tensor allocations ($<2.9\%$) are mathematically negligible. Pruning layers directly attacks the dominant physical compute bottleneck.
3. **Memory Bandwidth & Systems Trade-offs**:
   On memory-bandwidth bound mobile hardware (RTX 4050: 192 GB/s), batched verification across $K+1$ candidates requires 21–27 ms vs 19–24 ms for single-token vanilla generation. Speculative decoding incurs a dual pass (draft + verify) per cycle. The value of ZASSD is not an artificial claim that self-speculation is unconditionally faster than vanilla, but rather the multi-objective Pareto trade-off: **zero model-weight VRAM overhead**, **high draft acceptance ($78\% - 88\%$)**, **closed-loop thermal/power adaptation**, and **subnetwork flexibility**.
4. **Mathematical Bounding of 4-bit NF4 Quantization Noise (Phase 15.1)**:
   Across 450 comparisons on identical context states $C$, **0 theoretical bound violations** occurred. Argmax agreement is strictly 100.0% whenever logit margin $\Delta = z_{(1)} - z_{(2)} > 2\|L_{\text{single}} - L_{\text{batched}}\|_\infty$. Divergences occur exclusively on near-tie tokens within the dequantization noise margin ($\Delta \le 0.125$).
5. **Closed-Loop Hardware Controller & Model-Isolated Profiles (Phases 15.3–15.4)**:
   The controller operates on dedicated, non-overlapping cost profiles (`Qwen25_3B_CostProfile` and `Llama32_3B_CostProfile`). Under live closed-loop physical telemetry, the controller dynamically activates thermal protection (dropping 14–18 layers at $82^\circ\text{C}$) without any hardcoded constants.

---

## 🚀 Interactive Streaming Demo

Run the interactive live telemetry dashboard:

```bash
# Live interactive streaming demo with real-time UI:
python scripts/interactive_streaming_demo.py --max-new-tokens 64

# Custom prompt:
python scripts/interactive_streaming_demo.py --prompt "Explain speculative decoding in systems"
```

---

## 📂 Repository Structure

```
zero_additional_vram/
├── configs/                     # Model and hardware configurations
├── experiments/                 # Empirical experiment records & provenance
│   ├── 15_numerical_audit/      # Phase 15.1 Direct numerical equivalence audit
│   ├── 15_draft_profiling/      # Phase 15.2 Draft latency microprofiling decomposition
│   ├── 15_telemetry_replay/     # Phase 15.4 Closed-loop telemetry replay traces
│   ├── final_validation/        # Official unified validation dataset & plots
│   ├── 09_cost_validation/      # Holdout cost model generalization evidence
│   ├── 10_exactness/            # N>=20 prompt exactness & divergence logs
│   ├── 11_pareto/               # 4D Pareto frontier database
│   └── 12_generalization/       # Llama-3.2-3B cross-model transfer records
├── src/zassd/                   # Core Python package
│   ├── baselines/               # KnapSpec & SpecBound baseline implementations
│   ├── cache/                   # TargetKVCache & EphemeralDraftKV
│   ├── controllers/             # HardwareAwareJointController & AdaptiveK
│   ├── decoding/                # Self-speculative decoding pipeline
│   ├── layer_selection/         # CKA similarity & redundancy ranking
│   └── profiling/               # GPU NVML, energy & MeasuredActionCostModel
├── scripts/                     # Benchmark runners and interactive demo
└── tests/                       # Complete unit and regression test suite (56 tests)
```

## 📜 License

MIT License.
