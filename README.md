# Zero-Additional-VRAM Self-Speculative Decoding (ZASSD)

> Hardware-Aware Self-Speculative Decoding for Memory-Constrained LLM Inference on Consumer GPUs.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11](https://img.shields.io/badge/pytorch-2.11.0%2Bcu128-orange.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/cuda-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

Modern Large Language Models (LLMs) deployed on consumer edge hardware (such as laptop GPUs with 6 GB VRAM) face extreme memory and power constraints. Standard speculative decoding requires loading a separate draft model, exceeding consumer VRAM budgets.

**ZASSD (Zero-Additional-VRAM Self-Speculative Decoding)** solves this by using the target model checkpoint itself to perform self-speculation through logical layer-skipping, requiring **0 MB of additional model weight VRAM**.

### Key Architectural Pillars

1. **Zero-Copy KV Cache Reuse (`TargetKVCache` & `EphemeralDraftKV`)**:
   Maintains a canonical target KV cache and forks ephemeral draft KV caches with zero tensor copying, supporting $O(1)$ speculation cycles and exact rollback.
2. **CKA-Guided Layer Selection**:
   Identifies maximum representational redundancy in middle transformer layers using Centered Kernel Alignment (CKA) to construct draft networks training-free.
3. **Parametric Measured Action Cost Model**:
   Accurately predicts cycle latency ($T_{draft} + T_{verify}$), energy dissipation, and acceptance rates with hold-out error $< 5\%$.
4. **Hardware-Aware Joint Controller (`HardwareAwareJointController`)**:
   Jointly optimizes subnetwork depth $S_t$ and draft length $K_t$ conditioned dynamically on GPU VRAM headroom, operating power, thermal limits, and token entropy.
5. **Cross-Model Policy Transfer**:
   Empirically validated on both **Qwen2.5-3B-Instruct** (36 layers) and **Llama-3.2-3B-Instruct** (28 layers).

---

## 🖥️ Target Hardware & Evaluation Environment

All experiments are executed and verified on bare-metal hardware:
* **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6, 80W TGP)
* **Driver**: 595.84
* **CUDA**: 12.8
* **PyTorch**: 2.11.0+cu128
* **Quantization**: 4-bit NormalFloat (NF4 via `bitsandbytes`)
* **Precision / Decoding**: FP16 compute, greedy decoding (`temperature = 0.0`)

---

## 🔬 Benchmark Results across Competitive Baselines

Standardized benchmark across 6 decoding methods on RTX 4050 Laptop GPU under identical experimental conditions (`data/benchmarks/prompts.jsonl`):

### 1. Qwen2.5-3B-Instruct (36 Layers, 4-bit NF4)

| Method | Throughput (tok/s) | Speedup | Acceptance (%) | Exact Match (%) | Partial Match (%) | Peak VRAM (MB) | Energy (J/tok) | Draft (ms) | Verify (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla** | **41.7** | **1.00×** | 100.0% | 100.0% | 100.0% | **1983.8** | **1.384** | 0.0 | 24.0 |
| **CKA Fixed (K=2)** | 36.4 | 0.88× | 73.2% | 60.0% | 82.8% | 1991.2 | 1.755 | 37.6 | 26.3 |
| **Adaptive K** | 27.7 | 0.66× | 56.4% | 70.0% | 85.0% | 1992.2 | 2.155 | 66.7 | 36.0 |
| **KnapSpec (ICML'24)** | 35.9 | 0.86× | 73.2% | 60.0% | 82.8% | 1991.4 | 1.775 | 38.3 | 26.6 |
| **SpecBound (ACL'24)** | 34.4 | 0.83× | 74.1% | 50.0% | 79.4% | 1991.6 | 1.830 | 37.2 | 29.6 |
| **ZASSD HW Controller** | **38.6** | **0.93×** | **89.0%** | 60.0% | **82.8%** | 1991.2 | **1.625** | **19.4** | 25.8 |

### 2. Llama-3.2-3B-Instruct (28 Layers, 4-bit NF4)

| Method | Throughput (tok/s) | Speedup | Acceptance (%) | Exact Match (%) | Partial Match (%) | Peak VRAM (MB) | Energy (J/tok) | Draft (ms) | Verify (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla** | **54.2** | **1.00×** | 100.0% | 100.0% | 100.0% | **2158.5** | **1.321** | 0.0 | 18.5 |
| **CKA Fixed (K=2)** | 42.1 | 0.78× | 68.2% | 60.0% | 72.8% | 2171.0 | 1.810 | 28.9 | 22.5 |
| **Adaptive K** | 34.2 | 0.63× | 62.5% | 50.0% | 71.6% | 2173.2 | 2.092 | 40.4 | 33.4 |
| **KnapSpec (ICML'24)** | 41.0 | 0.76× | 68.2% | 60.0% | 72.8% | 2171.0 | 1.857 | 30.2 | 22.5 |
| **SpecBound (ACL'24)** | 43.3 | 0.80× | 76.4% | 50.0% | 70.9% | 2171.6 | 1.675 | 22.1 | 23.4 |
| **ZASSD HW Controller** | **44.5** | **0.83×** | **76.2%** | 60.0% | **72.8%** | 2170.8 | **1.619** | **15.3** | 20.8 |

---

## 🎯 Systems Findings & Scientific Takeaways

1. **Memory Bandwidth Verification Bottleneck on Mobile GPUs**:
   Under 4-bit quantization, batched verification passes of length $K+1$ process tokens through all layers. Because memory bandwidth on laptop GPUs (RTX 4050: 192 GB/s) is constrained, batched verify passes require 26–35 ms, compared to 18–23 ms for single-token vanilla generation. Speculative speedup $> 1.0\times$ requires acceptance rates exceeding $\alpha > 85\%$, achieved when $K=1$ on low-redundancy configurations (`cka_90`).
2. **Exactness vs. Quantization Noise Boundary**:
   Under greedy decoding ($T=0.0$), the target model's logits under speculative decoding have near-perfect cosine similarity ($\text{LogitCosine} = \mathbf{0.99990}$) with vanilla logits. Divergences occur **exclusively on near-tie tokens** where the top-1 vs top-2 logit margin is $\Delta \le 0.125$, induced by micro-variations in dequantized GEMM accumulation across different sequence tile shapes.

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
└── tests/                       # Complete unit and regression test suite
```

## 📜 License

MIT License.
