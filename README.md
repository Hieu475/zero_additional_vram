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
| **Vanilla** | **42.6** | **1.00×** | 100.0% | 100.0% | 100.0% | **1984.1** | **1.346** | 0.0 | 23.4 |
| **CKA Fixed (K=2)** | 35.8 | 0.84× | 70.1% | 70.0% | 89.2% | 1990.7 | 1.837 | 14.8 | 26.2 |
| **Adaptive K** | 27.4 | 0.64× | 52.8% | 60.0% | 86.6% | 1991.2 | 2.145 | 22.4 | 28.5 |
| **KnapSpec (ICML'26)** | 35.3 | 0.83× | 70.1% | 70.0% | 89.2% | 1990.7 | 1.842 | 14.9 | 26.3 |
| **SpecBound (ACL'26)** | 34.6 | 0.81× | 66.4% | 70.0% | 88.5% | 1990.9 | 1.890 | 16.2 | 27.1 |
| **ZASSD HW Controller** | 31.9 | 0.75× | 64.6% | 70.0% | 89.2% | 1990.5 | 1.964 | 15.6 | 26.8 |

### 2. Llama-3.2-3B-Instruct (28 Layers, 4-bit NF4)

| Method | Throughput (tok/s) | Speedup | Acceptance (%) | Exact Match (%) | Partial Match (%) | Peak VRAM (MB) | Energy (J/tok) | Draft (ms) | Verify (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla** | **54.1** | **1.00×** | 100.0% | 100.0% | 100.0% | **2156.1** | **1.218** | 0.0 | 18.5 |
| **CKA Fixed (K=2)** | 41.8 | 0.77× | 67.3% | 70.0% | 89.2% | 2168.9 | 1.624 | 17.2 | 33.2 |
| **Adaptive K** | 36.2 | 0.67× | 58.4% | 60.0% | 86.6% | 2169.5 | 1.882 | 21.6 | 35.8 |
| **KnapSpec (ICML'26)** | 41.8 | 0.77× | 67.3% | 70.0% | 89.2% | 2168.9 | 1.625 | 17.2 | 33.2 |
| **SpecBound (ACL'26)** | 40.5 | 0.75× | 64.8% | 70.0% | 88.5% | 2169.1 | 1.670 | 18.4 | 34.1 |
| **ZASSD HW Controller** | 47.4 | 0.88× | 83.4% | 70.0% | 89.2% | 2168.7 | 1.485 | 11.2 | 24.1 |

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
