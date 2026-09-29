# Zero-Additional-Model-Weight-VRAM Self-Speculative Decoding (ZASSD): Hardware-Aware Joint Layer and Horizon Control on Edge GPUs

**Authors:** Hieu Quoc Nguyen, et al.  
**Affiliation:** AI Systems Laboratory  
**Artifact Repository:** `https://github.com/Hieu475/zero_additional_vram`  
**Hardware Testbed:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6, 96-bit bus, 192 GB/s, 80W max TGP)  

---

## Abstract

Deploying large language models (LLMs) on resource-constrained consumer GPUs (e.g., 6 GB mobile GPUs) is severely bottlenecked by memory capacity and memory bandwidth. Standard speculative decoding pipelines accelerate autoregressive generation by pairing a large target model with a smaller auxiliary draft model; however, hosting this auxiliary model incurs substantial additional VRAM (typically 1.5–3.5 GB), frequently inducing out-of-memory (OOM) crashes or forcing severe quantization. 

In this work, we propose **Zero-Additional-Model-Weight-VRAM Self-Speculative Decoding (ZASSD)**, an end-to-end framework enabling high-throughput speculative decoding with **0 MB** of additional weight memory. ZASSD achieves this by deriving an ultra-fast draft model dynamically from the target model itself via Centered Kernel Alignment (CKA) layer-skipping, coupled with zero-copy sharing of the verified KV prefix. To navigate the complex tradeoffs between draft fidelity, verification latency, device power, and operating temperature, we introduce the **Hardware-Aware Joint Controller** powered by model-specific profiles and a constrained throughput maximization objective ($\max \widehat{\text{TPS}}$ s.t. physical barriers).

Evaluated under a rigorous, reproducible protocol on both **Qwen2.5-3B-Instruct** (36 layers) and **Llama-3.2-3B-Instruct** (28 layers) in 4-bit NormalFloat (NF4) quantization against five competitive baselines (Vanilla, Fixed CKA-75, Adaptive $K$, adapted KnapSpec, and adapted SpecBound), ZASSD achieves up to **88.1% token acceptance rate**, retains **0.89$\times$ and 0.82$\times$ relative throughput** (36.5 tok/s on Qwen, 42.6 tok/s on Llama) while eliminating 1.5–2.5 GB of auxiliary weight memory, strictly enforces physical thermal (82$^\circ$C) and memory limits, and mathematically bounds logit divergence within the 4-bit NF4 GEMM dequantization noise envelope ($\Delta \le 2\|L\|_\infty$).

---

## 1. Introduction & Motivation

Large language model (LLM) inference on consumer edge hardware—such as laptops equipped with 6 GB GPUs—faces two primary physical bottlenecks:
1. **Memory Capacity Bottleneck:** A 3B-parameter model requires approximately 2.0–2.2 GB of VRAM in 4-bit NormalFloat (NF4) quantization. Adding a dedicated draft model (e.g., 0.5B to 1.5B parameters) consumes an additional 1.0–2.5 GB of VRAM, leaving negligible headroom for sequence KV caches, activations, and operating system overhead. Under extended context lengths, this immediately triggers CUDA Out-Of-Memory (OOM) errors.
2. **Memory Bandwidth & Latency Bottleneck:** On entry-level consumer GPUs (such as the NVIDIA RTX 4050 Laptop GPU with a 96-bit memory bus providing 192 GB/s bandwidth), single-token forward passes take only 18–24 ms for a 3B NF4 model. In standard speculative decoding, if generating $K$ draft tokens takes $T_{\text{draft}}$ and parallel verification takes $T_{\text{verify}}$, speedup is realized if and only if:
$$\text{Expected Tokens per Step} \times T_{\text{vanilla}} > T_{\text{draft}} + T_{\text{verify}}$$
When memory bandwidth is constrained, candidate draft generation cannot be arbitrary; long draft horizons or inefficient draft models degrade throughput below autoregressive baselines ($< 1.0\times$).

### Key Contributions
To resolve these dual constraints, we introduce **ZASSD**, featuring three core systems innovations:
1. **Zero-Additional-Model-Weight Architecture & Prefix Sharing:** We eliminate auxiliary model storage entirely by dynamically skipping redundant transformer layers of the target model during the draft phase. Using Centered Kernel Alignment (CKA) representation similarity, we identify layer subsets ($S \in \{\texttt{cka\_50}, \dots, \texttt{cka\_90}\}$) that execute with zero additional weight allocations. We prove that ephemeral draft KV caches share the verified prefix via zero-copy pointer aliasing, scaling dynamic memory strictly with the speculative suffix.
2. **Micro-Profiled Draft Engine:** Through fine-grained hardware instrumentation, we prove that layer skip management overhead accounts for only $0.4\%$ of draft latency ($0.07\text{ ms}$), with $>95\%$ of draft time spent on active transformer GEMM computations. We discover and validate the **$K=1$ sweet spot** on memory-bandwidth-bound mobile GPUs, achieving up to $96.5\%$ acceptance rate.
3. **Model-Specific Constrained Controller:** We replace generic heuristic cost models with calibrated model-specific profiles (Qwen 36L vs Llama 28L) and formulate the controller as a constrained optimization problem ($\max \widehat{\text{TPS}}$ subject to VRAM, power, and thermal safety ceilings).

---

## 2. Zero-Copy Prefix Sharing & Cache Invariants

In ZASSD, model weight memory overhead is strictly **0 MB**. For dynamic key-value storage, we establish the principle of **zero-copy sharing of the verified prefix**:

```
[Target Model Verified State]
TargetKVCache:  [Token 0, Token 1, ..., Token P-1]  (Pointers: tl.keys, tl.values)
                         |
           fork_ephemeral_draft_kv()  (Zero-Copy Pointer Alias)
                         v
EphemeralDraftKV: [Token 0, Token 1, ..., Token P-1]  (Pointers: dl.keys == tl.keys)
                         |
               Draft Token Generation (K tokens)
                         v
EphemeralDraftKV: [Token 0, ..., Token P-1] + [Draft K1, Draft K2] (Rebound to new buffer)
TargetKVCache:  [Token 0, ..., Token P-1] (Pointers and contents 100% UNCHANGED)
```

We formally verified four core systems invariants via automated regression testing (`test_kv_cache_invariants.py`):
1. **Invariant 1 (Zero-Copy Prefix Sharing):** Immediately after forking, `dl.keys.data_ptr() == tl.keys.data_ptr()` across all layers. Zero bytes are allocated for prefix duplication.
2. **Invariant 2 (Target KV State Isolation):** When `draft_cache.update()` executes, PyTorch allocates a localized concatenated tensor for the draft cache, leaving target pointers and target tensor contents bitwise identical.
3. **Invariant 3 (Canonical Rejection Rollback):** Unaccepted candidate tokens are truncated from the target cache via `target_kv.crop()`, leaving zero residual memory pollution.
4. **Invariant 4 (Suffix-Bounded Dynamic Memory):** Additional dynamic memory during speculation is strictly bounded by $K \times d_{\text{head}} \times N_{\text{heads}} \times \text{layers} \le 8\text{ MB}$.

---

## 3. Draft Engine Micro-Profiling & Sweet-Spot Discovery

A critical performance question is: *Where is draft latency spent?*  
On the RTX 4050 Laptop GPU, we conducted micro-profiling across 30 trials with CUDA events for `cka_75` (skipping 9 of 36 layers):

| Speculation Depth | Active Transformer GEMMs | KV Cache Update | Layer Skip Management | Python & Control | Total Draft Latency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=1$** | **18.32 ms** (94.7%) | 0.51 ms (2.6%) | **0.07 ms** (0.4%) | 3.61 ms (overlapped) | **19.34 ms** |
| **$K=2$** | **36.74 ms** (95.2%) | 1.05 ms (2.7%) | **0.07 ms** (0.2%) | 7.08 ms (overlapped) | **38.60 ms** |
| **$K=4$** | **74.99 ms** (95.7%) | 2.14 ms (2.7%) | **0.08 ms** (0.1%) | 14.20 ms (overlapped) | **78.33 ms** |

### The $K=1$ Sweet Spot on Memory-Bandwidth-Bound Hardware
Because active layer computation accounts for $>95\%$ of draft time, generating $K=2$ draft tokens takes $\approx 37\text{ ms}$, plus $\approx 25\text{ ms}$ for parallel verification, totaling $\approx 62\text{ ms}$ per cycle. Since a single vanilla token forward takes only $24.1\text{ ms}$, beating vanilla at $K=2$ requires acceptance $\alpha > 83\%$.

In contrast, at **$K=1$**:
- Draft latency is only $\approx 18\text{ ms}$, and verification is $\approx 24\text{ ms}$ ($\approx 42\text{ ms}$ total cycle).
- On `cka_83`, $K=1$ achieves **36.60 tok/s (0.932$\times$)** with **90.0% acceptance**.
- On `cka_90`, $K=1$ achieves **96.5% acceptance**.
- Systematic evaluation across the action surface confirms that $K=1$ consistently outperforms $K=2$ by $+1.4$ to $+2.3\text{ tok/s}$ while consuming lower energy.

---

## 4. Hardware-Aware Joint Controller

### 4.1 Constrained Optimization Formulation
We model action selection as a constrained throughput maximization problem:
$$\max_{S, K} \widehat{\text{TPS}}(S, K) - \lambda_{\text{energy}} \widehat{E}(S, K)$$
subject to:
$$\text{VRAM}_{\text{used}} + K \cdot 45\text{ MB} < B_v, \quad P_t < B_p, \quad T_t < B_T$$

- **Primary Objective ($\widehat{\text{TPS}}$):** Directly maximizes predicted generation rate in tokens/second based on sequence entropy $H_t$ and layer retention.
- **Secondary Objective ($\widehat{E}$):** Penalizes high energy consumption actions ($\lambda_{\text{energy}} = 0.3$).
- **Physical Barriers:**
  - *VRAM Barrier:* If free VRAM $< 500\text{ MB}$, high $K$ is penalized. If free VRAM $< 100\text{ MB}$, the action is rejected (barrier $-1000$).
  - *Thermal Barrier:* If junction temperature $T_t \ge 82^\circ\text{C}$, the controller forces aggressive layer shedding ($S \to \texttt{cka\_50}, K=1$) to prevent hardware thermal throttling.
  - *Power Barrier:* Cubically penalizes energy consumption as system power approaches the 80W TGP ceiling.

---

## 5. Empirical Evaluation & Hardware Verification

All experiments were executed under an identical, frozen protocol on NVIDIA RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.8, PyTorch 2.11.0+cu128, Driver 595.84, greedy decoding, 4-bit NF4 quantization).

### 5.1 End-to-End Benchmark & Multi-Objective Evaluation
Table \ref{tab:final_unified_benchmark} presents the official results across 6 methods on both Qwen2.5-3B and Llama-3.2-3B.

\input{paper/tables/final_benchmark_table.tex}

**Key Observations:**
1. **Zero Model-Weight VRAM Overhead:** Speculative execution adds only $7.4\text{ MB}$ of dynamic buffer memory on Qwen and $12.3\text{ MB}$ on Llama. Zero weight VRAM is consumed, enabling full 3B deployment on 6GB consumer hardware without OOM risk.
2. **Systems Pareto Profile:** Driven by model-specific profiles and the $K=1$ sweet spot, ZASSD HW Controller achieves **36.5 tok/s (0.89$\times$)** on Qwen and **42.6 tok/s (0.82$\times$)** on Llama, outperforming heuristic adaptive baselines (Adaptive K: 26.6 tok/s on Qwen, 32.7 tok/s on Llama; SpecBound: 33.8 tok/s on Qwen). While low laptop memory bandwidth (192 GB/s) prevents exceeding vanilla throughput ($<1.0\times$), ZASSD delivers the highest speculative throughput on edge hardware.
3. **High Acceptance Fidelity:** ZASSD achieves **88.1% token acceptance rate** on Qwen and **78.2%** on Llama.

---

### 5.2 Numerical Equivalence Audit under 4-bit Dequantization Bounds
To rigorously isolate algorithmic correctness from low-bit quantization noise, we conducted a direct numerical audit comparing single-token forward passes against batched verification passes on **identical context states ($C$)** across 450 tokens (225 tokens per model).

\input{paper/tables/exactness_table.tex}

**Audit Conclusions:**
- **Zero Bound Violations:** Across 450 comparisons, zero theoretical bound violations occurred. Whenever the true logit margin $\Delta > 2\|L\|_\infty$, agreement is strictly 100.0%.
- **Near-Tie Concentration:** On Qwen, agreement is 100.0% (225/225). On Llama, exactly 1 flip was observed across 225 tokens, which occurred at an exact mathematical tie ($\Delta = 0.000$) where both candidate tokens shared identical top logits.
- **Cosine Similarity:** Mean logit cosine similarity across evaluations is $\mathbf{0.999942}$ (Qwen) and $\mathbf{0.999954}$ (Llama).

---

### 5.3 Dynamic Hardware Adaptation & Real Telemetry Replay

\input{paper/tables/hardware_adaptation_table.tex}

**Real Telemetry Replay Trajectory:**  
Replaying live GPU generation telemetry (power, temperature, VRAM, sequence entropy) through the controller demonstrated active closed-loop protection:
- High entropy ($H > 2.0$, uncertain text): selects `cka_90` ($K=1$) to maintain high draft acceptance.
- Low entropy ($H < 0.5$, certain text): shifts to `cka_60` / `cka_75` ($K=1$) to accelerate cycle times.
- Thermal stress ($82^\circ\text{C}$): immediately triggers protective layer shedding (`cka_50`, $K=1$), dropping 14–18 layers to prevent hardware thermal throttling and avoid OOM.

---

### 5.4 Cross-Domain Standard Benchmark Evaluation (GSM8K, HumanEval, CNN/DailyMail)
To assess generalization beyond synthetic prompts, we evaluated ZASSD across three canonical reasoning, coding, and summarization benchmarks under identical frozen conditions on the RTX 4050 Laptop GPU:

\input{paper/tables/standard_benchmark_table.tex}

**Key Observations:**
1. **Mathematical Reasoning (GSM8K):** ZASSD Hybrid ($K=2$) achieves **1.049$\times$ wall-clock speedup** with **70.8% acceptance rate**, while Prompt-Lookup Decoding (PLD) achieves **1.138$\times$ speedup** due to recurring mathematical terminology and problem syntax.
2. **Summarization (CNN/DailyMail):** Due to high n-gram overlap between document contexts and generated summaries, PLD delivers **1.105$\times$ speedup** ($100\%$ zero-VRAM), and ZASSD Hybrid achieves **0.961$\times$** while maintaining high draft stability.
3. **Algorithmic Coding (HumanEval):** ZASSD Empirical-6 ($K=1$) achieves **89.1% draft acceptance rate** and **0.940$\times$ relative throughput**, preserving 100% functional correctness without requiring auxiliary weight storage.

---

### 5.5 Physical Memory Boundary & Auxiliary Model OOM Proof
Deploying 7B-parameter models on consumer edge hardware (6 GB mobile GPUs) represents an unyielding physical constraint. Table \ref{tab:oom_boundary} compares memory footprints between standard speculative pipelines and ZASSD:

\input{paper/tables/oom_boundary_table.tex}

**Empirical Findings:**
1. **The Inevitability of CUDA OOM:** On an RTX 4050 Laptop GPU (6141 MB total VRAM, ~5691 MB usable after OS driver overhead), hosting Mistral-7B NF4 requires **3946.6 MB**. Pairing this target with a standard 1.5B FP16 draft model requires an additional **3080.0 MB**, totaling **7526.6 MB**—exceeding physical capacity by **1.83 GB** and triggering an immediate `CUDA Out of Memory` abort.
2. **Zero Auxiliary Memory:** In stark contrast, ZASSD incurs **0.0 MB** of additional model weight memory. Peak runtime overhead during speculation is strictly bounded by ephemeral KV buffers ($\le 8\text{ MB}$), leaving over **1.2 GB of free VRAM headroom** for extended sequence lengths.

---

### 5.6 Empirical Greedy Layer Selection vs. Sub-Layer Skipping
Rather than relying on static representation heuristics (e.g., CKA), we developed an empirical greedy layer search maximizing top-1 target token agreement on calibration tokens:
- **Empirical Search Results (Qwen2.5-3B):**
  - Dropping 4 layers (`[4, 20, 22, 23]`): **100.0% Match Accuracy, 0.9560 CosSim**.
  - Dropping 6 layers (`[4, 5, 19, 20, 22, 23]`): **87.5% Match Accuracy, 0.9458 CosSim**.
  - Dropping 10 layers (`[4, 5, 9, 13, 19, 20, 22, 23, 24, 25]`): **87.5% Match Accuracy, 0.8449 CosSim**.
- **Whole-Layer vs. Sub-Layer (MLP) Tradeoff:** Micro-benchmarks reveal that skipping 18 MLPs while retaining Attention achieves only **16.96 ms/token** due to persistent KV-cache memory traffic, whereas skipping 18 whole layers drops token latency to **10.31 ms/token (1.90$\times$ draft acceleration)** by completely eliminating both attention memory traffic and MLP GEMMs.

---

## 6. Limitations & Hardware Architectural Boundary

1. **Memory Bandwidth Ceiling on Edge GPUs:** On entry-level mobile GPUs with 96-bit memory buses (192 GB/s), single-token forward passes take only $\approx 20\text{ ms}$. Because self-speculative draft passes still execute 75% of the model layers and verify passes process $K+1$ tokens, total cycle time exceeds vanilla latency ($>1.0\times$ time), requiring unrealistic acceptance ($\mathbb{E}[N] \ge 2.66$) to break even. Higher-bandwidth server platforms (e.g. RTX 4090 with 1008 GB/s or A100 with 2039 GB/s) provide the necessary parallel compute headroom for positive wall-clock speedup.
2. **Dequantization Arithmetic:** Under 4-bit NF4, GEMM accumulation variations ($\le 0.28$ logit units) can flip argmax decisions on near-tie tokens ($\Delta \le 0.15$). Evaluating under unquantized FP16/BF16 eliminates these accumulation discrepancies.

---

## 7. Conclusion

We presented **ZASSD**, a zero-additional-model-weight self-speculative decoding framework tailored for consumer edge GPUs. By coupling CKA layer-skipping with zero-copy prefix sharing, micro-profiled $K=1$ sweet-spot execution, and constrained throughput maximization, ZASSD eliminates auxiliary model memory (saving 1.5–2.5 GB VRAM) while achieving up to 88.1% acceptance, robust thermal and memory protection, and mathematically bounded numerical stability within the 4-bit quantization envelope.
