# Zero-Additional-VRAM Self-Speculative Decoding

> Zero-Additional-VRAM Self-Speculative Decoding for memory-constrained LLM inference on consumer GPUs.

## Overview

This research project implements and evaluates a **hardware-aware self-speculative decoding** framework that achieves inference speedup **without requiring any additional VRAM** beyond the target model itself.

### Key Contributions

1. **CKA-based Layer Selection**: Using Centered Kernel Alignment to identify and skip redundant layers for draft model construction
2. **Adaptive Draft Length (K)**: Entropy-based dynamic adjustment of speculation depth
3. **Hardware-Aware Joint Controller**: Runtime optimization under multiple hardware constraints (VRAM, latency, energy)

### Target Hardware

- NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- Consumer-grade deployment scenario

## Project Structure

```
zero_additional_vram/
├── configs/          # Experiment configurations
├── src/zassd/        # Main source code
├── scripts/          # Utility scripts
├── experiments/      # Experiment results (numbered)
├── data/             # Datasets and calibration data
├── results/          # Aggregated results
├── notebooks/        # Analysis notebooks
├── tests/            # Unit tests
└── paper/            # Paper drafts and figures
```

## Quick Start

```bash
# Create environment
conda create -n zassd python=3.11 -y
conda activate zassd

# Install PyTorch — use the official selector for your CUDA version:
# https://pytorch.org/get-started/locally/
# Actual versions used are recorded in experiments/00_environment/

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Check environment
python scripts/check_environment.py

# Download model
python scripts/download_model.py
```

## Experiment Pipeline

```
Environment → Vanilla baseline → Layer-skipping → CKA → Self-speculative → Adaptive K → Hardware controller → Optimization → Ablation
```

## License

MIT
