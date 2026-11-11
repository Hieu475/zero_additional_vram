"""Environment validation script.

Checks that all required dependencies and hardware are properly configured.
"""

from __future__ import annotations

import platform
import sys

import torch
import transformers


def main() -> None:
    print("=" * 60)
    print("ZERO-ADDITIONAL-VRAM SELF-SPECULATIVE")
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"Python       : {sys.version.split()[0]}")
    print(f"Platform     : {platform.platform()}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"PyTorch CUDA : {torch.version.cuda}")
    print(f"Transformers : {transformers.__version__}")

    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device_count = torch.cuda.device_count()

    print(f"GPU count     : {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)

        total_gb = props.total_memory / (1024**3)

        print(f"\nGPU {i}")
        print(f"  Name       : {props.name}")
        print(f"  VRAM       : {total_gb:.2f} GB")
        print(f"  SM count   : {props.multi_processor_count}")
        print(f"  Compute cap: {props.major}.{props.minor}")

    print("\nEnvironment check PASSED.")


if __name__ == "__main__":
    main()
