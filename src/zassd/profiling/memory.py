"""Memory profiling utilities."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_vram_usage(device: str = "cuda:0") -> dict:
    """Get current VRAM usage from PyTorch."""
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


def reset_vram_stats(device: str = "cuda:0") -> None:
    """Reset VRAM peak tracking."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()


def log_vram_usage(device: str = "cuda:0", prefix: str = "") -> None:
    """Log current VRAM usage."""
    usage = get_vram_usage(device)
    logger.info(
        f"{prefix}VRAM: allocated={usage['allocated_mb']:.1f}MB, "
        f"reserved={usage['reserved_mb']:.1f}MB, "
        f"peak={usage['max_allocated_mb']:.1f}MB"
    )
