"""Latency measurement utilities."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import torch

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    """Record of a latency measurement."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_ms: float = 0.0
    cuda_elapsed_ms: float = 0.0


class LatencyTracker:
    """Track latency of operations with CUDA synchronization."""

    def __init__(self) -> None:
        self.records: list[LatencyRecord] = []

    @contextmanager
    def track(self, name: str) -> Generator[LatencyRecord, None, None]:
        """Context manager to track latency of an operation."""
        record = LatencyRecord(name=name)

        torch.cuda.synchronize()
        record.start_time = time.perf_counter()

        yield record

        torch.cuda.synchronize()
        record.end_time = time.perf_counter()
        record.elapsed_ms = (record.end_time - record.start_time) * 1000

        self.records.append(record)

    def summary(self) -> dict:
        """Get summary statistics."""
        if not self.records:
            return {}

        by_name: dict[str, list[float]] = {}
        for r in self.records:
            by_name.setdefault(r.name, []).append(r.elapsed_ms)

        result = {}
        for name, times in by_name.items():
            import numpy as np
            arr = np.array(times)
            result[name] = {
                "mean_ms": float(arr.mean()),
                "std_ms": float(arr.std()),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "count": len(times),
            }

        return result

    def reset(self) -> None:
        """Clear all records."""
        self.records.clear()
