"""GPU profiling script.

Collects detailed GPU metrics for hardware characterization.

Usage:
    python scripts/profile_gpu.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from zassd.profiling.gpu import GPUProfiler
from zassd.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    output_dir = Path("experiments/00_environment")
    output_dir.mkdir(parents=True, exist_ok=True)

    profiler = GPUProfiler(device_index=0)

    # Collect hardware info
    snapshot = profiler.snapshot()
    logger.info(f"GPU Snapshot: {json.dumps(snapshot, indent=2)}")

    # Save hardware record
    hardware_file = output_dir / "hardware.json"
    with open(hardware_file, "w") as f:
        json.dump(snapshot, f, indent=2)

    logger.info(f"Hardware info saved to {hardware_file}")

    # Continuous monitoring for 10 seconds
    logger.info("Monitoring GPU for 10 seconds...")
    samples = []
    start = time.perf_counter()
    while time.perf_counter() - start < 10:
        sample = profiler.snapshot()
        sample["timestamp"] = time.perf_counter() - start
        samples.append(sample)
        time.sleep(0.5)

    monitor_file = output_dir / "gpu_monitor.json"
    with open(monitor_file, "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Monitor data saved to {monitor_file}")


if __name__ == "__main__":
    main()
