"""GPU profiling using pynvml."""

from __future__ import annotations

import logging
from typing import Optional

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = logging.getLogger(__name__)


class GPUProfiler:
    """GPU profiling using NVIDIA Management Library."""

    def __init__(self, device_index: int = 0) -> None:
        if not HAS_PYNVML:
            raise ImportError("pynvml is required for GPU profiling")

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.device_index = device_index

        name = pynvml.nvmlDeviceGetName(self.handle)
        logger.info(f"GPU Profiler initialized for: {name}")

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "used_mb": info.used / (1024**2),
            "free_mb": info.free / (1024**2),
            "total_mb": info.total / (1024**2),
            "utilization_pct": info.used / info.total * 100,
        }

    def get_power_usage(self) -> float:
        """Get current power usage in watts."""
        power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
        return power_mw / 1000.0

    def get_utilization(self) -> dict:
        """Get GPU utilization rates."""
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return {
            "gpu_pct": util.gpu,
            "memory_pct": util.memory,
        }

    def get_temperature(self) -> float:
        """Get GPU temperature in Celsius."""
        return pynvml.nvmlDeviceGetTemperature(
            self.handle, pynvml.NVML_TEMPERATURE_GPU
        )

    def get_clock_speeds(self) -> dict:
        """Get current clock speeds."""
        return {
            "graphics_mhz": pynvml.nvmlDeviceGetClockInfo(
                self.handle, pynvml.NVML_CLOCK_GRAPHICS
            ),
            "memory_mhz": pynvml.nvmlDeviceGetClockInfo(
                self.handle, pynvml.NVML_CLOCK_MEM
            ),
        }

    def snapshot(self) -> dict:
        """Get complete GPU state snapshot."""
        return {
            "memory": self.get_memory_info(),
            "power_w": self.get_power_usage(),
            "utilization": self.get_utilization(),
            "temperature_c": self.get_temperature(),
            "clocks": self.get_clock_speeds(),
        }

    def __del__(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
