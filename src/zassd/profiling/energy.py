"""Energy measurement utilities."""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class EnergyTracker:
    """Track GPU energy consumption using power sampling."""

    def __init__(self, gpu_profiler=None, sampling_interval_s: float = 0.1) -> None:
        self.gpu_profiler = gpu_profiler
        self.sampling_interval = sampling_interval_s
        self.power_samples: list[tuple[float, float]] = []  # (timestamp, watts)

    def sample(self) -> Optional[float]:
        """Take a single power sample."""
        if self.gpu_profiler is None:
            return None
        power = self.gpu_profiler.get_power_usage()
        self.power_samples.append((time.perf_counter(), power))
        return power

    def compute_energy_joules(self) -> float:
        """Compute total energy from power samples using trapezoidal integration."""
        if len(self.power_samples) < 2:
            return 0.0

        total_energy = 0.0
        for i in range(1, len(self.power_samples)):
            dt = self.power_samples[i][0] - self.power_samples[i - 1][0]
            avg_power = (self.power_samples[i][1] + self.power_samples[i - 1][1]) / 2
            total_energy += avg_power * dt

        return total_energy

    def joules_per_token(self, num_tokens: int) -> float:
        """Compute energy per token."""
        if num_tokens == 0:
            return 0.0
        return self.compute_energy_joules() / num_tokens

    def reset(self) -> None:
        """Clear samples."""
        self.power_samples.clear()
