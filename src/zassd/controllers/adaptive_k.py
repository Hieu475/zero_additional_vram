"""Adaptive draft length (K) controller.

Adjusts draft speculation length K dynamically based on:
- Token-level entropy H_t (confidence of prediction)
- Historical acceptance rate A_{t-1}
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F

from zassd.controllers.entropy import compute_entropy

logger = logging.getLogger(__name__)


class AdaptiveKController:
    """Controls draft length K adaptively using entropy and acceptance feedback."""

    def __init__(
        self,
        k_min: int = 1,
        k_max: int = 8,
        initial_k: int = 4,
        entropy_low: float = 0.6,
        entropy_high: float = 1.8,
        acceptance_target: float = 0.35,
        history_window: int = 10,
    ) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.current_k = initial_k

        self.entropy_low = entropy_low
        self.entropy_high = entropy_high
        self.acceptance_target = acceptance_target

        self.acceptance_history: deque[float] = deque(maxlen=history_window)
        self.entropy_history: deque[float] = deque(maxlen=history_window)
        self.k_history: list[int] = [initial_k]

    def get_k(self) -> int:
        """Return the current K value."""
        return self.current_k

    def update(
        self,
        entropy: float,
        accepted: int,
        proposed: int,
    ) -> int:
        """Update K based on token entropy and acceptance rate.

        Args:
            entropy: Entropy of current token distribution H_t.
            accepted: Number of accepted draft tokens in last cycle.
            proposed: Number of proposed draft tokens in last cycle (K).

        Returns:
            New draft length K for next speculation cycle.
        """
        rate = accepted / max(proposed, 1)
        self.acceptance_history.append(rate)
        self.entropy_history.append(entropy)

        avg_rate = sum(self.acceptance_history) / len(self.acceptance_history)

        # Decision logic:
        # 1. Low entropy (confident model) + acceptable acceptance rate -> aggressively increase K
        if entropy < self.entropy_low and avg_rate >= self.acceptance_target * 0.8:
            step = 2 if entropy < (self.entropy_low * 0.5) else 1
            self.current_k = min(self.current_k + step, self.k_max)

        # 2. High entropy (uncertain model) OR poor acceptance rate -> decrease K
        elif entropy > self.entropy_high or avg_rate < (self.acceptance_target * 0.6):
            step = 2 if entropy > (self.entropy_high * 1.5) else 1
            self.current_k = max(self.current_k - step, self.k_min)

        # 3. Moderate conditions: tune towards acceptance rate target
        elif avg_rate > self.acceptance_target * 1.2:
            self.current_k = min(self.current_k + 1, self.k_max)
        elif avg_rate < self.acceptance_target * 0.7:
            self.current_k = max(self.current_k - 1, self.k_min)

        self.k_history.append(self.current_k)
        return self.current_k

    def reset(self) -> None:
        """Reset controller state."""
        self.acceptance_history.clear()
        self.entropy_history.clear()
        self.k_history = [self.k_min]
        self.current_k = 4

    @property
    def avg_acceptance_rate(self) -> float:
        if not self.acceptance_history:
            return 0.0
        return sum(self.acceptance_history) / len(self.acceptance_history)

    @property
    def mean_k(self) -> float:
        if not self.k_history:
            return float(self.current_k)
        return sum(self.k_history) / len(self.k_history)
