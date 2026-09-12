"""Adaptive draft length (K) controller.

Adjusts the number of draft tokens based on:
- Token entropy (confidence)
- Historical acceptance rate
"""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveKController:
    """Controls draft length K adaptively."""

    def __init__(
        self,
        k_min: int = 1,
        k_max: int = 8,
        initial_k: int = 4,
        entropy_low: float = 0.5,
        entropy_high: float = 2.0,
        acceptance_target: float = 0.7,
        history_window: int = 20,
    ) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.current_k = initial_k

        self.entropy_low = entropy_low
        self.entropy_high = entropy_high
        self.acceptance_target = acceptance_target

        self.acceptance_history: deque[float] = deque(maxlen=history_window)

    def update(
        self,
        entropy: float,
        accepted: int,
        proposed: int,
    ) -> int:
        """Update K based on latest observation.

        Args:
            entropy: Current token entropy.
            accepted: Number of accepted tokens in last iteration.
            proposed: Number of proposed tokens in last iteration.

        Returns:
            Updated K value.
        """
        # Record acceptance rate
        rate = accepted / max(proposed, 1)
        self.acceptance_history.append(rate)

        avg_rate = sum(self.acceptance_history) / len(self.acceptance_history)

        # Adjust K
        if entropy < self.entropy_low and avg_rate > self.acceptance_target:
            # High confidence, good acceptance -> increase K
            self.current_k = min(self.current_k + 1, self.k_max)
        elif entropy > self.entropy_high or avg_rate < self.acceptance_target * 0.7:
            # Low confidence or poor acceptance -> decrease K
            self.current_k = max(self.current_k - 1, self.k_min)

        return self.current_k

    def reset(self) -> None:
        """Reset controller state."""
        self.acceptance_history.clear()

    @property
    def avg_acceptance_rate(self) -> float:
        if not self.acceptance_history:
            return 0.0
        return sum(self.acceptance_history) / len(self.acceptance_history)
