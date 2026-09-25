"""SpecBound Baseline (Findings of ACL 2026).

Reference:
  Wen & Feng (2026).
  SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration.
  Findings of ACL 2026 (arXiv:2604.12247).

Formulation:
  SpecBound dynamically bounds the speculation draft length K_t in [1, K_max]
  using calibrated confidence thresholds and sequence entropy:
    K_t = BoundedLookahead(H_t, A_{t-1})
  where:
    - High confidence (entropy < H_low): expands K_t towards K_max (e.g. K=4)
    - Medium confidence (H_low <= entropy <= H_high): bounds K_t = 2
    - Low confidence / high entropy (entropy > H_high): bounds K_t = 1 (early exit from speculation)

SpecBound operates with a fixed layer-skip subnetwork (e.g. CKA-75) and adapts
speculation length purely from linguistic uncertainty, without hardware constraints.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class SpecBoundController:
    """SpecBound adaptive draft length controller (Wen & Feng, ACL 2026)."""

    def __init__(
        self,
        skip_indices: list[int],
        k_min: int = 1,
        k_max: int = 4,
        initial_k: int = 2,
        entropy_low: float = 0.5,
        entropy_high: float = 1.4,
        anneal_rate: float = 0.95,
        history_window: int = 10,
    ) -> None:
        self.skip_indices = skip_indices
        self.k_min = k_min
        self.k_max = k_max
        self.current_k = initial_k

        self.entropy_low = entropy_low
        self.entropy_high = entropy_high
        self.anneal_rate = anneal_rate

        self.acceptance_history: deque[float] = deque(maxlen=history_window)
        self.k_history: list[int] = [initial_k]

    def select_action(
        self,
        entropy: float,
        last_accepted: int = 1,
        last_proposed: int = 2,
        draft_ms: float = 20.0,
        verify_ms: float = 25.0,
    ) -> Any:
        """Select action: fixed skip indices with calibrated bounded K."""
        from zassd.controllers.hardware_controller import ControllerAction

        # Update historical acceptance rate
        if last_proposed > 0:
            acc = float(last_accepted) / float(last_proposed)
            self.acceptance_history.append(acc)

        mean_acc = sum(self.acceptance_history) / max(1, len(self.acceptance_history))

        # SpecBound Bounded Speculation Policy:
        # High confidence & high historical acceptance -> extend K
        if entropy < self.entropy_low and mean_acc >= 0.70:
            k = min(self.k_max, self.current_k + 1)
        elif entropy > self.entropy_high or mean_acc < 0.35:
            # Low confidence -> bound K to minimum
            k = max(self.k_min, self.current_k - 1)
        else:
            k = self.current_k

        self.current_k = k
        self.k_history.append(k)

        return ControllerAction(
            config_name="specbound",
            skip_indices=self.skip_indices,
            draft_length=k,
            predicted_utility=0.0,
        )

    def update(self, entropy: float, accepted: int, proposed: int) -> int:
        """Direct update method."""
        action = self.select_action(entropy, accepted, proposed)
        return int(action.draft_length)
