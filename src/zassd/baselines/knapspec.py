"""KnapSpec Baseline (ICML 2026).

Reference:
  Cha, Kim, Han, Yang, Han (2026).
  KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem.
  ICML 2026 (arXiv:2602.20217).

Formulation:
  Formulates draft subnetwork layer selection as a 0/1 Knapsack Problem:
    max sum_{l=1}^L v_l * x_l
    s.t. sum_{l=1}^L c_l * x_l <= B
    x_l in {0, 1}
  where:
    - v_l: Value of layer l based on representation cosine similarity / importance
    - c_l: Compute latency cost of layer l
    - B: Target latency / layer budget (e.g. 75% of full model)
    - x_l = 1 means layer l is kept in draft subnetwork, 0 means skipped.

KnapSpec uses fixed draft length K (typically K=2) and static/offline knapsack
layer selection without runtime hardware-state monitoring.

Note on Baseline Adaptation:
  The original KnapSpec (Cha et al., ICML 2026) decouples Attention and MLP submodules
  with length-dependent Tokens-Per-Time (TPT) dynamic optimization.
  This implementation provides a standardized whole-layer 0/1 Knapsack adaptation
  under identical layer-skipping infrastructure for fair baseline comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def solve_01_knapsack(values: list[float], costs: list[int], budget: int) -> list[int]:
    """Solve 0/1 Knapsack using dynamic programming to find optimal kept layer indices."""
    n = len(values)
    # Scale values to integers for DP table
    int_values = [int(round(v * 10000)) for v in values]

    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        c = costs[i - 1]
        v = int_values[i - 1]
        for w in range(budget + 1):
            if c <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - c] + v)
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack to identify selected items (kept layers)
    kept_indices = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            kept_indices.append(i - 1)
            w -= costs[i - 1]

    kept_indices.sort()
    return kept_indices


class KnapSpecController:
    """KnapSpec controller implementing 0/1 Knapsack layer selection (Cha et al., ICML 2026)."""

    def __init__(
        self,
        total_layers: int = 36,
        budget_ratio: float = 0.75,
        fixed_k: int = 2,
        ranking_file: str | Path = "experiments/03_cka/layer_redundancy_ranking.json",
        layer_ranks: Optional[list[int]] = None,
    ) -> None:
        self.total_layers = total_layers
        self.budget_ratio = budget_ratio
        self.k = fixed_k
        self.target_kept_count = int(round(total_layers * budget_ratio))

        # 1. Load layer redundancy / similarity scores
        layer_values = [1.0] * total_layers
        if layer_ranks is not None and len(layer_ranks) == total_layers:
            # layer_ranks is sorted most redundant to least redundant.
            # Least redundant layers have highest value.
            for rank_pos, l_idx in enumerate(layer_ranks):
                if 0 <= l_idx < total_layers:
                    layer_values[l_idx] = max(0.01, (rank_pos + 1) / float(total_layers))
        else:
            p = Path(ranking_file)
            if p.exists():
                try:
                    with open(p) as f:
                        ranking_data = json.load(f)
                    for item in ranking_data:
                        idx = item["layer_idx"]
                        if 0 <= idx < total_layers:
                            layer_values[idx] = max(0.01, item["rank"] / float(total_layers))
                except Exception as e:
                    logger.warning(f"Failed to load ranking data from {p}: {e}")


        # Uniform latency cost per layer: c_l = 1
        costs = [1] * total_layers

        # 2. Solve 0/1 Knapsack to select optimal kept layers
        self.kept_layers = solve_01_knapsack(layer_values, costs, self.target_kept_count)
        all_layers = set(range(total_layers))
        self.skip_indices = sorted(list(all_layers - set(self.kept_layers)))

        logger.info(
            f"KnapSpec initialized: budget={self.target_kept_count}/{total_layers} layers "
            f"({len(self.skip_indices)} skipped), K={self.k}"
        )

    def select_action(
        self,
        entropy: float,
        last_accepted: int = 1,
        last_proposed: int = 2,
        draft_ms: float = 20.0,
        verify_ms: float = 25.0,
    ) -> Any:
        """Select action adhering to the KnapSpec policy."""
        from zassd.controllers.hardware_controller import ControllerAction

        return ControllerAction(
            config_name="knapspec",
            skip_indices=self.skip_indices,
            draft_length=self.k,
            predicted_utility=0.0,
        )

    def update(self, entropy: float, accepted: int, proposed: int) -> int:
        """Fallback for AdaptiveK protocol."""
        return self.k
