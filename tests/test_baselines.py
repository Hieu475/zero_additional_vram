"""Tests for competitive baselines (KnapSpec and SpecBound)."""

from __future__ import annotations

import pytest

from zassd.baselines.knapspec import KnapSpecController, solve_01_knapsack
from zassd.baselines.specbound import SpecBoundController


class TestCompetitiveBaselines:
    """Validate KnapSpec and SpecBound implementations."""

    def test_solve_01_knapsack(self):
        values = [10.0, 20.0, 30.0, 40.0]
        costs = [1, 1, 1, 1]
        budget = 2
        selected = solve_01_knapsack(values, costs, budget)
        assert len(selected) == 2
        assert selected == [2, 3]  # Items with values 30 and 40

    def test_knapspec_controller_initialization(self):
        controller = KnapSpecController(total_layers=36, budget_ratio=0.75, fixed_k=2)
        assert len(controller.kept_layers) == 27
        assert len(controller.skip_indices) == 9
        assert controller.k == 2

        action = controller.select_action(entropy=1.0)
        assert action.config_name == "knapspec"
        assert action.draft_length == 2
        assert len(action.skip_indices) == 9

    def test_specbound_controller_adaptation(self):
        skips = [3, 4, 5, 6, 7, 10, 11, 12, 13]
        controller = SpecBoundController(skip_indices=skips, k_min=1, k_max=4, initial_k=2)

        # High confidence, high acceptance -> K increases
        for _ in range(5):
            action = controller.select_action(entropy=0.2, last_accepted=2, last_proposed=2)
        assert action.draft_length >= 2

        # Low confidence / high entropy -> K decreases
        for _ in range(10):
            action = controller.select_action(entropy=2.5, last_accepted=0, last_proposed=2)
        assert action.draft_length <= 2

    def test_prompt_lookup_candidate_matching(self):
        from zassd.baselines.prompt_lookup import find_candidate_tokens

        # Exact 3-gram match: [10, 20, 30] appears earlier followed by [40, 50, 60]
        tokens = [1, 2, 10, 20, 30, 40, 50, 60, 7, 8, 10, 20, 30]
        cands = find_candidate_tokens(tokens, ngram_size=3, max_candidates=3)
        assert cands == [40, 50, 60]

        # 2-gram fallback match
        tokens_2 = [1, 2, 88, 99, 100, 101, 7, 8, 88, 99]
        cands_2 = find_candidate_tokens(tokens_2, ngram_size=3, max_candidates=2)
        assert cands_2 == [100, 101]

        # No match found
        tokens_none = [1, 2, 3, 4, 5, 6, 7]
        cands_none = find_candidate_tokens(tokens_none, ngram_size=3, max_candidates=3)
        assert cands_none == []

