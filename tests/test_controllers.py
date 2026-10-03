"""Tests for controllers."""

from __future__ import annotations

import pytest
import torch

from zassd.controllers.entropy import compute_entropy, entropy_category
from zassd.controllers.adaptive_k import AdaptiveKController


class TestEntropy:
    """Test entropy computation."""

    def test_uniform_distribution(self):
        """Uniform distribution should have maximum entropy."""
        logits = torch.zeros(1, 100)  # Uniform
        entropy = compute_entropy(logits)
        assert entropy.item() > 0

    def test_peaked_distribution(self):
        """Very peaked distribution should have low entropy."""
        logits = torch.zeros(1, 100)
        logits[0, 0] = 100.0  # Very peaked
        entropy = compute_entropy(logits)
        assert entropy.item() < 0.1

    def test_entropy_category(self):
        assert entropy_category(0.3) == "low"
        assert entropy_category(1.0) == "medium"
        assert entropy_category(3.0) == "high"


class TestAdaptiveK:
    """Test adaptive K controller."""

    def test_initial_k(self):
        controller = AdaptiveKController(initial_k=4)
        assert controller.current_k == 4

    def test_increase_k(self):
        controller = AdaptiveKController(
            initial_k=4, k_max=8,
            entropy_low=0.5, acceptance_target=0.7,
        )
        # Simulate low entropy, high acceptance
        for _ in range(25):
            controller.update(entropy=0.2, accepted=4, proposed=4)
        assert controller.current_k > 4

    def test_decrease_k(self):
        controller = AdaptiveKController(
            initial_k=4, k_min=1,
            entropy_high=2.0, acceptance_target=0.7,
        )
        # Simulate high entropy, low acceptance
        for _ in range(25):
            controller.update(entropy=3.0, accepted=1, proposed=4)
        assert controller.current_k < 4

    def test_k_bounds(self):
        controller = AdaptiveKController(k_min=2, k_max=6, initial_k=4)
        # Try to push below min
        for _ in range(50):
            controller.update(entropy=5.0, accepted=0, proposed=4)
        assert controller.current_k >= 2

    def test_reset(self):
        controller = AdaptiveKController(initial_k=4)
        controller.update(entropy=1.0, accepted=2, proposed=4)
        controller.reset()
        assert controller.avg_acceptance_rate == 0.0


class TestHardwareAwareJointController:
    """Test hardware-aware joint speculation controller."""

    def test_controller_action_selection(self):
        from zassd.controllers.hardware_controller import HardwareAwareJointController

        configs = {
            "cka_75": [3, 5, 7, 9, 11, 13, 16, 18, 21],
            "cka_50": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21],
        }
        controller = HardwareAwareJointController(candidate_layer_configs=configs)

        action = controller.select_action(
            entropy=1.0,
            last_accepted=3,
            last_proposed=4,
            draft_ms=20.0,
            verify_ms=25.0,
        )

        # Feasibility-gate contract: the action is either a speculative (S,K)
        # or the VANILLA skip (K=0) when speculation is predicted to lose.
        if action.vanilla_skip:
            assert action.draft_length == 0
            assert action.config_name == "vanilla"
        else:
            assert action.config_name in configs
            assert 1 <= action.draft_length <= 8
            assert action.skip_indices == configs[action.config_name]
        assert len(controller.action_history) == 1

    def test_controller_with_action_cost_db(self):
        from zassd.controllers.hardware_controller import HardwareAwareJointController
        from zassd.profiling.action_profiler import ActionCostDatabase

        mock_db_data = {
            "cka_75": {
                "K4": {
                    "tokens_per_second": 32.0,
                    "total_cycle_ms": 75.0,
                    "tokens_per_step": 2.4,
                }
            },
            "cka_50": {
                "K4": {
                    "tokens_per_second": 45.0,
                    "total_cycle_ms": 55.0,
                    "tokens_per_step": 2.5,
                }
            },
        }
        db = ActionCostDatabase(mock_db_data)

        configs = {
            "cka_75": [3, 5, 7, 9],
            "cka_50": [3, 5, 7, 9, 11, 13],
        }
        controller = HardwareAwareJointController(
            candidate_layer_configs=configs,
            action_cost_db=db,
        )

        action = controller.select_action(
            entropy=0.5,
            last_accepted=3,
            last_proposed=3,
            draft_ms=15.0,
            verify_ms=25.0,
        )

        assert action.config_name in ("cka_75", "cka_50")
        assert 1 <= action.draft_length <= 4

