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
