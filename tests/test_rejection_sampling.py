"""Tests for rejection sampling."""

from __future__ import annotations

import pytest
import torch
from zassd.decoding.rejection_sampling import verify_tokens_rejection_sampling


class TestRejectionSampling:
    """Test speculative rejection sampling."""

    def test_identical_distributions_all_accepted(self):
        # When target_probs == draft_probs, acceptance probability is min(1, p/q) = 1.0
        torch.manual_seed(42)
        vocab_size = 50
        k = 3

        probs = torch.softmax(torch.randn(k + 1, vocab_size), dim=-1)
        draft_probs = probs[:k].clone()
        draft_tokens = [int(p.argmax().item()) for p in draft_probs]

        accepted, bonus, rejected_at = verify_tokens_rejection_sampling(
            target_probs=probs,
            draft_probs=draft_probs,
            draft_tokens=draft_tokens,
        )

        assert accepted == draft_tokens
        assert rejected_at is None
        assert 0 <= bonus < vocab_size
