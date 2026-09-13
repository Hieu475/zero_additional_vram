"""Tests for token verification."""

from __future__ import annotations

import pytest
import torch
from zassd.decoding.verification import verify_tokens_greedy


class TestGreedyVerification:
    """Test greedy token verification logic."""

    def test_all_tokens_accepted(self):
        # K = 3 draft tokens: [10, 20, 30]
        # Target logits predict: [10, 20, 30] and bonus token [40]
        target_logits = torch.zeros(4, 100)
        target_logits[0, 10] = 10.0
        target_logits[1, 20] = 10.0
        target_logits[2, 30] = 10.0
        target_logits[3, 40] = 10.0

        accepted, next_tok, rejected_at = verify_tokens_greedy(
            target_logits=target_logits,
            draft_tokens=[10, 20, 30],
        )

        assert accepted == [10, 20, 30]
        assert next_tok == 40
        assert rejected_at is None

    def test_partial_acceptance_first_mismatch(self):
        # Draft: [10, 20, 30]
        # Target predicts [10, 99, 30]
        # Mismatch at index 1: should accept [10], emit correction token 99, rejected_at = 1
        target_logits = torch.zeros(4, 100)
        target_logits[0, 10] = 10.0
        target_logits[1, 99] = 10.0
        target_logits[2, 30] = 10.0
        target_logits[3, 40] = 10.0

        accepted, next_tok, rejected_at = verify_tokens_greedy(
            target_logits=target_logits,
            draft_tokens=[10, 20, 30],
        )

        assert accepted == [10]
        assert next_tok == 99
        assert rejected_at == 1

    def test_immediate_rejection(self):
        # Draft: [10, 20, 30]
        # Target predicts [88, ...]
        target_logits = torch.zeros(4, 100)
        target_logits[0, 88] = 10.0

        accepted, next_tok, rejected_at = verify_tokens_greedy(
            target_logits=target_logits,
            draft_tokens=[10, 20, 30],
        )

        assert accepted == []
        assert next_tok == 88
        assert rejected_at == 0
