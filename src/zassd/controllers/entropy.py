"""Entropy-based metrics for adaptive control."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute token-level entropy from logits.

    Args:
        logits: Model output logits, shape (..., vocab_size).

    Returns:
        Entropy values.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def entropy_category(
    entropy: float,
    low_threshold: float = 0.5,
    high_threshold: float = 2.0,
) -> str:
    """Categorize entropy level.

    Args:
        entropy: Entropy value.
        low_threshold: Below this = low entropy (confident).
        high_threshold: Above this = high entropy (uncertain).

    Returns:
        Category string: 'low', 'medium', or 'high'.
    """
    if entropy < low_threshold:
        return "low"
    elif entropy > high_threshold:
        return "high"
    else:
        return "medium"
