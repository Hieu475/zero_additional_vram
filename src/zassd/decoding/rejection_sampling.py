"""Rejection sampling for speculative decoding.

Implements the modified rejection sampling scheme from Leviathan et al. (2023)
to guarantee exact distribution preservation when sampling at temperature > 0.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def verify_tokens_rejection_sampling(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_tokens: list[int] | torch.Tensor,
) -> tuple[list[int], int, Optional[int]]:
    """Verify draft tokens using speculative rejection sampling.

    Accepts token with probability min(1, p(x) / q(x)).
    If rejected, samples correction token from normalized max(0, p(x) - q(x)).

    Args:
        target_probs: Target probabilities over vocab. Shape (K+1, vocab_size).
        draft_probs: Draft probabilities over vocab. Shape (K, vocab_size).
        draft_tokens: List of K draft token IDs.

    Returns:
        Tuple of:
          - accepted_tokens: list of accepted draft tokens
          - next_token: correction token or bonus token
          - rejected_at: index of first rejected token, or None if all accepted
    """
    if target_probs.dim() == 3:
        target_probs = target_probs[0]
    if draft_probs.dim() == 3:
        draft_probs = draft_probs[0]

    if isinstance(draft_tokens, torch.Tensor):
        draft_tokens = draft_tokens.tolist()

    k = len(draft_tokens)
    accepted_tokens: list[int] = []

    for i in range(k):
        tok = draft_tokens[i]
        p = target_probs[i, tok].item()
        q = draft_probs[i, tok].item()

        # Acceptance ratio
        accept_prob = min(1.0, p / max(q, 1e-10))
        r = torch.rand(1).item()

        if r < accept_prob:
            accepted_tokens.append(tok)
        else:
            # Rejection: sample from adjusted distribution max(0, p - q)
            diff = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
            diff_sum = diff.sum()
            if diff_sum > 1e-8:
                adjusted_dist = diff / diff_sum
                correction_token = int(torch.multinomial(adjusted_dist, num_samples=1).item())
            else:
                correction_token = int(target_probs[i].argmax(dim=-1).item())
            return accepted_tokens, correction_token, i

    # All accepted: sample bonus token from target position K
    bonus_token = int(torch.multinomial(target_probs[k], num_samples=1).item())
    return accepted_tokens, bonus_token, None
