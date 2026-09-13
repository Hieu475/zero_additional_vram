"""Token verification for speculative decoding.

Implements exact greedy verification and stochastic verification.
Greedy verification mathematically guarantees exact output match with
the vanilla autoregressive model (Output_spec == Output_vanilla).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def verify_tokens_greedy(
    target_logits: torch.Tensor,
    draft_tokens: list[int] | torch.Tensor,
) -> tuple[list[int], int, Optional[int]]:
    """Greedy verification of candidate draft tokens.

    Compares draft tokens with target model predictions. At the first mismatch,
    rejects subsequent tokens and emits the target model's correction token.
    If all K draft tokens match, emits the target model's bonus token.

    Args:
        target_logits: Logits from target model forward pass over draft positions.
                       Shape (1, K+1, vocab_size) or (K+1, vocab_size).
        draft_tokens: List or 1D Tensor of K draft token IDs.

    Returns:
        Tuple of:
          - accepted_tokens: list of accepted draft token IDs (length 0..K)
          - next_token: the correction token (if mismatch) or bonus token (if all accepted)
          - rejected_at: index (0..K-1) of first rejected token, or None if all accepted.
    """
    if target_logits.dim() == 3:
        target_logits = target_logits[0]  # Shape: (K+1, vocab_size)

    if isinstance(draft_tokens, torch.Tensor):
        draft_tokens = draft_tokens.tolist()

    k = len(draft_tokens)
    accepted_tokens: list[int] = []

    for i in range(k):
        target_pred = int(target_logits[i].argmax(dim=-1).item())
        draft_tok = draft_tokens[i]

        if target_pred == draft_tok:
            accepted_tokens.append(draft_tok)
        else:
            # Mismatch: reject draft_tok and subsequent tokens
            # Emit target_pred as the correct next token
            return accepted_tokens, target_pred, i

    # All K draft tokens accepted! Emit bonus token from position K
    bonus_token = int(target_logits[k].argmax(dim=-1).item())
    return accepted_tokens, bonus_token, None
