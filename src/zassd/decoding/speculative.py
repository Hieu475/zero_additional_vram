"""Self-speculative decoding implementation.

This module implements the core self-speculative decoding loop:
1. Draft phase: Generate K candidate tokens using a subnetwork (skipped layers)
2. Verify phase: Verify all K tokens in parallel using the full model
3. Accept/reject based on verification results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeMetrics:
    """Metrics for speculative decoding."""
    total_tokens: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0
    draft_time_s: float = 0.0
    verify_time_s: float = 0.0
    peak_vram_mb: float = 0.0
    speedup_vs_vanilla: float = 0.0
    per_iteration_stats: list[dict] = field(default_factory=list)


# TODO: Implement after vanilla baseline is established
# See experiments/04_self_speculative/
