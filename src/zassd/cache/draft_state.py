"""Draft state management for speculative decoding."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class DraftState:
    """State of a draft generation pass."""
    draft_tokens: list[int] = field(default_factory=list)
    draft_logits: list[torch.Tensor] = field(default_factory=list)
    draft_probs: list[torch.Tensor] = field(default_factory=list)
    skipped_layers: list[int] = field(default_factory=list)
    draft_time_ms: float = 0.0

    def clear(self) -> None:
        """Clear draft state."""
        self.draft_tokens.clear()
        self.draft_logits.clear()
        self.draft_probs.clear()
        self.skipped_layers.clear()
        self.draft_time_ms = 0.0
