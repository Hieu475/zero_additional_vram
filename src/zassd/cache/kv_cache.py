"""KV cache utilities for speculative decoding.

Implements a two-tier cache architecture:
1. TargetKVCache: Canonical ground-truth state maintained by full model.
2. EphemeralDraftKV: Ephemeral state for candidate token generation, with zero-copy
   prefix sharing and safe discard/rollback semantics.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

logger = logging.getLogger(__name__)


class TargetKVCache:
    """Canonical target KV cache.

    Represents the ground-truth KV state computed strictly by the full target
    model over accepted tokens. Never polluted with unverified draft tokens.

    Attributes:
        cache: Underlying HuggingFace DynamicCache instance.
    """

    def __init__(self, cache: Optional[DynamicCache] = None) -> None:
        self.cache: DynamicCache = cache if cache is not None else DynamicCache()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length stored in the cache."""
        if not self.cache.layers:
            return 0
        return self.cache.get_seq_length(layer_idx)

    def crop(self, max_length: int) -> None:
        """Crop cache back to max_length tokens.

        Used to roll back unaccepted candidate tokens after verification.
        """
        self.cache.crop(max_length)

    def rollback(self, prefix_len: int, accepted_count: int) -> None:
        """Rollback target KV cache to prefix_len + accepted_count tokens."""
        target_len = prefix_len + accepted_count
        self.crop(target_len)

    def fork_ephemeral_draft_kv(self) -> DynamicCache:
        """Create an ephemeral draft KV cache sharing canonical prefix references.

        Uses zero-copy reference sharing for existing key/value tensors.
        When draft generation calls DynamicLayer.update(), torch.cat creates new
        tensors and rebinds dl.keys, leaving target KV canonical tensors completely
        intact.

        Returns:
            A new DynamicCache configured for ephemeral draft generation.
        """
        draft_cache = DynamicCache()
        draft_cache.layers = [DynamicLayer() for _ in self.cache.layers]
        for dl, tl in zip(draft_cache.layers, self.cache.layers):
            dl.keys = tl.keys
            dl.values = tl.values
            dl.is_initialized = tl.is_initialized
        return draft_cache

    def get_memory_mb(self) -> float:
        """Compute memory footprint of cached key/value tensors in MB."""
        total_bytes = 0
        for layer in self.cache.layers:
            if layer.is_initialized and layer.keys is not None:
                total_bytes += layer.keys.element_size() * layer.keys.nelement()
            if layer.is_initialized and layer.values is not None:
                total_bytes += layer.values.element_size() * layer.values.nelement()
        return total_bytes / (1024 * 1024)

    def reset(self) -> None:
        """Reset the cache completely."""
        self.cache = DynamicCache()


class EphemeralDraftKV:
    """Manager for ephemeral draft KV states."""

    @staticmethod
    def create_from_target(target_kv: TargetKVCache) -> DynamicCache:
        """Fork an ephemeral draft cache from canonical target KV."""
        return target_kv.fork_ephemeral_draft_kv()
