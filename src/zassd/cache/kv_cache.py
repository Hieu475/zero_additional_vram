"""KV cache utilities for speculative decoding.

Implements two-tier cache architectures supporting both dynamic growth and
zero-allocation static pre-allocated memory buffers:
1. TargetKVCache: Canonical ground-truth state maintained by full model.
2. StaticPreallocatedKVCache & StaticPreallocatedLayer: High-performance pre-allocated
   KV buffers with in-place slice copies and zero PyTorch tensor reallocations.
3. EphemeralDraftKV: Ephemeral state for candidate token generation, with zero-copy
   prefix sharing and safe discard/rollback semantics.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicCache, DynamicLayer

logger = logging.getLogger(__name__)


class StaticPreallocatedLayer(CacheLayerMixin):
    """Pre-allocated static KV cache layer for zero-allocation inference.

    Pre-allocates keys and values buffers of shape [bsz, num_heads, max_capacity, head_dim]
    during lazy initialization. Subsequent updates copy incoming key/value states into
    the pre-allocated slice without allocating new GPU memory or executing `torch.cat`.
    """

    is_sliding = False

    def __init__(self, max_capacity: int = 2048) -> None:
        super().__init__()
        self.max_capacity = max_capacity
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.seq_len: int = 0
        self.is_initialized: bool = False
        self.dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Allocate static memory buffers on first encounter."""
        self.dtype = key_states.dtype
        self.device = key_states.device
        bsz, num_heads, _, head_dim = key_states.shape
        self.keys = torch.empty(
            (bsz, num_heads, self.max_capacity, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.empty(
            (bsz, num_heads, self.max_capacity, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.is_initialized = True
        self.seq_len = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update key and value caches in-place into preallocated buffer slice.

        Args:
            key_states: New key states of shape [bsz, num_heads, cur_len, head_dim].
            value_states: New value states of shape [bsz, num_heads, cur_len, head_dim].

        Returns:
            Tuple of active key and value tensor slices up to current sequence length.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        cur_len = key_states.shape[-2]
        start_idx = self.seq_len
        end_idx = start_idx + cur_len
        if end_idx > self.max_capacity:
            raise ValueError(
                f"StaticPreallocatedLayer exceeded max_capacity={self.max_capacity} "
                f"(attempted to store sequence length {end_idx})."
            )

        self.keys[:, :, start_idx:end_idx, :].copy_(key_states)
        self.values[:, :, start_idx:end_idx, :].copy_(value_states)
        self.seq_len = end_idx

        return self.keys[:, :, :self.seq_len, :], self.values[:, :, :self.seq_len, :]

    def get_seq_length(self) -> int:
        """Return the current sequence length stored in the cache."""
        return self.seq_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the length and offset of the cache for causal attention mask creation."""
        return self.get_seq_length() + query_length, 0

    def get_max_length(self) -> int:
        """Return the maximum capacity of the static cache buffer."""
        return self.max_capacity

    def crop(self, max_length: int) -> None:
        """Crop cache back to max_length tokens (O(1) pointer adjustment)."""
        if max_length <= 0:
            max_length = self.get_seq_length() - abs(max_length)
        self.seq_len = min(self.seq_len, max(0, max_length))

    def reset(self) -> None:
        """Reset sequence length without deallocating GPU memory."""
        self.seq_len = 0

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cache across batch dimension if needed."""
        if self.is_initialized and self.keys is not None and self.values is not None:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices."""
        if self.is_initialized and self.keys is not None and self.values is not None:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


class StaticPreallocatedKVCache(Cache):
    """Pre-allocated static KV cache with zero-allocation updates during inference.

    Pre-allocates buffer tensors of size [batch_size, num_heads, max_capacity, head_dim]
    on the first forward pass. Subsequent updates perform in-place `.copy_()` operations
    into the preallocated slice, eliminating PyTorch tensor allocation overheads and
    fragmentation during speculative draft & verify cycles.
    """

    def __init__(self, max_capacity: int = 2048) -> None:
        super().__init__(layer_class_to_replicate=lambda: StaticPreallocatedLayer(max_capacity=max_capacity))
        self.max_capacity = max_capacity

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get sequence length of the specified layer or maximum length across active layers.

        HuggingFace model decoders query `get_seq_length()` without arguments (defaulting to layer 0)
        to construct RoPE position_ids. If layer 0 is skipped during draft speculation,
        layer 0's cache is not updated; returning the maximum length across active layers
        prevents positional regression and maintains strictly monotonic RoPE coordinates.
        """
        if not self.layers:
            return 0
        if layer_idx is None or layer_idx == 0:
            return max((layer.get_seq_length() for layer in self.layers), default=0)
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def crop(self, max_length: int) -> None:
        """Crop all layers back to max_length tokens."""
        for layer in self.layers:
            layer.crop(max_length)

    def reset(self) -> None:
        """Reset sequence lengths across all layers."""
        for layer in self.layers:
            layer.reset()

    def fork_ephemeral_draft_kv(self) -> StaticPreallocatedKVCache:
        """Fork ephemeral draft cache sharing preallocated memory buffers.

        The draft cache shares the exact same underlying keys and values buffers
        as the target cache, but maintains its own independent `seq_len`.
        Draft writes candidate tokens into [target_seq_len : target_seq_len + K].
        Target verification then evaluates and overwrites this segment with canonical
        target KV activations.
        """
        draft_cache = StaticPreallocatedKVCache(max_capacity=self.max_capacity)
        draft_cache.layers = []
        for tl in self.layers:
            dl = StaticPreallocatedLayer(max_capacity=self.max_capacity)
            dl.keys = tl.keys
            dl.values = tl.values
            dl.seq_len = tl.seq_len
            dl.is_initialized = tl.is_initialized
            dl.dtype = getattr(tl, "dtype", None)
            dl.device = getattr(tl, "device", None)
            draft_cache.layers.append(dl)
        return draft_cache


class TargetKVCache:
    """Canonical target KV cache.

    Represents the ground-truth KV state computed strictly by the full target
    model over accepted tokens. Never polluted with unverified draft tokens.

    Supports both 'static' (pre-allocated static buffer for zero-allocation updates)
    and 'dynamic' (HuggingFace DynamicCache) backends.

    Attributes:
        cache: Underlying HuggingFace Cache instance (StaticPreallocatedKVCache or DynamicCache).
        backend: The active backend identifier ('static', 'dynamic', or 'custom').
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        backend: str = "static",
        max_capacity: int = 2048,
    ) -> None:
        if cache is not None:
            self.cache = cache
            self.backend = "custom"
        elif backend == "static":
            self.cache = StaticPreallocatedKVCache(max_capacity=max_capacity)
            self.backend = "static"
        elif backend == "dynamic":
            self.cache = DynamicCache()
            self.backend = "dynamic"
        else:
            raise ValueError(f"Unknown KV cache backend: {backend}. Expected 'static' or 'dynamic'.")
        self.max_capacity = max_capacity

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get sequence length stored in the cache."""
        if not hasattr(self.cache, "layers") or not self.cache.layers:
            return 0
        if layer_idx is None or layer_idx == 0:
            return max((layer.get_seq_length() for layer in self.cache.layers), default=0)
        if layer_idx >= len(self.cache.layers):
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

    def fork_ephemeral_draft_kv(self) -> Cache:
        """Create an ephemeral draft KV cache sharing canonical prefix references.

        Returns:
            A new Cache configured for ephemeral draft generation.
        """
        if hasattr(self.cache, "fork_ephemeral_draft_kv"):
            return self.cache.fork_ephemeral_draft_kv()

        # Fallback for DynamicCache
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
            if getattr(layer, "is_initialized", False):
                if getattr(layer, "keys", None) is not None:
                    total_bytes += layer.keys.element_size() * layer.keys.nelement()
                if getattr(layer, "values", None) is not None:
                    total_bytes += layer.values.element_size() * layer.values.nelement()
        return total_bytes / (1024 * 1024)

    def reset(self) -> None:
        """Reset the cache completely."""
        if hasattr(self.cache, "reset"):
            self.cache.reset()
        elif isinstance(self.cache, DynamicCache):
            self.cache = DynamicCache()
        else:
            self.cache = type(self.cache)()


class EphemeralDraftKV:
    """Manager for ephemeral draft KV states."""

    @staticmethod
    def create_from_target(target_kv: TargetKVCache) -> Cache:
        """Fork an ephemeral draft cache from canonical target KV."""
        return target_kv.fork_ephemeral_draft_kv()
