"""Layer manager for enabling/disabling layers via logical layer skipping."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator

import torch.nn as nn

from .model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class LayerManager:
    """Manages layer activation/deactivation via logical layer skipping.

    Instead of mutating the model's ModuleList (which breaks layer_idx indexing
    and invalidates KV cache alignments), logical layer skipping patches the
    forward method of skipped layers to act as identity functions.

    Advantages:
    - All layer_idx attributes remain identical to the original model.
    - HuggingFace DynamicCache and attention caches remain properly aligned.
    - Zero module mutation or state destruction.
    - Zero additional VRAM overhead.
    """

    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter
        self._original_forwards: dict[int, Callable[..., Any]] = {}
        self._active_skips: set[int] = set()

    @property
    def active_skip_indices(self) -> list[int]:
        """Currently skipped layer indices."""
        return sorted(self._active_skips)

    def is_skipped(self, layer_idx: int) -> bool:
        """Check if a specific layer is currently skipped."""
        return layer_idx in self._active_skips

    @contextmanager
    def skip_layers(
        self, skip_indices: list[int]
    ) -> Generator[None, None, None]:
        """Context manager to logically skip specified layers.

        Patches the forward method of skipped layers to identity.
        Restores original forward methods upon exit.

        Args:
            skip_indices: Layer indices to skip (0-based).
        """
        self.set_skipped_layers(skip_indices)
        try:
            yield
        finally:
            self.restore_layers()

    def set_skipped_layers(self, skip_indices: list[int]) -> None:
        """Logically skip specified layers by replacing forward with identity."""
        layers = self.adapter.get_layers()
        valid_indices = set(skip_indices) & set(range(len(layers)))

        for idx in valid_indices:
            if idx not in self._original_forwards:
                layer = layers[idx]
                self._original_forwards[idx] = layer.forward

                # Identity forward: returns hidden_states unchanged, preserving tuple/tensor signature
                def make_identity_forward(orig_fn: Callable[..., Any], layer_index: int) -> Callable[..., Any]:
                    def identity_forward(hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
                        if isinstance(hidden_states, tuple):
                            # Transformer layers returning (hidden_states, attention_weights, ...)
                            return (hidden_states[0],) + (None,) * (len(hidden_states) - 1)
                        return hidden_states
                    return identity_forward

                layer.forward = make_identity_forward(layer.forward, idx)
                self._active_skips.add(idx)

        logger.debug(
            f"Logically skipping {len(self._active_skips)}/{self.adapter.num_layers} layers: "
            f"{sorted(self._active_skips)}"
        )

    def restore_layers(self) -> None:
        """Restore all original layer forward methods."""
        layers = self.adapter.get_layers()
        for idx, orig_forward in self._original_forwards.items():
            layers[idx].forward = orig_forward

        self._original_forwards.clear()
        self._active_skips.clear()
