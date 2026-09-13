"""Layer manager for enabling/disabling layers during draft generation."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import torch.nn as nn

from .model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class LayerManager:
    """Manages layer activation/deactivation for self-speculative decoding.

    Uses layer-list replacement instead of identity-forward patching.
    This is more robust across architectures because the model simply
    iterates over fewer layers — KV cache, attention, and RoPE all
    work naturally.
    """

    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter
        self._original_layers: nn.ModuleList | None = None

    @contextmanager
    def skip_layers(
        self, skip_indices: list[int]
    ) -> Generator[None, None, None]:
        """Context manager to skip specified layers.

        Temporarily replaces the model's layer list with only the kept
        layers.  On exit the original list is restored.

        Args:
            skip_indices: Layer indices to skip (0-based).
        """
        layers = self.adapter.get_layers()
        self._original_layers = layers

        keep_indices = sorted(
            set(range(self.adapter.num_layers)) - set(skip_indices)
        )
        kept_layers = nn.ModuleList([layers[i] for i in keep_indices])

        # Replace layers in the model
        self._set_layers(kept_layers)

        logger.debug(
            f"Skipping {len(skip_indices)}/{self.adapter.num_layers} layers "
            f"({len(keep_indices)} kept): skip={skip_indices}"
        )

        try:
            yield
        finally:
            # Restore original layers
            if self._original_layers is not None:
                self._set_layers(self._original_layers)
                self._original_layers = None

    def _set_layers(self, layers: nn.ModuleList) -> None:
        """Set the transformer layers in the model."""
        architecture = self.adapter.architecture
        model = self.adapter.model

        # Walk the known path and replace the final attribute
        path = self.adapter.LAYER_PATHS.get(architecture)
        if path is None:
            for p in ["model.layers", "transformer.h", "gpt_neox.layers"]:
                try:
                    parts = p.split(".")
                    obj = model
                    for attr in parts[:-1]:
                        obj = getattr(obj, attr)
                    setattr(obj, parts[-1], layers)
                    return
                except AttributeError:
                    continue
            raise ValueError(f"Cannot set layers for architecture: {architecture}")

        parts = path.split(".")
        obj = model
        for attr in parts[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, parts[-1], layers)
