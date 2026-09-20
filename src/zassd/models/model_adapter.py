"""Model adapter for unified layer access across architectures."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class ModelAdapter:
    """Provides a unified interface for accessing model layers.

    Abstracts away architecture-specific layer naming to support
    Qwen, LLaMA, and other transformer architectures.
    """

    # Known architecture -> layer accessor mappings
    LAYER_PATHS = {
        "Qwen2ForCausalLM": "model.layers",
        "LlamaForCausalLM": "model.layers",
        "MistralForCausalLM": "model.layers",
        "Phi3ForCausalLM": "model.layers",
    }

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
        self.architecture = type(model).__name__
        self._layers = self._get_layers()
        logger.info(
            f"ModelAdapter initialized for {self.architecture} "
            f"with {self.num_layers} layers"
        )

    def _get_layers(self) -> nn.ModuleList:
        """Get the transformer layers from the model."""
        layer_path = self.LAYER_PATHS.get(self.architecture)
        if layer_path is None:
            # Try common paths
            for path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
                try:
                    obj = self.model
                    for attr in path.split("."):
                        obj = getattr(obj, attr)
                    return obj
                except AttributeError:
                    continue
            raise ValueError(
                f"Unknown architecture: {self.architecture}. "
                f"Cannot find transformer layers."
            )
        obj = self.model
        for attr in layer_path.split("."):
            obj = getattr(obj, attr)
        return obj

    @property
    def num_layers(self) -> int:
        """Total number of transformer layers."""
        return len(self._layers)

    def get_layer(self, idx: int) -> nn.Module:
        """Get a specific layer by index."""
        return self._layers[idx]

    def get_layers(self) -> nn.ModuleList:
        """Get all transformer layers."""
        return self._layers

    def get_layer_indices(self) -> list[int]:
        """Get list of all layer indices."""
        return list(range(self.num_layers))
