"""KV cache utilities for speculative decoding."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# TODO: Implement KV cache management
# Key considerations:
# - Sharing KV cache between draft and verify passes
# - Rolling back cache on rejected tokens
# - Memory-efficient cache for 6GB VRAM constraint
