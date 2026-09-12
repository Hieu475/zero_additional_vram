"""Rejection sampling for speculative decoding.

Implements the modified rejection sampling scheme that ensures
the output distribution matches the target model exactly,
regardless of draft model quality.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# TODO: Implement rejection sampling
# Reference: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"
