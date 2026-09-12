"""Static (fixed) layer selection strategies."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def select_even_layers(num_layers: int) -> tuple[list[int], list[int]]:
    """Keep even-indexed layers, skip odd-indexed."""
    kept = [i for i in range(num_layers) if i % 2 == 0]
    skipped = [i for i in range(num_layers) if i % 2 != 0]
    return kept, skipped


def select_odd_layers(num_layers: int) -> tuple[list[int], list[int]]:
    """Keep odd-indexed layers, skip even-indexed."""
    kept = [i for i in range(num_layers) if i % 2 != 0]
    # Always keep first layer
    if 0 not in kept:
        kept = [0] + kept
        skipped = [i for i in range(num_layers) if i not in kept]
    else:
        skipped = [i for i in range(num_layers) if i % 2 == 0]
    return sorted(kept), sorted(skipped)


def select_middle_skip(
    num_layers: int,
    skip_start: int,
    skip_end: int,
) -> tuple[list[int], list[int]]:
    """Skip a contiguous block of middle layers."""
    skipped = list(range(skip_start, min(skip_end, num_layers)))
    kept = [i for i in range(num_layers) if i not in skipped]
    return kept, skipped
