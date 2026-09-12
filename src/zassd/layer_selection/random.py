"""Random layer selection for baseline comparison."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def select_random_layers(
    num_layers: int,
    skip_ratio: float = 0.25,
    keep_first: bool = True,
    keep_last: bool = True,
    seed: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    """Randomly select layers to skip.

    Args:
        num_layers: Total number of layers.
        skip_ratio: Fraction of layers to skip.
        keep_first: Always keep the first layer.
        keep_last: Always keep the last layer.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (kept_indices, skipped_indices).
    """
    rng = np.random.RandomState(seed)

    all_indices = list(range(num_layers))
    candidates = all_indices.copy()

    if keep_first:
        candidates.remove(0)
    if keep_last:
        candidates.remove(num_layers - 1)

    num_skip = int(len(candidates) * skip_ratio)
    skipped = sorted(rng.choice(candidates, size=num_skip, replace=False).tolist())
    kept = sorted(set(all_indices) - set(skipped))

    logger.info(
        f"Random selection: keeping {len(kept)}/{num_layers} layers, "
        f"skipping {len(skipped)}"
    )

    return kept, skipped
