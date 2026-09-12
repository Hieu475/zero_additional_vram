"""Benchmark layer skipping strategies.

Compares random and fixed layer skipping against vanilla baseline.

Usage:
    python scripts/benchmark_layer_skip.py
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    # TODO: Implement after vanilla baseline (Experiment 02)
    logger.info("Layer skip benchmark - to be implemented after vanilla baseline")
    raise NotImplementedError(
        "Implement after vanilla baseline is established. "
        "See experiments/02_layer_skip/"
    )


if __name__ == "__main__":
    main()
