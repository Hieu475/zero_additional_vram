"""Benchmark self-speculative decoding.

Usage:
    python scripts/benchmark_speculative.py
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    # TODO: Implement after CKA layer selection (Experiment 04)
    logger.info("Speculative benchmark - to be implemented")
    raise NotImplementedError(
        "Implement after CKA analysis. "
        "See experiments/04_self_speculative/"
    )


if __name__ == "__main__":
    main()
