"""Generic experiment runner.

Usage:
    python scripts/run_experiment.py --config configs/experiments/layer_skip.yaml
    python scripts/run_experiment.py --experiment 01_vanilla
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--config", type=str, help="Experiment config YAML")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    args = parser.parse_args()

    # TODO: Implement generic experiment runner
    logger.info("Experiment runner - to be implemented")
    raise NotImplementedError("Generic experiment runner not yet implemented")


if __name__ == "__main__":
    main()
