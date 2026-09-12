"""Configuration loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {path}")
    return config


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep merge multiple configs (later ones override earlier)."""
    result: dict[str, Any] = {}
    for config in configs:
        _deep_merge(result, config)
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
