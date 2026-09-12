"""Download and cache model for reproducibility.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import logging

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "main",
    },
]


def download_model(model_name: str, revision: str = "main") -> str:
    """Download model snapshot.

    Args:
        model_name: HuggingFace model name.
        revision: Git revision to pin.

    Returns:
        Local path to downloaded model.
    """
    logger.info(f"Downloading {model_name} (revision: {revision})")

    local_path = snapshot_download(
        repo_id=model_name,
        revision=revision,
    )

    logger.info(f"Model downloaded to: {local_path}")
    return local_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to download",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision",
    )
    args = parser.parse_args()

    if args.model:
        download_model(args.model, args.revision)
    else:
        for model_info in DEFAULT_MODELS:
            download_model(model_info["name"], model_info["revision"])


if __name__ == "__main__":
    main()
