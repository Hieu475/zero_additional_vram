"""Fail-closed source/runtime identity check for the NF4 singleton K=2/R=3 experiment manifest.

This validates recorded experiment inputs only. It does not certify numerical equivalence
or enable a decoding runtime path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/experiment_manifest.json"
EXPECTED_SCHEMA = "zassd-nf4-singleton-k2-r3-experiment-manifest-v1"
EXPECTED_MODEL = {"id": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1"}
EXPECTED_SCHEDULE = {"k": 2, "ngram_size": 3, "physical_rows": 3, "attention": "triton_batched", "attention_block_k": 32}
EXPECTED_RUNTIME = {"python": "3.12.13", "torch": "2.11.0+cu128", "cuda": "12.8", "triton": "3.6.0", "transformers": "5.14.1", "bitsandbytes": "0.50.0", "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    mismatches = []
    if manifest.get("schema") != EXPECTED_SCHEMA:
        mismatches.append("manifest:schema")
    if manifest.get("model") != EXPECTED_MODEL:
        mismatches.append("manifest:model")
    schedule = manifest.get("schedule", {})
    if any(schedule.get(key) != value for key, value in EXPECTED_SCHEDULE.items()):
        mismatches.append("manifest:schedule")
    if not manifest.get("source_sha256"):
        mismatches.append("manifest:source_sha256_empty")
    if manifest.get("runtime") != EXPECTED_RUNTIME:
        mismatches.append("manifest:runtime_pin")
    for relpath, expected in manifest["source_sha256"].items():
        path = ROOT / relpath
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(f"source:{relpath}")

    live = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "transformers": importlib.metadata.version("transformers"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for key, expected in manifest["runtime"].items():
        if live.get(key) != expected:
            mismatches.append(f"runtime:{key} expected={expected!r} actual={live.get(key)!r}")

    if mismatches:
        print(json.dumps({"valid": False, "mismatches": mismatches}, indent=2))
        return 1
    print(json.dumps({
        "valid": True,
        "manifest": str(manifest_path),
        "source_count": len(manifest["source_sha256"]),
        "runtime": live,
        "scope": manifest["classification"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
