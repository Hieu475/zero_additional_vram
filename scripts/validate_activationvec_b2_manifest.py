#!/usr/bin/env python3
"""Fail-closed hash/runtime check for the B2 activationvec experiment manifest.

Verifies recorded inputs only; it does not certify numerical equivalence.
Evidence files listed under evidence_stale_source are reported but do not fail the check.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments/exact_quantized_feasibility/activationvec_b2_v1/experiment_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT)
    ap.add_argument("--skip-runtime", action="store_true", help="skip torch/triton/bitsandbytes version pins")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    problems = []
    for key in ("source_sha256", "evidence_sha256"):
        if not manifest.get(key):
            problems.append(f"manifest:{key}_empty")
    stale = set(manifest.get("evidence_stale_source", {}).get("files", []))
    for group in ("source_sha256", "evidence_sha256"):
        for rel, expected in manifest.get(group, {}).items():
            path = ROOT / rel
            if not path.is_file() or sha256(path) != expected:
                problems.append(f"{group}:{rel}")
    if not args.skip_runtime:
        live = {"torch": importlib.metadata.version("torch"), "triton": importlib.metadata.version("triton"),
                "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                "transformers": importlib.metadata.version("transformers"), "python": platform.python_version()}
        for name, value in live.items():
            if manifest["runtime"].get(name) != value:
                problems.append(f"runtime:{name}:{manifest['runtime'].get(name)}!={value}")
    print(json.dumps({"manifest": str(args.manifest), "problems": problems, "stale_evidence_listed": sorted(stale)}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
