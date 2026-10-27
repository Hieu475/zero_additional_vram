#!/usr/bin/env python3
"""Inventory architecture-specific module layouts for the locked model cohort."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.model_capabilities import (
    inspect_architecture_operator_layout,
    inspect_b1_fixed_row_capability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    reports = []

    for model_spec in protocol["models"]:
        row = {
            "model": model_spec["id"],
            "revision": model_spec["revision"],
            "family": model_spec["family"],
        }
        try:
            loaded = load_backend("hf_bnb_nf4", row["model"], row["revision"], args.device, "sdpa")
            model = loaded.model.eval()
            row.update({
                "status": "completed",
                "backend": loaded.metadata,
                "model_class": type(model).__name__,
                "model_type": getattr(model.config, "model_type", None),
                "operator_layout": inspect_architecture_operator_layout(model),
                "fixed_row_capability": inspect_b1_fixed_row_capability(model),
            })
        except Exception as exc:
            row.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
        reports.append(row)
        print(json.dumps({
            "model": row["model"], "status": row["status"],
            "operator_layout_complete": row.get("operator_layout", {}).get("complete"),
            "fixed_row_supported": row.get("fixed_row_capability", {}).get("supported"),
            "rejection_reasons": row.get("fixed_row_capability", {}).get("rejection_reasons"),
        }), flush=True)
        if "model" in locals():
            del model
        if "loaded" in locals():
            del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "schema": "b1-architecture-capability-inventory-v1",
        "scope": "runtime module-layout inventory; not a correctness proof or adapter validation",
        "protocol": str(args.protocol),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "device": args.device,
        "models": reports,
        "completed_models": sum(row["status"] == "completed" for row in reports),
        "failed_models": sum(row["status"] != "completed" for row in reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
