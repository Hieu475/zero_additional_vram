#!/usr/bin/env python3
"""Correct inherited kernel labels in experimental three-row result JSONs.

The shared K2/R3 runner writes a generic pairwise-kernel label. The outer
experimental wrapper intercepts M=3 calls and dispatches the three-row kernel.
This script preserves the old label and records the correction explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel")
    corrected = "experimental fixed-three-row packed NF4 N=16/K=256/1-warp kernel for M=3 block calls; singleton calls retain the existing one-row prototype"
    changed = []
    paths = set(root.glob("closed_loop*.json")) | set(root.glob("state_machine*.json"))
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            continue
        previous = parameters.get("block_nf4_kernel")
        if previous == corrected:
            continue
        payload["metadata_correction"] = {
            "field": "parameters.block_nf4_kernel",
            "previous_value": previous,
            "corrected_value": corrected,
            "reason": "The inherited K2/R3 runner label described its base pairwise dispatch; the opt-in wrapper intercepted each M=3 block call and dispatched the fixed-three-row kernel. Numerical outputs and timings were not changed by this annotation.",
        }
        parameters["block_nf4_kernel"] = corrected
        limitations = payload.get("limitations", [])
        if limitations:
            limitations[0] = "Finite development-only closed loop; block M=3 uses the fixed-three-row prototype and singleton calls use the one-row prototype."
        path.write_text(json.dumps(payload, indent=2) + "\n")
        changed.append(str(path))
    print(json.dumps({"corrected_artifacts": changed, "count": len(changed)}, indent=2))


if __name__ == "__main__":
    main()
