#!/usr/bin/env python3
"""Same-process B1 A/B for a lower-warp per-shape M=3 launch map."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import benchmark_activationvec_nf4_m3_launch_ab as ab


def lower_warp_config(module):
    k, n = int(module.in_features), int(module.out_features)
    if k == 2048 and n == 2048:
        return 4, 1
    if k == 11008 or n == 11008:
        return 4, 4
    return 8, 8


def pin_lower_warp_manifest():
    data = json.loads(ab.BASE_MANIFEST.read_text())
    data["classification"] = "same-process paired baseline-vs-lower-warp NF4 M=3 launch A/B with B1 fallback; source-pinned; development only"
    for name in ("scripts/audit_bnb_nf4_activationvec_block_integration.py",
                 "scripts/benchmark_activationvec_nf4_m3_launch_ab.py",
                 "scripts/benchmark_activationvec_nf4_m3_lower_warp_ab.py",
                 "scripts/tune_activationvec_nf4_m3_lower_warps.py",
                 "scripts/tune_activationvec_nf4_m3_launch.py"):
        data["source_sha256"][name] = hashlib.sha256((ab.ROOT / name).read_bytes()).hexdigest()
    data["launch_ab"] = {
        "default": {"out_tile": 8, "num_warps": 8},
        "q_o_2048x2048": {"out_tile": 4, "num_warps": 1},
        "wide_mlp": {"out_tile": 4, "num_warps": 4},
        "selector": "K=2048,N=2048 -> 4x1; K=11008 or N=11008 -> 4x4; else 8x8",
    }
    ab.MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
    ab.fallback.MANIFEST = ab.MANIFEST
    ab.fallback.validate_manifest()


def main():
    ab.tuned_config = lower_warp_config
    ab.pin_manifest = pin_lower_warp_manifest
    ab.MANIFEST = ab.ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_nf4_m3_lower_warp_ab_manifest_v1.json"
    original_main = ab.main
    args = sys.argv[1:]
    original_argv = sys.argv
    sys.argv = [str(Path(ab.__file__).resolve()), *args]
    original_main()
    sys.argv = original_argv
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timing-rounds", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(args)
    payload = json.loads(parsed.output.read_text())
    payload["launch_configs"] = {
        "default": "8x8 for all M=3 quantized linears",
        "candidate": "4x1 for K=N=2048; 4x4 for K or N=11008; 8x8 otherwise",
    }
    payload["launch_driver"] = {
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    parsed.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
