#!/usr/bin/env python3
"""Paired B1 closed-loop run of tuned M=3 NF4 plus native no-proposal fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import audit_activationvec_no_proposal_fallback as fallback_runner
import audit_bnb_nf4_activationvec_block_integration as integration
import benchmark_bnb_nf4_activationvec_closed_loop as baseline_runner


def select_config(module):
    if int(module.in_features) == 11008 or int(module.out_features) == 11008:
        return 4, 4
    return 8, 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timing-rounds", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    repo = Path(__file__).resolve().parents[1]
    manifest_path = repo / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_nf4_m3_tuned_fallback_manifest_v1.json"
    base_manifest = repo / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_no_proposal_fallback_manifest.json"
    manifest = json.loads(base_manifest.read_text())
    manifest["classification"] = "development-only tuned activationvec M=3 plus native BNB M=1 no-proposal fallback; source-pinned; not product-runtime authorization"
    for source in (
        "scripts/audit_bnb_nf4_activationvec_block_integration.py",
        "scripts/benchmark_activationvec_nf4_m3_tuned_fallback.py",
        "scripts/tune_activationvec_nf4_m3_launch.py",
    ):
        manifest["source_sha256"][source] = hashlib.sha256((repo / source).read_bytes()).hexdigest()
    manifest["candidate_launch"] = {
        "default": {"out_tile": 8, "num_warps": 8},
        "wide_mlp": {"out_tile": 4, "num_warps": 4},
        "selector": "in_features == 11008 or out_features == 11008",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    fallback_runner.MANIFEST = manifest_path
    tuned_override = lambda model: integration.triton_m3_overrides(model, config_selector=select_config)
    fallback_runner.triton_m3_overrides = tuned_override
    baseline_runner.triton_m3_overrides = tuned_override
    sys.argv = [
        str(Path(fallback_runner.__file__).resolve()),
        "--benchmark", args.benchmark,
        "--prompt-id", args.prompt_id,
        "--max-new-tokens", str(args.max_new_tokens),
        "--timing-rounds", str(args.timing_rounds),
        "--output", str(args.output),
    ]
    fallback_runner.main()
    payload = json.loads(args.output.read_text())
    payload["launch_override"] = {
        "default_config": {"out_tile": 8, "num_warps": 8},
        "wide_mlp_config": {"out_tile": 4, "num_warps": 4},
        "selector": "in_features == 11008 or out_features == 11008",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(integration.__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id,
                      "exactness": payload.get("exactness"),
                      "acceptance": payload.get("acceptance"),
                      "ratio_medians": payload.get("ratio_medians"),
                      "launch_override": payload["launch_override"],
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
