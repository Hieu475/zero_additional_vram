#!/usr/bin/env python3
"""Same-process B1 A/B for streaming (.cs) vs cached (.ca) activation loads."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import audit_bnb_nf4_activationvec_block_integration as integration
import benchmark_activationvec_nf4_m3_launch_ab as ab
import benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 as base_kernel
import tune_activationvec_nf4_cache_policy as cache_sweep


CA_KERNEL, CA_SOURCE = cache_sweep.build_ca_kernel()


def cached_candidate_config(_module):
    return 8, 8


def pin_cache_manifest():
    data = json.loads(ab.BASE_MANIFEST.read_text())
    data["classification"] = "same-process paired streaming-vs-cached activation-load M=3 experiment with B1 fallback; source-pinned; development only"
    for name in (
        "scripts/audit_bnb_nf4_activationvec_block_integration.py",
        "scripts/benchmark_activationvec_nf4_m3_launch_ab.py",
        "scripts/benchmark_activationvec_nf4_cache_policy_ab.py",
        "scripts/tune_activationvec_nf4_cache_policy.py",
        "scripts/benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2.py",
        "scripts/_nf4_activationvec_ca_kernel.py",
    ):
        data["source_sha256"][name] = hashlib.sha256((ab.ROOT / name).read_bytes()).hexdigest()
    data["cache_policy_ab"] = {
        "default": "activation vector loads use ld.global.cs.v4.u32",
        "candidate": "activation vector loads use ld.global.ca.v4.u32",
        "weight_loads": "ld.global.cs.v4.u32 in both variants",
        "launch": {"out_tile": 8, "num_warps": 8},
    }
    ab.MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
    ab.fallback.MANIFEST = ab.MANIFEST
    ab.fallback.validate_manifest()


def main():
    ab.MANIFEST = ab.ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_nf4_cache_policy_ab_manifest_v1.json"
    ab.tuned_config = cached_candidate_config
    ab.pin_manifest = pin_cache_manifest
    original_generate = ab.generate_fallback

    def generate_with_cache_policy(model, tokenizer, prompt_ids, common, config_fn, state_store=None):
        integration._fused_nf4_m3_kernel = (
            base_kernel._fused_nf4_m3_kernel if config_fn is ab.default_config else CA_KERNEL
        )
        return original_generate(model, tokenizer, prompt_ids, common, config_fn, state_store)

    ab.generate_fallback = generate_with_cache_policy
    args = sys.argv[1:]
    original_argv = sys.argv
    sys.argv = [str(Path(ab.__file__).resolve()), *args]
    ab.main()
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
        "default": "8x8 activationvec M=3 with ld.global.cs.v4 activation loads",
        "candidate": "8x8 activationvec M=3 with ld.global.ca.v4 activation loads",
    }
    payload["cache_policy"] = {
        "activation_loads_default": "ld.global.cs.v4.u32",
        "activation_loads_candidate": "ld.global.ca.v4.u32",
        "packed_weight_loads": "ld.global.cs.v4.u32 in both",
        "generated_ca_kernel_sha256": hashlib.sha256(CA_SOURCE.encode()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    parsed.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
