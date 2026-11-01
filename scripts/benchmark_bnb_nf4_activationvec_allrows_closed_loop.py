#!/usr/bin/env python3
"""Experimental variant: route singleton fixed-row linears through Triton M=3."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch.nn.functional as F

try:
    import benchmark_bnb_nf4_activationvec_closed_loop as baseline
    from audit_bnb_nf4_activationvec_block_integration import PROJECTIONS, module_for
except ModuleNotFoundError:
    from scripts import benchmark_bnb_nf4_activationvec_closed_loop as baseline
    from scripts.audit_bnb_nf4_activationvec_block_integration import PROJECTIONS, module_for


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_allrows_manifest.json"
ORIGINAL_ACTIVATIONVEC_HOOK = baseline.activationvec_schedule_hook


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_variant_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema") != "zassd-activationvec-allrows-variant-manifest-v1":
        raise RuntimeError("unexpected all-rows variant manifest schema")
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"source pin mismatch: {relative}")
    if manifest.get("source_sha256", {}).get("scripts/benchmark_bnb_nf4_activationvec_allrows_closed_loop.py") != sha256(Path(__file__)):
        raise RuntimeError("all-rows runner is not pinned by its variant manifest")
    return manifest


@contextmanager
def all_rows_schedule_hook(model):
    """Pad 1/2 real rows with zeros, call the exact M=3 Triton kernel, crop output."""
    with ORIGINAL_ACTIVATIONVEC_HOOK(model):
        saved = []
        for layer in model.model.layers:
            for projection in PROJECTIONS:
                module = module_for(layer, projection)
                original = module.forward

                def padded_forward(input_tensor, *args, _original=original, **kwargs):
                    if (input_tensor.ndim == 3 and input_tensor.shape[0] == 1
                            and 1 <= input_tensor.shape[1] < 3):
                        real_rows = int(input_tensor.shape[1])
                        padded = F.pad(input_tensor, (0, 0, 0, 3 - real_rows))
                        output = _original(padded, *args, **kwargs)
                        return output[:, :real_rows, :]
                    return _original(input_tensor, *args, **kwargs)

                saved.append((module, original))
                module.forward = padded_forward
        try:
            yield model
        finally:
            for module, original in reversed(saved):
                module.forward = original


def main() -> None:
    manifest = validate_variant_manifest()
    baseline.activationvec_schedule_hook = all_rows_schedule_hook
    try:
        baseline.main()
    finally:
        baseline.activationvec_schedule_hook = ORIGINAL_ACTIVATIONVEC_HOOK

    try:
        output_arg = sys.argv[sys.argv.index("--output") + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("baseline runner did not receive --output") from exc
    output = Path(output_arg)
    artifact = json.loads(output.read_text())
    artifact["classification"] = "experimental development-only K=2/R=3 all-rows Triton M=3 closed-loop audit"
    artifact["linear_override_mode"] = "pad singleton/partial rows with zeros to 3, run Triton M=3, crop real rows"
    artifact["variant_manifest"] = str(MANIFEST)
    artifact["variant_manifest_sha256"] = sha256(MANIFEST)
    artifact["variant_source"] = str(Path(__file__).resolve())
    artifact["variant_source_sha256"] = sha256(Path(__file__))
    artifact["source_sha256"] = sha256(Path(__file__))
    artifact["scope_limits"].append(
        "Experimental M=1/2-to-M=3 zero-padding override; exactness must be checked against the pinned B1 singleton before interpreting timing.")
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "linear_override_mode": artifact["linear_override_mode"],
        "variant_source_sha256": artifact["variant_source_sha256"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
