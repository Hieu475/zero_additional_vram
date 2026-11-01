#!/usr/bin/env python3
"""Fail-closed audit of a saved Triton B1 reference contract and identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(contract_path: Path, identity_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    identity = json.loads(identity_path.read_text())
    payload = {key: value for key, value in contract.items() if key != "contract_id"}
    failures: list[str] = []

    actual_id = canonical_hash(payload)
    if actual_id != contract.get("contract_id"):
        failures.append("contract_id_integrity")
    if contract.get("schema") != "zassd-b1-triton-reference-v1":
        failures.append("unsupported_schema")
    if not str(contract.get("status", "")).startswith("experimental finite reference"):
        failures.append("unexpected_contract_status")

    model = contract.get("model", {})
    if model.get("model_type") != identity.get("model_type"):
        failures.append("model_type")
    if model.get("config_sha256") != identity.get("model_config_sha256"):
        failures.append("model_config_sha256")
    if contract.get("weights", {}).get("quantized_state_sha256") != identity.get(
        "quantized_state_sha256"
    ):
        failures.append("quantized_state_sha256")
    if contract.get("tokenizer") != identity.get("tokenizer"):
        failures.append("tokenizer_identity")
    runtime = contract.get("runtime", {})
    for key in (
        "gpu", "python", "torch", "cuda", "transformers", "bitsandbytes",
        "triton", "model_attention_implementation", "backend_flags",
    ):
        if runtime.get(key) != identity.get(key):
            failures.append(f"runtime.{key}")

    pinned_sources = contract.get("source_sha256", {})
    identity_sources = identity.get("source_sha256", {})
    if any(identity_sources.get(name) != digest for name, digest in pinned_sources.items()
           if name in identity_sources):
        failures.append("identity_source_hash_disagreement")
    source_mismatches = []
    missing_sources = []
    for name, expected in sorted(pinned_sources.items()):
        path = Path(name)
        if not path.is_file():
            missing_sources.append(name)
        elif sha256(path) != expected:
            source_mismatches.append(name)
    if missing_sources:
        failures.append("missing_source_files")
    if source_mismatches:
        failures.append("source_hash_mismatch")

    expected_schedule = {
        "physical_rows": 2,
        "attention": {"strategy": "triton_batched"},
        "quantized_linear": {
            "strategy": "triton_nf4_group_scale_packed_rows2_n16_warp1_k256",
            "num_warps": 1,
            "output_tile_n": 16,
            "reduction_tile_k": 256,
        },
    }
    schedule = contract.get("operator_schedule", {})
    for key, value in expected_schedule.items():
        if key == "attention":
            mismatch = schedule.get(key, {}).get("strategy") != value["strategy"]
        elif key == "quantized_linear":
            mismatch = any(schedule.get(key, {}).get(k) != v for k, v in value.items())
        else:
            mismatch = schedule.get(key) != value
        if mismatch:
            failures.append(f"operator_schedule.{key}")
    decoder = contract.get("decoder", {})
    if decoder.get("k") != 1 or decoder.get("ngram_size") != 3:
        failures.append("decoder_parameters")

    return {
        "validator": "saved_identity_and_source_audit",
        "status": "pass" if not failures else "fail",
        "contract_path": str(contract_path),
        "identity_path": str(identity_path),
        "contract_id": contract.get("contract_id"),
        "recomputed_contract_id": actual_id,
        "pinned_source_count": len(pinned_sources),
        "source_mismatches": source_mismatches,
        "missing_sources": missing_sources,
        "failures": failures,
        "scope_note": (
            "Validates the saved identity artifact and current pinned source files; "
            "it does not measure a newly loaded model or prove exactness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.contract, args.identity)
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
