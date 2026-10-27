#!/usr/bin/env python3
"""Build or validate a pinned B1 singleton reference contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def reference_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    identity = manifest["runtime_identity"]
    payload = {
        "schema": "b1-reference-contract-v1",
        "status": "experimental finite reference identity; not a proof",
        "reference": "greedy singleton under the fixed-row Qwen2 operator patch",
        "model": {"id": manifest["model"], "revision": manifest["revision"],
                  "model_type": identity["model_type"],
                  "config_sha256": identity["model_config_sha256"]},
        "weights": {
            "quantization": manifest["backend"]["quantization"],
            "quantized_state_sha256": identity["quantized_state_sha256"],
        },
        "tokenizer": identity["tokenizer"],
        "runtime": {
            key: identity[key]
            for key in ("gpu", "python", "torch", "cuda", "transformers",
                        "bitsandbytes", "model_attention_implementation",
                        "backend_flags", "source_sha256")
        },
        "operator_schedule": {
            "name": manifest["reference_schedule"],
            "attention_strategy": manifest["attention_strategy"],
            "physical_rows": manifest["physical_rows"],
        },
        "cache": {"backend": manifest["cache_backend"],
                  "capacity": manifest["capacity"], "batch_size": 1,
                  "logical_length": "must equal committed prefix length"},
        "greedy": {"tie_rule": manifest["greedy_tie_rule"],
                   "eos_token_id": manifest["eos_token_id"],
                   "sampling": "disabled", "stop_rules": ["EOS", "caller token budget"]},
        "supported_domain": {
            "architecture": "Qwen2 only",
            "batch_size": 1,
            "attention": "SDPA padded fixed-row path",
            "external_attention_mask": False,
            "external_position_ids": False,
            "custom_cache_position": False,
            "inputs_embeds": False,
            "sliding_window": False,
            "output_attentions": False,
        },
    }
    # The pilot runner records its own source hash, which changes when the
    # optional contract check itself is added. Decode/operator sources remain
    # pinned; the exact runner hash stays in the source run manifest provenance.
    payload["runtime"]["source_sha256"].pop("scripts/run_b1_block_pilot.py", None)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--validate-manifest", type=Path)
    args = parser.parse_args()

    if args.source_manifest:
        if not args.output:
            parser.error("--output is required with --source-manifest")
        source = load_json(args.source_manifest)
        payload = reference_payload(source)
        contract = {
            **payload,
            "reference_contract_id": canonical_hash(payload),
            "provenance": {
                "source_manifest": str(args.source_manifest),
                "source_contract_id": source.get("contract_id"),
                "source_prompt_file": source.get("prompt_file"),
                "source_input_hashes": source.get("input_hashes", {}),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(contract, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(contract["reference_contract_id"])

    if args.validate_manifest:
        if not args.contract:
            parser.error("--contract is required with --validate-manifest")
        contract = load_json(args.contract)
        manifest = load_json(args.validate_manifest)
        expected = {key: value for key, value in contract.items()
                    if key not in {"reference_contract_id", "provenance"}}
        actual = reference_payload(manifest)
        actual_id = canonical_hash(expected)
        mismatches = [key for key in expected if expected[key] != actual.get(key)]
        if actual_id != contract.get("reference_contract_id"):
            mismatches.append("reference_contract_id integrity")
        if mismatches:
            raise SystemExit("REFERENCE CONTRACT MISMATCH: " + ", ".join(mismatches))
        print(f"REFERENCE CONTRACT MATCH: {contract['reference_contract_id']}")

    if not args.source_manifest and not args.validate_manifest:
        parser.error("Provide --source-manifest or --validate-manifest")


if __name__ == "__main__":
    main()
