#!/usr/bin/env python3
"""Validate the live pinned runtime, then run the fixed R=2 B1 audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zassd.backends.base import load_backend
from scripts.run_b1_block_pilot import runtime_identity


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
ATTENTION_STRATEGY = "triton_batched"
LINEAR_STRATEGY = "triton_nf4_group_scale_packed_rows2_n16_warp1_k256"


def canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(contract_path: Path, model: Any, tokenizer: Any) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    payload = {key: value for key, value in contract.items() if key != "contract_id"}
    failures: list[str] = []
    if canonical_hash(payload) != contract.get("contract_id"):
        failures.append("contract_id_integrity")
    if contract.get("schema") != "zassd-b1-triton-reference-v1":
        failures.append("contract_schema")
    if contract.get("model", {}).get("id") != MODEL_ID:
        failures.append("contract_model_id")
    if contract.get("model", {}).get("revision") != MODEL_REVISION:
        failures.append("contract_model_revision")
    if getattr(model.config, "_commit_hash", None) != MODEL_REVISION:
        failures.append("loaded_model_revision")
    schedule = contract.get("operator_schedule", {})
    if (schedule.get("physical_rows") != 2 or
            schedule.get("attention", {}).get("strategy") != ATTENTION_STRATEGY or
            schedule.get("quantized_linear", {}).get("strategy") != LINEAR_STRATEGY):
        failures.append("operator_schedule")
    decoder = contract.get("decoder", {})
    if decoder.get("k") != 1 or decoder.get("ngram_size") != 3:
        failures.append("decoder_schedule")

    live = runtime_identity(
        model, tokenizer, "cuda:0", importlib.metadata.version,
        quantized_linear_strategy=LINEAR_STRATEGY,
    )
    if contract.get("model", {}).get("model_type") != live.get("model_type"):
        failures.append("model_type")
    if contract.get("model", {}).get("config_sha256") != live.get("model_config_sha256"):
        failures.append("model_config_sha256")
    if contract.get("weights", {}).get("quantized_state_sha256") != live.get(
        "quantized_state_sha256"
    ):
        failures.append("quantized_state_sha256")
    if contract.get("tokenizer") != live.get("tokenizer"):
        failures.append("tokenizer_identity")
    for key in (
        "gpu", "python", "torch", "cuda", "transformers", "bitsandbytes",
        "triton", "model_attention_implementation", "backend_flags",
    ):
        if contract.get("runtime", {}).get(key) != live.get(key):
            failures.append(f"runtime.{key}")

    source_mismatches = []
    for name, expected in sorted(contract.get("source_sha256", {}).items()):
        path = Path(name)
        if not path.is_file() or file_sha256(path) != expected:
            source_mismatches.append(name)
    if source_mismatches:
        failures.append("pinned_source_hashes")
    return {
        "status": "pass" if not failures else "fail",
        "contract_id": contract.get("contract_id"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "live_quantized_state_sha256": live.get("quantized_state_sha256"),
        "pinned_source_count": len(contract.get("source_sha256", {})),
        "source_mismatches": source_mismatches,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(
        "experiments/exact_quantized_feasibility/"
        "b1_triton_batched_k1r2_n16_reference_contract_v3.json"
    ))
    parser.add_argument("--prompt-id", choices=("cnndm_000", "cnndm_001", "cnndm_002"),
                        default="cnndm_000")
    parser.add_argument("--max-new-tokens", type=int, choices=(64, 128, 256, 512),
                        default=64)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--timing-rounds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.preflight_output and args.preflight_output.exists():
        raise FileExistsError(args.preflight_output)

    loaded = load_backend("hf_bnb_nf4", MODEL_ID, MODEL_REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    report = preflight(args.contract, model, tokenizer)
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.preflight_output:
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(serialized)
    print(serialized, end="", flush=True)
    if report["status"] != "pass":
        raise SystemExit("contract preflight failed; decode was not started")

    command = [
        sys.executable, "scripts/audit_triton_b1_closed_loop.py",
        "--benchmark", "data/benchmarks/cnndm_eval.jsonl",
        "--prompt-id", args.prompt_id,
        "--k", "1", "--ngram-size", "3", "--physical-rows", "2",
        "--max-new-tokens", str(args.max_new_tokens), "--block-k", "32",
        "--session", str(args.session), "--timing-rounds", str(args.timing_rounds),
        "--attention-strategy", ATTENTION_STRATEGY,
        "--quantized-linear-strategy", LINEAR_STRATEGY,
        "--output", str(args.output),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
