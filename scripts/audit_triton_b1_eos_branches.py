#!/usr/bin/env python3
"""Audit EOS draft/commit branches under the live-gated R=2 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_triton_b1_closed_loop import (
    diagnostic_cache_prefix_digest,
    fixed_row_sdpa_decode,
)
from zassd.certification.triton_b1_reference_contract import (
    ATTENTION_STRATEGY,
    LINEAR_STRATEGY,
    MODEL_ID,
    MODEL_REVISION,
    VerifiedTritonB1Contract,
    verify_triton_b1_reference_contract,
)
from zassd.decoding.b1_triton_contract import (
    b1_triton_contract_block,
    b1_triton_contract_singleton,
)
from zassd.backends.base import load_backend


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_one(model, tokenizer, row: dict, *, max_new_tokens: int,
              verified_contract: VerifiedTritonB1Contract) -> dict:
    device = "cuda:0"
    prompt_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to(device)
    capacity = int(prompt_ids.shape[1]) + max_new_tokens + 4
    reference_states: dict[int, str] = {}
    block_states: dict[int, str] = {}
    reference_lengths: dict[int, list[int]] = {}
    block_lengths: dict[int, list[int]] = {}

    def observer(store, lengths):
        def observe(n: int, cache) -> None:
            store[n] = diagnostic_cache_prefix_digest(
                cache, int(prompt_ids.shape[1]) + n
            )
            lengths[n] = [int(layer.get_seq_length()) for layer in cache.cache.layers]
        return observe

    common = {
        "verified_contract": verified_contract,
        "max_new_tokens": max_new_tokens,
        "device": device,
        "kv_cache_max_capacity": capacity,
    }
    with fixed_row_sdpa_decode(model, 2, 32, ATTENTION_STRATEGY):
        _, reference_metrics = b1_triton_contract_singleton(
            model, tokenizer, prompt_ids, **common,
            state_observer=observer(reference_states, reference_lengths),
        )
    with fixed_row_sdpa_decode(model, 2, 32, ATTENTION_STRATEGY):
        _, block_metrics = b1_triton_contract_block(
            model, tokenizer, prompt_ids, **common, trace=True,
            state_observer=observer(block_states, block_lengths),
        )

    eos = int(tokenizer.eos_token_id)
    eos_draft_events = []
    for cycle_index, cycle in enumerate(block_metrics.cycle_trace):
        candidates = cycle.get("draft_tokens", [])
        if eos not in candidates:
            continue
        eos_index = candidates.index(eos)
        reject_at = cycle.get("rejected_at")
        eos_draft_events.append({
            "cycle_index": cycle_index,
            "draft_tokens": candidates,
            "eos_candidate_index": eos_index,
            "accepted": cycle.get("accepted", 0),
            "rejected_at": reject_at,
            "accepted_eos": reject_at is None and int(cycle.get("accepted", 0)) > eos_index,
            "rejected_eos": reject_at == eos_index,
            "emitted_tokens": cycle.get("emitted_tokens", []),
        })
    shared = sorted(set(reference_states) & set(block_states))
    return {
        "id": row["id"],
        "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
        "prompt_tokens": int(prompt_ids.shape[1]),
        "reference_token_ids": reference_metrics.generated_token_ids,
        "block_token_ids": block_metrics.generated_token_ids,
        "token_match": reference_metrics.generated_token_ids == block_metrics.generated_token_ids,
        "reference_terminal_eos": bool(reference_metrics.generated_token_ids and
                                        reference_metrics.generated_token_ids[-1] == eos),
        "block_terminal_eos": bool(block_metrics.generated_token_ids and
                                    block_metrics.generated_token_ids[-1] == eos),
        "eos_draft_events": eos_draft_events,
        "block_cycle_trace": block_metrics.cycle_trace,
        "reference_prefixes": sorted(reference_states),
        "block_prefixes": sorted(block_states),
        "all_shared_kv_exact": all(reference_states[n] == block_states[n] for n in shared),
        "same_prefix_domain": reference_states.keys() == block_states.keys(),
        "logical_lengths_exact": all(reference_lengths.get(n) == block_lengths.get(n)
                                      for n in shared),
        "shared_prefix_count": len(shared),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(
        "experiments/exact_quantized_feasibility/"
        "b1_triton_batched_k1r2_n16_reference_contract_v3.json"
    ))
    parser.add_argument("--prompt-file", type=Path, action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    loaded = load_backend("hf_bnb_nf4", MODEL_ID, MODEL_REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    verified_contract = verify_triton_b1_reference_contract(
        model, tokenizer, args.contract, device="cuda:0"
    )
    prompt_files = args.prompt_file or [Path(
        "experiments/exact_quantized_feasibility/b1_eos_candidate_prompt_v2.jsonl"
    )]
    rows = [json.loads(line) for prompt_file in prompt_files
            for line in prompt_file.read_text().splitlines() if line.strip()]
    results = [audit_one(model, tokenizer, row, max_new_tokens=args.max_new_tokens,
                         verified_contract=verified_contract)
               for row in rows]
    events = [event for result in results for event in result["eos_draft_events"]]
    report = {
        "experiment": "triton_b1_contract_gated_eos_branch_audit",
        "status": "pass" if all(
            item["token_match"] and item["all_shared_kv_exact"] and
            item["same_prefix_domain"] and item["logical_lengths_exact"]
            for item in results
        ) else "mismatch",
        "scope": "development-only; finite branch audit; no proof or BNB equivalence",
        "contract_id": verified_contract.contract_id,
        "live_preflight": {
            "status": "pass",
            "model_state_sha256": verified_contract.model_state_sha256,
            "verified_source_count": verified_contract.verified_source_count,
        },
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameters": {
            "k": 1, "ngram_size": 3, "physical_rows": 2,
            "attention_strategy": ATTENTION_STRATEGY,
            "quantized_linear_strategy": LINEAR_STRATEGY,
            "max_new_tokens": args.max_new_tokens,
        },
        "prompt_files": [
            {"path": str(prompt_file), "sha256": sha256_file(prompt_file)}
            for prompt_file in prompt_files
        ],
        "runner_sha256": sha256_file(Path(__file__)),
        "branch_coverage": {
            "accepted_eos_candidate": any(e["accepted_eos"] for e in events),
            "rejected_eos_candidate": any(e["rejected_eos"] for e in events),
            "eos_terminal": any(item["block_terminal_eos"] for item in results),
            "eos_draft_events": len(events),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "branch_coverage": report["branch_coverage"],
        "runs": len(results),
        "token_match": all(x["token_match"] for x in results),
        "kv_exact": all(x["all_shared_kv_exact"] for x in results),
    }, indent=2))
    if report["status"] != "pass":
        raise SystemExit("EOS audit found token or state mismatch")


if __name__ == "__main__":
    main()
