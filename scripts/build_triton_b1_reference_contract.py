#!/usr/bin/env python3
"""Pin a finite composite B1 reference identity for the Triton closed-loop runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True,
                        help="runtime_identity output with quantized model-state hash")
    parser.add_argument("--source-run", type=Path, required=True,
                        help="closed-loop run JSON whose source hashes pin the decode path")
    parser.add_argument("--validation-artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    identity = json.loads(args.identity.read_text())
    run = json.loads(args.source_run.read_text())
    params = run["parameters"]
    expected = {
        "k": 1,
        "ngram_size": 3,
        "physical_rows": 2,
        "attention_strategy": "triton_batched",
        "quantized_linear_strategy": "triton_nf4_group_scale_packed_rows2_n16_warp1_k256",
    }
    mismatches = [key for key, value in expected.items() if params.get(key) != value]
    if mismatches:
        raise ValueError("source run does not describe the intended schedule: " + ", ".join(mismatches))
    if (run["model"] != "Qwen/Qwen2.5-3B-Instruct" or
            run["revision"] != "aa8e72537993ba99e69dfaafa59ed015b17504d1"):
        raise ValueError("unexpected model or revision")

    sources = dict(identity["source_sha256"])
    for name, digest in run["source_sha256"].items():
        if name in sources and sources[name] != digest:
            raise ValueError(f"source hash disagreement for {name}")
        sources[name] = digest
    for name, digest in sources.items():
        path = Path(name)
        if not path.exists() or sha256(path) != digest:
            raise ValueError(f"working tree source does not match recorded hash: {name}")

    evidence = []
    branch_counts = {
        "runs": 0, "cycles": 0, "accepted_draft_tokens": 0,
        "draft_tokens": 0, "no_candidate_cycles": 0,
        "rejected_at": {}, "budget_terminal_runs": 0,
        "budget_finish_in_accepted_cycle_runs": 0,
        "budget_finish_via_final_pending_runs": 0,
        "eos_terminal_runs": 0, "eos_draft_candidates": 0,
        "eos_draft_candidates_accepted": 0, "eos_draft_candidates_rejected": 0,
    }
    for path in args.validation_artifact:
        item = json.loads(path.read_text())
        if item.get("model") != run["model"] or item.get("revision") != run["revision"]:
            raise ValueError(f"validation artifact model mismatch: {path}")
        if any(item["parameters"].get(key) != value for key, value in expected.items()):
            raise ValueError(f"validation artifact schedule mismatch: {path}")
        if not (item["token_match"] and item["all_shared_states_equal"] and
                item["logical_lengths_exact_at_shared_commit_frontiers"]):
            raise ValueError(f"validation artifact failed B1 identity checks: {path}")
        tokens = item["block"]["token_ids"]
        budget = item["parameters"]["max_new_tokens"]
        eos = identity["tokenizer"]["eos_token_id"]
        if tokens and tokens[-1] == eos:
            terminal = "eos"
        elif len(tokens) == budget:
            terminal = "budget"
        else:
            raise ValueError(f"validation artifact ended unexpectedly: {path}")
        cycles = item["block"].get("cycles", [])
        if terminal == "budget":
            last_cycle = cycles[-1] if cycles else {}
            cycle_end = last_cycle.get("output_start", 0) + len(last_cycle.get("emitted_tokens", []))
            if cycle_end == budget:
                boundary = "verified_cycle"
                branch_counts["budget_finish_in_accepted_cycle_runs"] += int(
                    int(last_cycle.get("accepted", 0)) > 0
                )
            elif cycle_end == budget - 1:
                boundary = "final_pending"
                branch_counts["budget_finish_via_final_pending_runs"] += 1
            else:
                raise ValueError(f"unrecognized budget boundary in {path}: {cycle_end}/{budget}")
        else:
            boundary = "eos"
        branch_counts["runs"] += 1
        branch_counts["cycles"] += len(cycles)
        branch_counts[f"{terminal}_terminal_runs"] += 1
        for cycle in cycles:
            accepted = int(cycle.get("accepted", 0))
            drafted = len(cycle.get("draft_tokens", []))
            branch_counts["accepted_draft_tokens"] += accepted
            branch_counts["draft_tokens"] += drafted
            branch_counts["no_candidate_cycles"] += drafted == 0
            reject = cycle.get("rejected_at")
            if reject is not None:
                key = str(reject)
                branch_counts["rejected_at"][key] = branch_counts["rejected_at"].get(key, 0) + 1
            eos_candidate = eos in cycle.get("draft_tokens", [])
            branch_counts["eos_draft_candidates"] += eos_candidate
            if eos_candidate:
                if reject is None and accepted:
                    branch_counts["eos_draft_candidates_accepted"] += 1
                else:
                    branch_counts["eos_draft_candidates_rejected"] += 1
        evidence.append({
            "path": str(path),
            "sha256": sha256(path),
            "prompt_id": item["prompt_id"],
            "max_new_tokens": item["parameters"]["max_new_tokens"],
            "session": item["session"],
            "token_match": item["token_match"],
            "all_shared_states_equal": item["all_shared_states_equal"],
            "logical_lengths_exact": item["logical_lengths_exact_at_shared_commit_frontiers"],
            "shared_prefixes": item["shared_state_prefixes"],
            "terminal_boundary": boundary,
        })

    payload = {
        "schema": "zassd-b1-triton-reference-v1",
        "status": "experimental finite reference identity; not a proof or BNB equivalence",
        "reference": "greedy singleton under the pinned Qwen2 Triton fixed-row schedule",
        "model": {
            "id": run["model"],
            "revision": run["revision"],
            "model_type": identity["model_type"],
            "config_sha256": identity["model_config_sha256"],
        },
        "weights": {
            "quantization": "bitsandbytes NF4, double_quant=True, FP16 compute",
            "quantized_state_sha256": identity["quantized_state_sha256"],
        },
        "tokenizer": identity["tokenizer"],
        "runtime": {
            key: identity[key] for key in (
                "gpu", "python", "torch", "cuda", "transformers", "bitsandbytes",
                "triton", "model_attention_implementation", "backend_flags",
            )
        },
        "source_sha256": sources,
        "operator_schedule": {
            "name": "qwen2_triton_batched_nf4_n16_warp1_k1_r2_singleton",
            "attention": {
                "strategy": "triton_batched",
                "kernel": "fixed_row_attention_tiled",
                "block_k": 32,
                "causal_prefix": "each query row attends through its logical prefix",
            },
            "quantized_linear": {
                "strategy": expected["quantized_linear_strategy"],
                "output_tile_n": 16,
                "reduction_tile_k": 256,
                "num_warps": 1,
                "rows_per_program": 2,
                "fp_fusion": False,
            },
            "physical_rows": 2,
            "normalization_and_lm_head": "fixed to the declared physical row count",
        },
        "decoder": {
            "k": params["k"],
            "ngram_size": params["ngram_size"],
            "tie_rule": "torch.argmax_first_index",
            "eos_token_id": identity["tokenizer"]["eos_token_id"],
            "sampling": "disabled",
        },
        "cache": {
            "backend": "StaticPreallocatedKVCache",
            "batch_size": 1,
            "capacity_policy": "prompt_length + requested output budget + 4; maximum 2048",
            "logical_length": "must equal committed prefix length",
        },
        "supported_domain": {
            "architecture": "Qwen2 only",
            "batch_size": 1,
            "external_attention_mask": False,
            "external_position_ids": False,
            "custom_cache_position": False,
            "inputs_embeds": False,
            "sliding_window": False,
            "output_attentions": False,
            "empirically_checked_budgets": [64, 128, 256, 512],
        },
        "finite_validation": {
            "note": "Empirical B1 singleton/block checks only; does not prove all inputs or B2 identity.",
            "natural_branch_coverage": branch_counts,
            "branches_not_observed": ["accepted EOS candidate", "rejected EOS candidate", "EOS terminal"],
            "artifacts": evidence,
        },
        "provenance": {
            "runtime_identity": str(args.identity),
            "source_run": str(args.source_run),
        },
    }
    contract = {**payload, "contract_id": canonical_hash(payload)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    print(contract["contract_id"])


if __name__ == "__main__":
    main()
