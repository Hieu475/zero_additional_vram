#!/usr/bin/env python3
"""Compare B1 singleton token and active-KV traces across independent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Any

from build_b1_reference_contract import canonical_hash, reference_payload


def read_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines() if line.strip()]
    return manifest, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--prompt-id", action="append", default=None,
                        help="Optional prompt ID filter; may be repeated")
    parser.add_argument("--reference-contract", type=Path,
                        help="Validate runs against a contract, including seed runs created before pinning it")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    run_info = []
    reference = json.loads(args.reference_contract.read_text()) if args.reference_contract else None
    expected = ({key: value for key, value in reference.items()
                 if key not in {"reference_contract_id", "provenance"}} if reference else None)
    for path in args.runs:
        manifest, rows = read_run(path)
        contract_valid = None
        reference_id = manifest.get("reference_contract_id")
        if reference is not None:
            actual = reference_payload(manifest)
            contract_valid = actual == expected and canonical_hash(expected) == reference.get("reference_contract_id")
            reference_id = reference.get("reference_contract_id") if contract_valid else None
        run_info.append({
            "directory": str(path),
            "session_index": manifest.get("session_index"),
            "reference_contract_id": reference_id,
            "reference_contract_valid": contract_valid,
            "contract_id": manifest.get("contract_id"),
            "rows": len(rows),
        })
        for row in rows:
            if args.prompt_id and row["id"] not in args.prompt_id:
                continue
            grouped[(row["task"], row["id"])].append(row)

    prompt_results = []
    for (task, prompt_id), rows in sorted(grouped.items()):
        token_seqs = [row["results"]["b1_reference"]["token_ids"] for row in rows]
        state_traces = [row.get("state_audit", {}).get("reference_state_trace") for row in rows]
        token_equal = all(seq == token_seqs[0] for seq in token_seqs[1:])
        state_equal = all(trace == state_traces[0] for trace in state_traces[1:])
        input_hashes_equal = len({row["input_token_ids_sha256"] for row in rows}) == 1
        prompt_results.append({
            "task": task, "id": prompt_id, "sessions": len(rows),
            "input_hashes_equal": input_hashes_equal,
            "reference_tokens_equal": token_equal,
            "reference_states_equal": state_equal,
            "reference_token_count": len(token_seqs[0]) if token_seqs else 0,
            "reference_prefix_count": len(state_traces[0]) if state_traces[0] else 0,
            "all_reference_prefixes": [
                item["prefix_length"] for item in state_traces[0]
            ] if state_traces[0] else [],
            "block_matches_reference_all_runs": all(row.get("block_matches_b1") for row in rows),
            "shared_state_audits_pass_all_runs": all(
                row.get("state_audit", {}).get("all_shared_states_equal", False) for row in rows
            ),
        })

    all_contract_ids = {item["reference_contract_id"] for item in run_info}
    all_contracts_valid = all(item["reference_contract_valid"] is not False for item in run_info)
    payload = {
        "status": "finite independent-process reproduction; not a proof",
        "runs": run_info,
        "same_reference_contract": len(all_contract_ids) == 1 and None not in all_contract_ids and all_contracts_valid,
        "prompts": prompt_results,
        "summary": {
            "prompt_count": len(prompt_results),
            "token_exact_prompts": sum(item["reference_tokens_equal"] for item in prompt_results),
            "state_exact_prompts": sum(item["reference_states_equal"] for item in prompt_results),
            "input_identity_prompts": sum(item["input_hashes_equal"] for item in prompt_results),
            "all_conditions_pass": (
                len(all_contract_ids) == 1 and None not in all_contract_ids and all_contracts_valid and
                all(item["input_hashes_equal"] and item["reference_tokens_equal"] and
                    item["reference_states_equal"] and
                    item["block_matches_reference_all_runs"] and
                    item["shared_state_audits_pass_all_runs"] for item in prompt_results)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
