#!/usr/bin/env python3
"""Summarize B1 verifier rejection and terminal-branch coverage from pilot traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eos-token-id", type=int, default=None,
                        help="Fallback for older manifests without EOS metadata")
    parser.add_argument("--prompt-id", action="append", default=None,
                        help="Include only these prompt IDs; may be repeated")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    by_k: dict[int, dict] = defaultdict(lambda: {
        "runs": [], "prompts": [], "reject_cycles": Counter(),
        "cycle_count": 0, "eos_draft_cycles": 0,
        "eos_outputs": 0, "budget_outputs": 0, "unexpected_stops": 0,
        "budget_final_cycles": 0, "budget_final_short_drafts": 0,
        "b1_token_matches": 0, "b1_token_mismatches": 0,
        "shared_state_audits_passed": 0, "shared_state_audits_failed": 0,
    })
    for run_name in args.runs:
        run_dir = Path(run_name)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        eos = manifest.get("eos_token_id", args.eos_token_id)
        if eos is None:
            raise ValueError(f"missing eos_token_id in {run_dir}; pass --eos-token-id")
        k = int(manifest["k"])
        budget = int(manifest["max_new_tokens"])
        group = by_k[k]
        group["runs"].append({
            "directory": str(run_dir),
            "contract_id": manifest.get("contract_id"),
            "session_index": manifest.get("session_index"),
            "start_index": manifest.get("start_index", 0),
            "budget": budget,
            "attention_strategy": manifest.get("attention_strategy"),
            "physical_rows": manifest.get("physical_rows"),
        })
        for line in (run_dir / "rows.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if args.prompt_id is not None and row.get("id") not in args.prompt_id:
                continue
            block = row["results"]["b1_block"]
            tokens = block["token_ids"]
            if tokens and tokens[-1] == eos:
                terminal = "eos"
                group["eos_outputs"] += 1
            elif len(tokens) == budget:
                terminal = "budget"
                group["budget_outputs"] += 1
            else:
                terminal = "unexpected_stop"
                group["unexpected_stops"] += 1
            if row.get("block_matches_b1"):
                group["b1_token_matches"] += 1
            else:
                group["b1_token_mismatches"] += 1
            audit = row.get("state_audit", {})
            if audit.get("all_shared_states_equal"):
                group["shared_state_audits_passed"] += 1
            elif audit:
                group["shared_state_audits_failed"] += 1

            cycles = block.get("cycles", [])
            for cycle in cycles:
                group["cycle_count"] += 1
                rejection = cycle["rejected_at"]
                key = "accept_all" if rejection is None else f"reject_{rejection}"
                group["reject_cycles"][key] += 1
                if eos in cycle["draft_tokens"]:
                    group["eos_draft_cycles"] += 1
            if terminal == "budget" and cycles:
                last = cycles[-1]
                if last["output_start"] + len(last["emitted_tokens"]) == budget:
                    group["budget_final_cycles"] += 1
                    if len(last["draft_tokens"]) < k:
                        group["budget_final_short_drafts"] += 1
            group["prompts"].append({
                "run": str(run_dir), "task": row["task"], "id": row["id"],
                "generated_tokens": len(tokens), "terminal": terminal,
                "block_matches_b1": row.get("block_matches_b1"),
                "block_matches_original_bnb": row.get("block_matches_original_bnb"),
                "shared_prefixes": audit.get("shared_prefixes"),
                "cycle_branches": [
                    "accept_all" if c["rejected_at"] is None else f"reject_{c['rejected_at']}"
                    for c in cycles
                ],
            })

    for group in by_k.values():
        group["reject_cycles"] = dict(group["reject_cycles"])
    report = {
        "status": "finite diagnostic branch-coverage audit; not a proof",
        "by_k": dict(sorted(by_k.items())),
        "limitations": [
            "Coverage reflects only the listed finite pilot prompts and runtime.",
            "Active KV digest checks occur only at shared cycle frontiers.",
            "EOS-draft coverage is reported separately from emitting EOS as the pending token.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        k: {key: value for key, value in group.items()
            if key not in {"runs", "prompts"}}
        for k, group in report["by_k"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
