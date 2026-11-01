#!/usr/bin/env python3
"""Capture speculative cycle traces for fixed-three-row K2/R3 EOS branch audits."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import audit_nf4_fixed_three_rows_closed_loop as three_row_runner


def main() -> None:
    base = three_row_runner.base_runner
    original_block = base.b1_fixed_row_block_generate
    captured: list[dict] = []

    def traced_block(*args, **kwargs):
        kwargs["trace"] = True
        result = original_block(*args, **kwargs)
        metrics = result[1]
        captured[:] = list(getattr(metrics, "cycle_trace", []))
        return result

    base.b1_fixed_row_block_generate = traced_block
    try:
        three_row_runner.main()
    finally:
        base.b1_fixed_row_block_generate = original_block

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    payload = json.loads(output.read_text())
    eos_id = 151645
    events = []
    for cycle_index, cycle in enumerate(captured):
        draft = cycle.get("draft_tokens", [])
        if eos_id not in draft:
            continue
        eos_index = draft.index(eos_id)
        rejected_at = cycle.get("rejected_at")
        events.append({
            "cycle_index": cycle_index,
            "draft_tokens": draft,
            "eos_candidate_index": eos_index,
            "accepted": cycle.get("accepted", 0),
            "rejected_at": rejected_at,
            "accepted_eos_candidate": rejected_at is None and int(cycle.get("accepted", 0)) > eos_index,
            "rejected_eos_candidate": rejected_at == eos_index,
            "emitted_tokens": cycle.get("emitted_tokens", []),
        })
    payload["terminal_branch_audit"] = {
        "eos_token_id": eos_id,
        "block_terminal_eos": bool(payload["block"]["token_ids"] and payload["block"]["token_ids"][-1] == eos_id),
        "cycle_trace": captured,
        "eos_draft_events": events,
        "accepted_eos_candidate_seen": any(event["accepted_eos_candidate"] for event in events),
        "rejected_eos_candidate_seen": any(event["rejected_eos_candidate"] for event in events),
        "trace_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "token_match": payload["token_match"],
        "all_shared_states_equal": payload["all_shared_states_equal"],
        "block_terminal_eos": payload["terminal_branch_audit"]["block_terminal_eos"],
        "eos_draft_events": events,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
