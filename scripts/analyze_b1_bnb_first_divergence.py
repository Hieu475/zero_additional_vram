"""Locate first token divergence in saved B1 block vs original BNB pilot rows."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    records = []
    for rows_path in sorted(root.glob("session*/rows.jsonl")):
        for line in rows_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            reference = row["results"]["b1_block"]["token_ids"]
            baseline = row["results"]["original_greedy"]["token_ids"]
            position = first_difference(reference, baseline)
            records.append({
                "prompt_id": row["id"],
                "session_index": row["session_index"],
                "b1_length": len(reference),
                "bnb_length": len(baseline),
                "first_difference_index_zero_based": position,
                "common_prefix_tokens": min(len(reference), len(baseline)) if position is None else position,
                "first_64_tokens_equal": reference[:64] == baseline[:64],
                "token_match_b1": row["block_matches_b1"],
                "token_match_bnb": row["block_matches_original_bnb"],
                "b1_kv_state_exact": row["state_audit"]["all_shared_states_equal"],
            })
    if not records:
        raise ValueError(f"no session*/rows.jsonl records found under {root}")
    differences = [r["first_difference_index_zero_based"] for r in records if r["first_difference_index_zero_based"] is not None]
    payload = {
        "status": "token-sequence divergence analysis only; does not locate the first numerical operator drift",
        "run_root": str(root),
        "runs": len(records),
        "all_b1_and_kv_checks_passed": all(r["token_match_b1"] and r["b1_kv_state_exact"] for r in records),
        "all_first_64_tokens_equal_to_bnb": all(r["first_64_tokens_equal"] for r in records),
        "bnb_token_match_count": sum(r["token_match_bnb"] for r in records),
        "first_divergence_index_range": [min(differences), max(differences)] if differences else None,
        "first_divergence_index_median": statistics.median(differences) if differences else None,
        "runs": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
