#!/usr/bin/env python3
"""Aggregate exactness and paired timings from native-BNB closed-loop sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geomean requires positive samples")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", action="append", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(args.manifest).read_text())
    source_name = "scripts/summarize_native_bnb_rowwise_closed_loop.py"
    if manifest.get("source_sha256", {}).get(source_name) != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("summary script is not pinned by experiment manifest")
    sessions = []
    for name in args.artifact:
        path = Path(name)
        if not path.is_absolute():
            path = repo / path
        data = json.loads(path.read_text())
        exact = data["exactness_audit"]
        if not (exact["token_match"] and exact["all_shared_states_equal"]
                and exact["same_prefix_domain"] and exact["logical_lengths_equal"]):
            raise ValueError(f"session lacks full token/state identity: {path}")
        sessions.append((path, data))
    prompt_ids = {data["prompt_id"] for _, data in sessions}
    budgets = {data["parameters"]["max_new_tokens"] for _, data in sessions}
    if len(prompt_ids) != 1 or len(budgets) != 1:
        raise ValueError("aggregate only sessions with the same prompt and token budget")
    ratios_singleton = [x for _, d in sessions for x in d["timing"]["block_speedup_vs_singleton"]]
    ratios_bnb = [x for _, d in sessions for x in d["timing"]["block_speedup_vs_original_bnb"]]
    session_summaries = []
    paired_rounds = []
    for path, data in sessions:
        exact = data["exactness_audit"]
        singleton_times = [round_data["methods"]["native_bnb_singleton"]["total_s"]
                           for round_data in data["timing"]["paired_rounds"]]
        singleton_median = statistics.median(singleton_times)
        for round_data in data["timing"]["paired_rounds"]:
            methods = round_data["methods"]
            singleton_time = methods["native_bnb_singleton"]["total_s"]
            paired_rounds.append({
                "session": data["parameters"]["session"],
                "round": round_data["round"],
                "method_order": round_data["method_order"],
                "native_bnb_singleton_s": singleton_time,
                "native_bnb_block_s": methods["native_bnb_block"]["total_s"],
                "original_bnb_s": methods["original_bnb_greedy"]["total_s"],
                "block_speedup_vs_singleton": methods["native_bnb_singleton"]["total_s"] /
                                              methods["native_bnb_block"]["total_s"],
                "block_speedup_vs_original_bnb": methods["original_bnb_greedy"]["total_s"] /
                                                 methods["native_bnb_block"]["total_s"],
                "singleton_time_gt_2x_session_median": singleton_time > 2 * singleton_median,
            })
        session_summaries.append({
            "artifact": str(path.relative_to(repo)),
            "session": data["parameters"]["session"],
            "generated_tokens": len(exact["singleton_tokens"]),
            "token_match": exact["token_match"],
            "state_prefixes_compared": exact["shared_state_prefixes"],
            "state_mismatches": exact["state_mismatches"],
            "full_prefix_domain": exact["same_prefix_domain"],
            "logical_lengths_equal": exact["logical_lengths_equal"],
            "accepted_draft_tokens": exact["acceptance"]["accepted"],
            "draft_tokens": exact["acceptance"]["drafted"],
            "paired_speedup_vs_native_singleton": data["timing"]["block_speedup_vs_singleton"],
            "paired_speedup_vs_original_bnb": data["timing"]["block_speedup_vs_original_bnb"],
        })
    result = {
        "classification": "two-session, one-prompt native-BNB exact speculative block audit",
        "prompt_id": next(iter(prompt_ids)),
        "max_new_tokens": next(iter(budgets)),
        "sessions": session_summaries,
        "paired_round_count": len(ratios_singleton),
        "paired_geomean_speedup_vs_native_singleton": geomean(ratios_singleton),
        "paired_geomean_speedup_vs_native_singleton_excluding_flagged_rounds": geomean([
            row["block_speedup_vs_singleton"] for row in paired_rounds
            if not row["singleton_time_gt_2x_session_median"]
        ]),
        "paired_geomean_speedup_vs_original_bnb": geomean(ratios_bnb),
        "speedup_range_vs_native_singleton": [min(ratios_singleton), max(ratios_singleton)],
        "speedup_range_vs_original_bnb": [min(ratios_bnb), max(ratios_bnb)],
        "paired_rounds": paired_rounds,
        "limitations": ["Two independent process sessions but one fixed development prompt and one GPU.",
                        "Paired rounds are nested within sessions; this does not estimate prompt-level generalization.",
                        "Empirical finite exactness is not a proof across prompts, runtimes, GPUs or library versions.",
                        "No held-out GSM8K prompts used."],
    }
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "classification", "prompt_id", "max_new_tokens", "paired_round_count",
        "paired_geomean_speedup_vs_native_singleton",
        "paired_geomean_speedup_vs_original_bnb",
        "paired_geomean_speedup_vs_native_singleton_excluding_flagged_rounds",
        "speedup_range_vs_native_singleton", "speedup_range_vs_original_bnb",
    )}, indent=2))


if __name__ == "__main__":
    main()
