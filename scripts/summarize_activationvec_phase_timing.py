#!/usr/bin/env python3
"""Summarize prefill/decode and block phase timing from paired K2/R3 A/B runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifacts", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    runs = [json.loads(path.read_text()) for path in args.artifacts]
    rows = []
    all_exact = True
    for run, path in zip(runs, args.artifacts):
        if run.get("max_new_tokens") != 256:
            raise ValueError(f"expected 256-token budget: {path}")
        run_exact = all(
            item["tokens_match_b1_singleton"]
            and item["state_domains_equal"]
            and not item["state_mismatches"]
            for item in run["exactness_preflight"].values()
        ) and all(count == len(run["paired_rounds"])
                  for count in run["timed_token_match_rounds"].values())
        all_exact = all_exact and run_exact

        round_rows = []
        for timing in run["paired_rounds"]:
            methods = timing["methods"]
            block = methods["fallback_default"]
            bnb = methods["original_bnb"]
            decode_wall = block["wall_after_prefill_s"]
            round_rows.append({
                "round": timing["round"],
                "block_over_bnb_end_to_end": bnb["total_s"] / block["total_s"],
                "block_over_bnb_decode_only": bnb["decode_s"] / decode_wall,
                "block_vs_bnb_decode_tokens_per_s": {
                    "block": 256 / decode_wall,
                    "bnb": 256 / bnb["decode_s"],
                },
                "block_phase_shares_of_post_prefill_wall": {
                    "draft": block["draft_s"] / decode_wall,
                    "verify": block["verify_s"] / decode_wall,
                    "unattributed": block["unattributed_s"] / decode_wall,
                },
                "prefill_s": {"block": block["prefill_s"], "bnb": bnb["prefill_s"]},
                "block_total_s": block["total_s"],
                "bnb_total_s": bnb["total_s"],
                "block_decode_wall_s": decode_wall,
                "bnb_decode_s": bnb["decode_s"],
            })
        rows.append({
            "prompt_id": run["prompt_id"],
            "prompt_length": run["prompt_length"],
            "max_new_tokens": run["max_new_tokens"],
            "acceptance_rate": sum(x["methods"]["fallback_default"]["accepted"]
                                    for x in run["paired_rounds"])
                               / sum(x["methods"]["fallback_default"]["drafted"]
                                     for x in run["paired_rounds"]),
            "mean_cycles": statistics.mean(
                x["methods"]["fallback_default"]["cycles"] for x in run["paired_rounds"]
            ),
            "mean_no_proposal_fallback_cycles": statistics.mean(
                x["methods"]["fallback_default"]["fallback_counts"]["no_proposal_cycles"]
                for x in run["paired_rounds"]
            ),
            "exact_to_b1": run_exact,
            "artifacts": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
            "paired_rounds": round_rows,
            "geomean_block_over_bnb_end_to_end": geomean(
                x["block_over_bnb_end_to_end"] for x in round_rows
            ),
            "geomean_block_over_bnb_decode_only": geomean(
                x["block_over_bnb_decode_only"] for x in round_rows
            ),
            "mean_block_phase_shares": {
                phase: statistics.mean(x["block_phase_shares_of_post_prefill_wall"][phase]
                                       for x in round_rows)
                for phase in ("draft", "verify", "unattributed")
            },
            "mean_prefill_s": {
                method: statistics.mean(x["prefill_s"][method] for x in round_rows)
                for method in ("block", "bnb")
            },
        })

    payload = {
        "schema": "zassd-activationvec-k2r3-phase-timing-summary-v1",
        "classification": "three selected development prompts; same-process paired phase diagnostics",
        "model": runs[0]["model"],
        "revision": runs[0]["revision"],
        "schedule": runs[0]["schedule"],
        "candidate": "default streaming .cs activation-vectorized K2/R3 block fallback",
        "prompt_count": len(rows),
        "process_count": len(runs),
        "all_runs_exact_to_b1_and_timed_tokens": all_exact,
        "aggregate_equal_prompt_geomean": {
            "end_to_end_speedup_vs_original_bnb": geomean(
                row["geomean_block_over_bnb_end_to_end"] for row in rows
            ),
            "decode_only_speedup_vs_original_bnb": geomean(
                row["geomean_block_over_bnb_decode_only"] for row in rows
            ),
        },
        "prompts": rows,
        "limitations": ["one process per prompt and four paired rounds", "one GPU",
                        "three selected development prompts", "decode wall is total minus prefill and includes bookkeeping",
                        "unattributed phase includes any timer residual and is not a named kernel cost",
                        "original-BNB comparison is latency/token only, not KV identity",
                        "GSM8K held-out indices 300-349 excluded"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "exact": all_exact,
        "aggregate": payload["aggregate_equal_prompt_geomean"],
        "prompts": [{"id": row["prompt_id"], "acceptance": row["acceptance_rate"],
                     "e2e": row["geomean_block_over_bnb_end_to_end"],
                     "decode": row["geomean_block_over_bnb_decode_only"],
                     "shares": row["mean_block_phase_shares"]} for row in rows],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
