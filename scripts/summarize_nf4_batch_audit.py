#!/usr/bin/env python3
"""Aggregate batch-versus-sequential NF4 audit JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize(paths: list[Path]) -> dict[str, Any]:
    configurations = []
    for path in paths:
        audit = json.loads(path.read_text())
        rows = audit["results"]
        horizon_stats = {}
        for k in sorted({key for row in rows for key in row["horizons"]}, key=int):
            items = [(row["id"], row["horizons"][k]) for row in rows if k in row["horizons"]]
            flips = [(prompt_id, logit["position"]) for prompt_id, report in items
                     for logit in report["logits_by_position"]
                     if logit["greedy_batch"] != logit["greedy_sequential"]]
            errors = [logit["max_abs"] for _, report in items for logit in report["logits_by_position"]]
            horizon_stats[k] = {
                "n_prompts": len(items),
                "prefill_logits_bitwise_equal_all": all(
                    report["prefill_logits_bitwise_equal"] for _, report in items),
                "prefill_kv_bitwise_equal_all": all(
                    report["prefill_cache_bitwise_equal"] for _, report in items),
                "first_differing_kv_layer_counts": dict(Counter(
                    str(report["first_kv_differing_layer"]) for _, report in items)),
                "argmax_flip_count": len(flips), "argmax_flips": flips,
                "max_abs_logit_error": max(errors) if errors else None,
            }
        configurations.append({
            "input_file": str(path), "attention_backend": audit["attention_backend"],
            "cache_backend": audit["cache_backend"],
            "model": audit["model"], "model_revision": audit["model_revision"],
            "prompt_ids": [row["id"] for row in rows], "horizons": horizon_stats,
        })
    return {
        "configurations": configurations,
        "interpretation": [
            "Bitwise KV differences are not by themselves evidence of a token error; argmax flips are reported separately.",
            "Three prompts and K<=4 are a targeted diagnosis, not a prevalence estimate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
