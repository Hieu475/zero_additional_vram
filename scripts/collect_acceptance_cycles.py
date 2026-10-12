#!/usr/bin/env python3
"""Validate cycle traces and summarize observed acceptance survival."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    summaries = {}
    for split in ("calibration", "heldout"):
        path = root / "acceptance_cycles" / f"{split}.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        if not rows:
            raise FileNotFoundError(f"No {split} cycle traces at {path}; rerun the roofline collector with cycle tracing")
        buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        seen: set[tuple[str, str, str, int, int]] = set()
        for row in rows:
            key = (row["task"], row["prompt_id"], row["action"], int(row["k"]), int(row["cycle_index"]))
            if key in seen:
                raise ValueError(f"duplicate cycle record: {key}")
            seen.add(key)
            if int(row["accepted_tokens"]) > int(row["draft_tokens"]):
                raise ValueError(f"accepted > drafted for {key}")
            buckets[(str(row["action"]), int(row["k"]))].append(row)
        summaries[split] = {
            "n_cycles": len(rows),
            "n_prompts": len({(r["task"], r["prompt_id"]) for r in rows}),
            "by_action_k": {
                f"{action}_K{k}": {
                    "n_cycles": len(group),
                    "mean_draft_tokens": sum(int(r["draft_tokens"]) for r in group) / len(group),
                    "mean_accepted_tokens": sum(int(r["accepted_tokens"]) for r in group) / len(group),
                    "survival": {
                        str(i): (sum(int(r["accepted_tokens"]) >= i for r in group) /
                                 sum(int(r["draft_tokens"]) >= i for r in group))
                        for i in range(1, k + 1)
                        if any(int(r["draft_tokens"]) >= i for r in group)
                    },
                } for (action, k), group in buckets.items()
            },
        }
    out = root / "acceptance_survival.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(out)


if __name__ == "__main__":
    main()
