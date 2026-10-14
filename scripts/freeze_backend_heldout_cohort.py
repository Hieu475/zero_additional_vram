#!/usr/bin/env python3
"""Freeze a stratified cohort excluded from backend diagnostic selection."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl"
DIAGNOSTIC = ROOT / "experiments/backend_validation/paired_benchmark/diagnostic_cohort.txt"
OUT = ROOT / "experiments/backend_validation/paired_benchmark"
SEED = 20261020
PER_TASK = 4


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    excluded = {line.strip() for line in DIAGNOSTIC.read_text().splitlines() if line.strip()}
    eligible = defaultdict(list)
    for row in rows:
        if row["prompt_id"] not in excluded:
            eligible[row["source_task"]].append(row)
    rng = random.Random(SEED)
    chosen = []
    for task in sorted(eligible):
        candidates = sorted(eligible[task], key=lambda x: x["prompt_id"])
        if len(candidates) < PER_TASK:
            raise ValueError(f"only {len(candidates)} eligible prompts for task {task}")
        chosen.extend(rng.sample(candidates, PER_TASK))
    rng.shuffle(chosen)
    ids = [row["prompt_id"] for row in chosen]
    (OUT / "heldout_cohort.txt").write_text("\n".join(ids) + "\n")
    manifest = {
        "purpose": "backend evaluation cohort excluded from the 16-prompt backend diagnostic cohort",
        "claim_scope": "held out from backend selection; prompts may have appeared in earlier ZASSD studies",
        "seed": SEED,
        "stratification": "4 prompts per source task",
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "diagnostic_exclusion_file": str(DIAGNOSTIC.relative_to(ROOT)),
        "diagnostic_exclusion_sha256": hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest(),
        "prompt_count": len(ids),
        "prompt_ids": ids,
        "source_task_counts": {task: PER_TASK for task in sorted(eligible)},
        "prompt_token_counts": {row["prompt_id"]: row["prompt_tokens"] for row in chosen},
    }
    (OUT / "heldout_cohort_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prompt_count": len(ids), "tasks": manifest["source_task_counts"], "ids": ids}, indent=2))


if __name__ == "__main__":
    main()
