#!/usr/bin/env python3
"""Freeze an untouched prompt cohort by ID/hash before algorithm selection.

Freezing IDs prevents later selection by observed mismatch. It cannot prove
that a source dataset was never used in any earlier study; that requires a
separate provenance audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/benchmarks/gsm8k_ext150.jsonl")
    ap.add_argument("--exclude-source", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--salt", default="exact-quantized-20261024")
    ap.add_argument("--output", default="experiments/exact_quantized_feasibility/frozen_gsm8k_ext50.json")
    args = ap.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    excluded = {str(json.loads(line)["id"]) for line in
                Path(args.exclude_source).read_text(encoding="utf-8").splitlines() if line}
    eligible = [row for row in rows if str(row["id"]) not in excluded]
    if len({str(row["id"]) for row in eligible}) != len(eligible):
        raise ValueError("duplicate eligible IDs")
    if args.count < 1 or len(eligible) < args.count:
        raise ValueError("invalid count or too few eligible prompts")
    eligible.sort(key=lambda row: digest(f"{args.salt}|{row['id']}".encode()))
    selected = eligible[:args.count]
    manifest = {
        "status": "reserved: do not inspect outputs or tune on these prompts",
        "selection_rule": "SHA256(salt|id) ascending after exclusion",
        "salt": args.salt,
        "source": str(source),
        "source_sha256": digest(source.read_bytes()),
        "exclude_source": args.exclude_source,
        "exclude_source_sha256": digest(Path(args.exclude_source).read_bytes()),
        "source_count": len(rows),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "overlap_with_excluded_ids": len({str(row["id"]) for row in selected} & excluded),
        "selected": [{"id": str(row["id"]),
                      "prompt_sha256": digest(row["prompt"].encode("utf-8"))}
                     for row in selected],
        "provenance_limit": "Prior use of this source anywhere outside the explicitly excluded file has not been ruled out.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen {len(selected)} IDs to {output}; excluded-ID overlap: 0")


if __name__ == "__main__":
    main()
