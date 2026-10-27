#!/usr/bin/env python3
"""Freeze a new GSM8K test slice for a future, untouched exactness evaluation.

The slice is selected by public source index, before running a candidate
decoder. This script rejects IDs or prompt text already in local benchmark
files; it does not claim that public GSM8K questions were unseen by a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", default="data/benchmarks/gsm8k_exactness_holdout.jsonl")
    parser.add_argument("--manifest", default="experiments/exact_quantized_feasibility/frozen_gsm8k_300_349.json")
    args = parser.parse_args()
    output, manifest = Path(args.output), Path(args.manifest)
    if output.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite a frozen cohort or manifest")
    if args.start < 0 or args.count < 1:
        raise ValueError("start and count must be nonnegative and positive")

    prior_ids: set[str] = set()
    prior_prompt_hashes: set[str] = set()
    prior_files = sorted(Path("data/benchmarks").glob("*.jsonl"))
    for path in prior_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            prior_ids.add(str(row.get("id", "")))
            if "prompt" in row:
                prior_prompt_hashes.add(sha256(row["prompt"].encode("utf-8")))

    stop = args.start + args.count
    revision = HfApi().dataset_info("openai/gsm8k").sha
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[{args.start}:{stop}]",
                           revision=revision)
    if len(dataset) != args.count:
        raise ValueError(f"expected {args.count} rows, got {len(dataset)}")
    records = []
    for index, item in enumerate(dataset, args.start):
        prompt = f"Question: {item['question']}\nLet's think step by step.\nAnswer:"
        row = {"id": f"gsm8k_{index:03d}", "benchmark": "gsm8k", "prompt": prompt}
        if row["id"] in prior_ids or sha256(prompt.encode("utf-8")) in prior_prompt_hashes:
            raise ValueError(f"local prior benchmark overlap at {row['id']}")
        records.append(row)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate selected IDs")
    if len({sha256(r["prompt"].encode("utf-8")) for r in records}) != len(records):
        raise ValueError("duplicate selected prompts")

    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    frozen = {
        "status": "frozen; do not inspect decoder outputs or tune on these prompts",
        "source": "openai/gsm8k", "source_revision": revision,
        "config": "main", "split": "test",
        "index_start_inclusive": args.start, "index_stop_exclusive": stop,
        "prompt_template": "Question: {question}\\nLet's think step by step.\\nAnswer:",
        "local_prior_benchmark_files": [str(path) for path in prior_files],
        "local_prior_id_or_prompt_hash_overlap": 0,
        "cohort_path": str(output), "cohort_sha256": sha256(payload.encode("utf-8")),
        "selected": [{"id": row["id"], "prompt_sha256": sha256(row["prompt"].encode("utf-8"))}
                     for row in records],
        "provenance_limit": "No overlap with local benchmark JSONL files. Public dataset pretraining exposure and external prior use are not ruled out.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    manifest.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frozen {args.count} prompts: {output}; manifest: {manifest}")


if __name__ == "__main__":
    main()
