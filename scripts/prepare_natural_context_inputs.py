#!/usr/bin/env python3
"""Prepare reproducible natural-document contexts from LongBench.

The source archive is pinned and hash-checked. Only token IDs and source IDs
are written to the experiment directory; source documents are not copied into
the repository. Contexts are natural document prefixes, never repeated text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


SOURCE_REPO = "THUDM/LongBench"
SOURCE_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
SOURCE_ARCHIVE = "data.zip"
SOURCE_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
TASKS = ("narrativeqa", "qasper", "gov_report", "multi_news", "qmsum")
DEFAULT_TARGETS = (384, 768, 1536, 3072)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(output: Path, model: str, model_revision: str, per_bucket: int,
            targets: tuple[int, ...], seed: int,
            exclude_prompt_files: tuple[Path, ...] = ()) -> dict[str, Any]:
    archive = Path(hf_hub_download(
        repo_id=SOURCE_REPO, filename=SOURCE_ARCHIVE,
        repo_type="dataset", revision=SOURCE_REVISION,
    ))
    actual_hash = sha256_file(archive)
    if actual_hash != SOURCE_SHA256:
        raise RuntimeError(f"LongBench archive hash mismatch: {actual_hash}")

    tokenizer = AutoTokenizer.from_pretrained(model, revision=model_revision)
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zf:
        for task in TASKS:
            member = f"data/{task}.jsonl"
            with zf.open(member) as stream:
                for index, raw in enumerate(stream):
                    row = json.loads(raw)
                    context = row.get("context", "")
                    task_text = row.get("input", "").strip()
                    if not task_text:
                        task_text = "Summarize the document."
                    header = "Read the following document and complete the task.\n\nDocument:\n"
                    suffix = f"\n\nTask:\n{task_text}"
                    header_ids = tokenizer(header, add_special_tokens=True)["input_ids"]
                    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
                    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
                    candidates.append({
                        "task": task, "index": index, "source_id": str(row.get("_id", index)),
                        "header_ids": header_ids, "context_ids": context_ids,
                        "suffix_ids": suffix_ids,
                    })

    rng.shuffle(candidates)
    excluded: set[tuple[str, str]] = set()
    for path in exclude_prompt_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("source_task") is not None and record.get("source_id") is not None:
                excluded.add((str(record["source_task"]), str(record["source_id"])))
    used: set[tuple[str, str]] = set(excluded)
    selected: list[dict[str, Any]] = []
    for target in targets:
        bucket = {
            384: "128-512", 768: "512-1024",
            1536: "1024-2048", 3072: "2048-4096",
        }.get(target, str(target))
        bucket_rows = 0
        for row in candidates:
            key = (row["task"], row["source_id"])
            if key in used:
                continue
            context_budget = target - len(row["header_ids"]) - len(row["suffix_ids"])
            if context_budget <= 0 or len(row["context_ids"]) < context_budget:
                continue
            ids = row["header_ids"] + row["context_ids"][:context_budget] + row["suffix_ids"]
            if len(ids) != target:
                raise AssertionError("Prepared input does not match requested token length")
            selected.append({
                "prompt_id": f"{row['task']}_{row['index']}_ctx{target}",
                "source_task": row["task"], "source_id": row["source_id"],
                "source_row": row["index"], "context_bucket": bucket,
                "prompt_tokens": target, "input_ids": ids,
            })
            used.add(key)
            bucket_rows += 1
            if bucket_rows >= per_bucket:
                break
        if bucket_rows < per_bucket:
            raise RuntimeError(
                f"Only found {bucket_rows}/{per_bucket} suitable examples at {target} tokens"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "source": "THUDM/LongBench",
        "source_url": "https://github.com/THUDM/LongBench",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_archive": SOURCE_ARCHIVE,
        "source_archive_sha256": actual_hash,
        "tokenizer_model": model,
        "tokenizer_revision": model_revision,
        "seed": seed,
        "targets_tokens": list(targets),
        "examples_per_bucket": per_bucket,
        "excluded_source_ids": len(excluded),
        "excluded_input_files": [str(path) for path in exclude_prompt_files],
        "selection": "deterministic shuffled source rows; unique source IDs; natural context prefix; no repetition",
        "prompt_format": "fixed header + document prefix + task/question suffix",
        "stored_prompt_content": "token IDs only; source text not redistributed",
        "prepared_sha256": sha256_file(output),
        "n_examples": len(selected),
        "bucket_counts": {str(target): sum(r["prompt_tokens"] == target for r in selected) for target in targets},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model-revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--per-bucket", type=int, default=3)
    parser.add_argument("--targets", type=int, nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exclude-prompt-files", type=Path, nargs="*", default=())
    args = parser.parse_args()
    manifest = prepare(args.output, args.model, args.model_revision, args.per_bucket,
                       tuple(args.targets), args.seed, tuple(args.exclude_prompt_files))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
