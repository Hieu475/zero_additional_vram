#!/usr/bin/env python3
"""Build development-only long-context prompts by concatenating CNN/DM dev articles to target token lengths."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
INSTRUCTION = "\n\nSummarize the key points of the above articles in 2-3 sentences:\nSummary:"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/benchmarks/cnndm_ext150.jsonl")
    ap.add_argument("--lengths", nargs="+", type=int, default=[512, 1024, 2048, 3072])
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    articles = []
    for line in Path(args.source).read_text().splitlines():
        if line.strip():
            text = json.loads(line)["prompt"]
            body = re.sub(r"\n\nSummarize the key points.*$", "", text, flags=re.S)
            articles.append(body.removeprefix("Article:\n"))
    inst_len = len(tok(INSTRUCTION)["input_ids"])
    rows, cursor = [], 0
    for length in args.lengths:
        for variant in range(args.variants):
            parts, ids = [], []
            while len(ids) < length - inst_len:
                parts.append(articles[cursor % len(articles)])
                cursor += 1
                ids = tok("Article:\n" + "\n\n".join(parts))["input_ids"]
            text = tok.decode(ids[:length - inst_len]) + INSTRUCTION
            total = len(tok(text)["input_ids"])
            rows.append({"id": f"long_{length:04d}_{'abc'[variant]}", "benchmark": "cnndm_long_concat_dev", "prompt": text,
                         "target_tokens": length, "actual_tokens": total})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print([(r["id"], r["actual_tokens"]) for r in rows])


if __name__ == "__main__":
    main()
