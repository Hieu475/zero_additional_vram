#!/usr/bin/env python3
"""Prepare and cache standardized evaluation subsets: GSM8K, HumanEval, and CNN/DailyMail.

Creates:
- data/benchmarks/gsm8k_eval.jsonl (50 items)
- data/benchmarks/humaneval_eval.jsonl (50 items)
- data/benchmarks/cnndm_eval.jsonl (50 items)
"""

from __future__ import annotations

import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data/benchmarks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_gsm8k(n_samples: int = 50):
    print(f"Preparing GSM8K ({n_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n_samples}]")
    out_file = OUTPUT_DIR / "gsm8k_eval.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            prompt = f"Question: {item['question']}\nLet's think step by step.\nAnswer:"
            record = {
                "id": f"gsm8k_{idx:03d}",
                "benchmark": "gsm8k",
                "prompt": prompt,
                "gold_answer": item["answer"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved to {out_file}")


def prepare_humaneval(n_samples: int = 50):
    print(f"Preparing HumanEval ({n_samples} samples)...")
    ds = load_dataset("openai/openai_humaneval", split=f"test[:{n_samples}]")
    out_file = OUTPUT_DIR / "humaneval_eval.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            record = {
                "id": item["task_id"],
                "benchmark": "humaneval",
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved to {out_file}")


def prepare_cnndm(n_samples: int = 50):
    print(f"Preparing CNN/DailyMail ({n_samples} samples)...")
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{n_samples}]")
    out_file = OUTPUT_DIR / "cnndm_eval.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            # Truncate article to first ~300 words to fit context comfortably
            article = " ".join(item["article"].split()[:300])
            prompt = f"Article:\n{article}\n\nSummarize the key points of the above article in 2-3 sentences:\nSummary:"
            record = {
                "id": f"cnndm_{idx:03d}",
                "benchmark": "cnndm",
                "prompt": prompt,
                "gold_summary": item["highlights"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved to {out_file}")


if __name__ == "__main__":
    prepare_gsm8k(50)
    prepare_humaneval(50)
    prepare_cnndm(50)
    print("All standard benchmarks prepared successfully!")
