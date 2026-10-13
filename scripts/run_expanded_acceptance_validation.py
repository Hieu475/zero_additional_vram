#!/usr/bin/env python3
"""Collect leakage-controlled acceptance traces across prompt-level splits.

The checked-in benchmark snapshots contain only 50 prompts per task. The
protocol therefore uses 60 train, 20 development and 20 in-domain held-out
prompts across GSM8K/HumanEval, plus all 50 CNN/DM prompts as cross-task holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_heldout_roofline_validation import (  # noqa: E402
    ACTIONS, K_VALUES, BENCHMARKS, append_jsonl, gpu_snapshot, measure_vanilla,
    package_version, run_action, sha256,
)
from zassd.models.layer_manager import LayerManager  # noqa: E402
from zassd.models.model_adapter import ModelAdapter  # noqa: E402
from zassd.profiling.action_cost_model import MeasuredActionCostModel  # noqa: E402

CALIBRATION_TASKS = ("gsm8k", "humaneval")
SPLITS = ("train", "development", "heldout_in_domain", "heldout_cross_task")


def prompt_split(task: str, prompt_id: str) -> str:
    if task == "cnndm":
        return "heldout_cross_task"
    digest = int(hashlib.sha256(f"{task}:{prompt_id}".encode()).hexdigest()[:8], 16)
    bucket = digest % 5
    return {0: "development", 1: "heldout_in_domain"}.get(bucket, "train")


def read_task(task: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in BENCHMARKS[task].read_text().splitlines() if line.strip()]


def cycle_row(split: str, task: str, prompt_id: str, action: dict[str, Any], configured_k: int,
              cycle_index: int, cycle: dict[str, Any], model_id: str) -> dict[str, Any]:
    attempted = int(cycle.get("k_attempted", cycle.get("k", 0)))
    accepted = int(cycle.get("accepted_prefix_length", cycle.get("accepted", 0)))
    return {
        "split": split, "task": task, "prompt_id": prompt_id, "model": model_id,
        "cycle_index": cycle_index, "context_length": int(cycle.get("cycle_context_length", 0)),
        "action": action["config"], "k": configured_k,
        "prediction_horizon": min(configured_k, max(0, int(cycle.get("remaining_token_budget", configured_k + 1)) - 1)),
        "k_configured": int(cycle.get("k_configured", configured_k)),
        "k_attempted": attempted, "draft_tokens": attempted,
        "accepted_prefix_length": accepted, "accepted_tokens": accepted,
        "emitted_tokens": int(cycle.get("emitted_count", 0)),
        "emitted_token_ids": cycle.get("emitted_token_ids", []),
        "remaining_token_budget": cycle.get("remaining_token_budget"),
        "eos_encountered": bool(cycle.get("eos_encountered", False)),
        "truncated_by_eos": bool(cycle.get("truncated_by_eos", False)),
        "truncated_by_budget": bool(cycle.get("truncated_by_budget", False)),
        "draft_complete": bool(cycle.get("draft_complete", False)),
        "verify_complete": bool(cycle.get("verify_complete", False)),
        "accepted_survival": [int(accepted >= i) for i in range(1, attempted + 1)],
        "entropy": cycle.get("cycle_entropy"), "logit_margin": cycle.get("cycle_logit_margin"),
        "draft_ms": float(cycle.get("draft_time_ms", 0)),
        "verify_ms": float(cycle.get("verify_time_ms", 0)),
        "cache_ms": float(cycle.get("cache_time_ms", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=4171)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA device is required")
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {out}")
    for split in SPLITS:
        (out / split).mkdir(parents=True, exist_ok=True)
    (out / "acceptance_cycles").mkdir(parents=True, exist_ok=True)

    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for task in (*CALIBRATION_TASKS, "cnndm"):
        for row in read_task(task):
            prompt_id = str(row.get("id", row.get("prompt_id", "")))
            split = prompt_split(task, prompt_id)
            records_by_split[split].append({"task": task, **row, "prompt_id": prompt_id})
    for rows in records_by_split.values():
        rows.sort(key=lambda row: (row["task"], row["prompt_id"]))
    all_keys = [f"{row['task']}:{row['prompt_id']}" for rows in records_by_split.values() for row in rows]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("Prompt ID overlap between splits")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quant,
        device_map=args.device, attn_implementation="sdpa").eval()
    manager = LayerManager(ModelAdapter(model))
    counts = {split: len(rows) for split, rows in records_by_split.items()}
    config = {
        "execution_commit": commit, "model": args.model, "model_revision": args.revision,
        "quantization": "NF4", "compute_dtype": "float16", "attention": "sdpa",
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
        "split_rule": "sha256(task:prompt_id) % 5; buckets 0=dev, 1=in-domain heldout, 2-4=train; CNN/DM cross-task heldout",
        "prompt_counts": counts, "calibration_tasks": list(CALIBRATION_TASKS),
        "cross_task_heldout": ["cnndm"], "actions": [
            {"config": a["config"], "kept_layers": a["kept_layers"], "k": list(K_VALUES)} for a in ACTIONS],
        "dataset_sha256": {task: sha256(path) for task, path in BENCHMARKS.items()},
        "limits": "Repository snapshots cap each task at 50 prompts; calibration train/dev and in-domain heldout are smaller than the proposed 100/40/50 targets.",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = {"commit_sha": commit, "python": platform.python_version(), "torch": torch.__version__,
           "transformers": package_version("transformers"), "bitsandbytes": package_version("bitsandbytes"),
           "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(args.device),
           "initial_telemetry": gpu_snapshot(args.device)}
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    # Warm each execution path once; never mix warmups with evaluation data.
    first = records_by_split["train"][0]["prompt"]
    measure_vanilla(model, tokenizer, first, min(args.max_new_tokens, 8), args.device)
    for action in ACTIONS:
        run_action(model, tokenizer, manager, first, action, 2,
                   min(args.max_new_tokens, 8), args.device)

    rng = random.Random(args.seed)
    cost_model: MeasuredActionCostModel | None = None
    for split in SPLITS:
        raw_path = out / split / "raw_measurements.jsonl"
        cycle_path = out / "acceptance_cycles" / f"{split}.jsonl"
        for record in records_by_split[split]:
            prompt_id, task, prompt = record["prompt_id"], record["task"], record["prompt"]
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            jobs: list[tuple[str, Any, int | None]] = [("vanilla", None, None)]
            jobs.extend(("action", action, k) for action in ACTIONS for k in K_VALUES)
            rng.shuffle(jobs)
            pending: list[dict[str, Any]] = []
            vanilla_result = None
            for method, action, k in jobs:
                before = gpu_snapshot(args.device)
                if method == "vanilla":
                    measured = measure_vanilla(model, tokenizer, prompt, args.max_new_tokens, args.device)
                    vanilla_result = measured
                    row = {"split": split, "task": task, "prompt_id": prompt_id,
                           "method": "vanilla", "prompt_tokens": int(ids.shape[-1]),
                           "measured": measured, "telemetry_before": before,
                           "telemetry_after": gpu_snapshot(args.device)}
                    pending.append(row)
                    continue
                prediction = (cost_model.evaluate_action(action["config"], int(k))
                              if cost_model is not None else None)
                _, metrics = run_action(model, tokenizer, manager, prompt, action, int(k),
                                        args.max_new_tokens, args.device)
                cycles = metrics.pop("acceptance_cycles")
                for cycle_index, cycle in enumerate(cycles):
                    append_jsonl(cycle_path, cycle_row(
                        split, task, prompt_id, action, int(k), cycle_index, cycle, args.model))
                row = {"split": split, "task": task, "prompt_id": prompt_id,
                       "method": "layer_skip", "action": action["config"], "k": int(k),
                       "prompt_tokens": int(ids.shape[-1]), "measured": metrics,
                       "prediction": None if prediction is None else {
                           "total_cycle_ms": prediction.total_cycle_ms,
                           "decode_tps": prediction.expected_tps,
                       }, "telemetry_before": before,
                       "telemetry_after": gpu_snapshot(args.device)}
                pending.append(row)
            if vanilla_result is None:
                raise RuntimeError(f"Vanilla missing for {task}:{prompt_id}")
            for row in pending:
                if row["method"] == "vanilla":
                    row["measured_speedup_vs_prompt_vanilla"] = 1.0
                else:
                    row["measured_speedup_vs_prompt_vanilla"] = (
                        row["measured"]["decode_tps"] / max(vanilla_result["decode_tps"], 1e-9))
                append_jsonl(raw_path, row)
            print(f"[{split}] {task}:{prompt_id} complete", flush=True)
        if split == "train":
            vanilla_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
            decode_tokens = sum(r["measured"]["decode_tokens"] for r in vanilla_rows if r["method"] == "vanilla")
            decode_ms = sum(r["measured"]["decode_ms"] for r in vanilla_rows if r["method"] == "vanilla")
            baseline_tps = decode_tokens * 1000.0 / max(decode_ms, 1e-9)
            cost_model = MeasuredActionCostModel.for_model("qwen25_3b", baseline_tps=baseline_tps)
            (out / "train" / "baseline.json").write_text(json.dumps({
                "baseline_decode_tps": baseline_tps, "n_train_prompts": counts["train"],
                "cost_model_profile": "Frozen action coefficients; baseline calibrated on train only",
            }, indent=2) + "\n")
    print(f"DONE: {out}")


if __name__ == "__main__":
    main()
