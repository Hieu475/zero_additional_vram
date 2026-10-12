#!/usr/bin/env python3
"""Run a paired natural-document context pilot against a frozen roofline.

Example:
  python scripts/run_natural_context_validation.py \
    --output-dir experiments/natural_context_validation/pilot_rtx4050
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_natural_context_inputs import prepare  # noqa: E402
from run_heldout_roofline_validation import (  # noqa: E402
    ACTIONS, K_VALUES, append_jsonl, gpu_snapshot, measure_vanilla, package_version,
    run_action,
)
from zassd.models.layer_manager import LayerManager  # noqa: E402
from zassd.models.model_adapter import ModelAdapter  # noqa: E402
from zassd.profiling.action_cost_model import MeasuredActionCostModel  # noqa: E402


def summarize(raw_rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw_rows:
        pairs.setdefault((row["prompt_id"], row["context_bucket"]), {})[row["method_key"]] = row
    actions = sorted({key for pair in pairs.values() for key in pair if key != "vanilla"})
    rng = random.Random(seed)
    report: dict[str, Any] = {"n_prompts": len(pairs), "actions": {}}
    for action in actions:
        speedups, token_mae_parts, tps_abs_errors, tps_pct_errors = [], [], [], []
        draft_ms, verify_ms, cache_ms = [], [], []
        measured_profitable, predicted_profitable = [], []
        for pair in pairs.values():
            if "vanilla" not in pair or action not in pair:
                continue
            speedups.append(pair[action]["measured_speedup"])
            measured_profitable.append(pair[action]["measured_speedup"] > 1.0)
            predicted_profitable.append(pair[action]["prediction"]["speedup"] > 1.0)
            token_mae_parts.append(abs(
                pair[action]["measured"]["generated_tokens_per_cycle"]
                - pair[action]["prediction"]["tokens_per_cycle"]
            ))
            measured_tps = pair[action]["measured"]["decode_tps"]
            predicted_tps = pair[action]["prediction"]["decode_tps"]
            tps_abs_errors.append(abs(measured_tps - predicted_tps))
            tps_pct_errors.append(abs(measured_tps - predicted_tps) / max(measured_tps, 1e-9))
            draft_ms.append(pair[action]["measured"]["draft_ms_per_cycle"])
            verify_ms.append(pair[action]["measured"]["verify_ms_per_cycle"])
            cache_ms.append(pair[action]["measured"]["cache_ms_per_cycle"])
            vanilla_ids = pair["vanilla"]["measured"].get("token_ids", [])
            action_ids = pair[action]["measured"].get("token_ids", [])
            first_divergence = next(
                (i for i, (a, b) in enumerate(zip(vanilla_ids, action_ids)) if a != b),
                min(len(vanilla_ids), len(action_ids)),
            )
            pair[action]["sequence_exact_match"] = vanilla_ids == action_ids
            pair[action]["first_divergence_position"] = (
                None if vanilla_ids == action_ids else first_divergence
            )
        if not speedups:
            continue
        boot = []
        for _ in range(5000):
            sample = [speedups[rng.randrange(len(speedups))] for _ in speedups]
            boot.append(sum(sample) / len(sample))
        boot.sort()
        tp = sum(p and m for p, m in zip(predicted_profitable, measured_profitable))
        fp = sum(p and not m for p, m in zip(predicted_profitable, measured_profitable))
        fn = sum(not p and m for p, m in zip(predicted_profitable, measured_profitable))
        tn = sum(not p and not m for p, m in zip(predicted_profitable, measured_profitable))
        report["actions"][action] = {
            "n_paired_prompts": len(speedups),
            "mean_speedup": sum(speedups) / len(speedups),
            "median_speedup": sorted(speedups)[len(speedups) // 2],
            "bootstrap_prompt_mean_95ci": [boot[int(.025 * len(boot))], boot[int(.975 * len(boot))]],
            "measured_profitable_count": sum(measured_profitable),
            "predicted_profitable_count": sum(predicted_profitable),
            "feasibility_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                      "profitable_recall": tp / max(1, tp + fn),
                                      "false_positive_rate": fp / max(1, fp + tn)},
            "tokens_per_cycle_mae": sum(token_mae_parts) / len(token_mae_parts),
            "decode_tps_mae": sum(tps_abs_errors) / len(tps_abs_errors),
            "decode_tps_mape": sum(tps_pct_errors) / len(tps_pct_errors),
            "measured_stage_ms_per_cycle_mean": {
                "draft": sum(draft_ms) / len(draft_ms),
                "verify": sum(verify_ms) / len(verify_ms),
                "cache": sum(cache_ms) / len(cache_ms),
            },
            "sequence_exact_match_count": sum(
                bool(p[action]["sequence_exact_match"]) for p in pairs.values()
                if action in p and "vanilla" in p
            ),
            "sequence_exact_match_rate": sum(
                bool(p[action]["sequence_exact_match"]) for p in pairs.values()
                if action in p and "vanilla" in p
            ) / len(speedups),
            "first_divergence_positions": [
                p[action]["first_divergence_position"] for p in pairs.values()
                if action in p and "vanilla" in p and not p[action]["sequence_exact_match"]
            ],
        }
    by_bucket: dict[str, Any] = {}
    for bucket in sorted({key[1] for key in pairs}):
        bucket_rows = [pair for (prompt_id, b), pair in pairs.items() if b == bucket]
        by_bucket[bucket] = {}
        for action in actions:
            ratios = [p[action]["measured_speedup"] for p in bucket_rows if action in p]
            if ratios:
                by_bucket[bucket][action] = {
                    "n": len(ratios), "mean_speedup": sum(ratios) / len(ratios),
                    "median_speedup": sorted(ratios)[len(ratios) // 2],
                    "sequence_exact_match_rate": sum(
                        bool(p[action]["sequence_exact_match"]) for p in bucket_rows if action in p
                    ) / len(ratios),
                }
    report["by_context_bucket"] = by_bucket
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-bucket", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA device is required")
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True)

    inputs_path = out / "prepared_inputs.jsonl"
    source_manifest = prepare(inputs_path, args.model, args.revision, args.per_bucket,
                              (384, 768, 1536, 3072), args.seed)
    records = [json.loads(line) for line in inputs_path.read_text().splitlines() if line.strip()]
    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quantization,
        device_map=args.device, attn_implementation="sdpa").eval()
    manager = LayerManager(ModelAdapter(model))
    # Frozen baseline and action coefficients from the prior independent roofline run.
    frozen_baseline_tps = 41.75674834837685
    cost_model = MeasuredActionCostModel.for_model("qwen25_3b", baseline_tps=frozen_baseline_tps)
    config = {
        "model": args.model, "model_revision": args.revision, "quantization": "NF4",
        "compute_dtype": "float16", "attention_implementation": "sdpa",
        "max_new_tokens": args.max_new_tokens, "per_context_bucket": args.per_bucket,
        "context_target_tokens": [384, 768, 1536, 3072], "seed": args.seed,
        "execution_commit": base_sha, "dataset_manifest": source_manifest,
        "roofline": "Frozen action coefficients and baseline from run_rtx4050_frozen_b2bfb01/calibration/baseline.json",
        "fit_policy": "No fitting or threshold selection on natural-context pilot",
        "actions": [{"config": a["config"], "kept_layers": a["kept_layers"], "k": list(K_VALUES)} for a in ACTIONS],
        "measurement": "one paired observation per prompt and action; within-prompt order randomized",
        "limitations": "exploratory pilot; prompt bootstrap does not capture between-session thermal variance",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    environment = {
        "commit_sha": base_sha, "python": platform.python_version(), "torch": torch.__version__,
        "transformers": package_version("transformers"), "bitsandbytes": package_version("bitsandbytes"),
        "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(args.device),
        "initial_telemetry": gpu_snapshot(args.device),
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2))

    # Warm each path, then measure paired methods in a seeded random order.
    warm = torch.tensor([records[0]["input_ids"]], dtype=torch.long, device=args.device)
    measure_vanilla(model, tokenizer, warm, min(8, args.max_new_tokens), args.device)
    for action in ACTIONS:
        run_action(model, tokenizer, manager, warm, action, 2,
                   min(8, args.max_new_tokens), args.device)
    rng = random.Random(args.seed)
    raw_path = out / "raw_measurements.jsonl"
    raw_rows: list[dict[str, Any]] = []
    for record in records:
        prompt = torch.tensor([record["input_ids"]], dtype=torch.long, device=args.device)
        jobs: list[tuple[str, Any, int | None]] = [("vanilla", None, None)]
        jobs.extend(("action", action, k) for action in ACTIONS for k in K_VALUES)
        rng.shuffle(jobs)
        vanilla_tps: float | None = None
        pending: list[dict[str, Any]] = []
        for kind, action, k in jobs:
            before = gpu_snapshot(args.device)
            if kind == "vanilla":
                measured = measure_vanilla(model, tokenizer, prompt, args.max_new_tokens, args.device)
                vanilla_tps = measured["decode_tps"]
                row = {"prompt_id": record["prompt_id"], "source_task": record["source_task"],
                       "context_bucket": record["context_bucket"], "prompt_tokens": record["prompt_tokens"],
                       "method_key": "vanilla", "method": "vanilla", "measured": measured,
                       "telemetry_before": before, "telemetry_after": gpu_snapshot(args.device)}
            else:
                prediction = cost_model.evaluate_action(action["config"], int(k))
                measured = run_action(model, tokenizer, manager, prompt, action, int(k),
                                      args.max_new_tokens, args.device)
                cycles = measured.pop("acceptance_cycles")
                row = {"prompt_id": record["prompt_id"], "source_task": record["source_task"],
                       "context_bucket": record["context_bucket"], "prompt_tokens": record["prompt_tokens"],
                       "method_key": f"{action['config']}_K{k}", "method": "layer_skip",
                       "action": action["config"], "k": k,
                       "prediction": {"draft_ms_per_cycle": prediction.draft_ms,
                                      "verify_ms_per_cycle": prediction.verify_ms,
                                      "tokens_per_cycle": prediction.expected_tokens_per_step,
                                      "decode_tps": prediction.expected_tps,
                                      "speedup": prediction.expected_speedup},
                       "measured": measured, "acceptance_cycles": cycles,
                       "telemetry_before": before, "telemetry_after": gpu_snapshot(args.device)}
                pending.append(row)
            if kind == "vanilla":
                append_jsonl(raw_path, row)
                raw_rows.append(row)
        if vanilla_tps is None:
            raise RuntimeError(f"Vanilla observation missing for {record['prompt_id']}")
        # Append paired speedups after all methods for this prompt are measured.
        for row in pending:
            row["measured_speedup"] = row["measured"]["decode_tps"] / max(vanilla_tps, 1e-9)
            append_jsonl(raw_path, row)
            raw_rows.append(row)
        # Vanilla row appears once in raw; update in-memory row for pairing only.
        for row in raw_rows:
            if row["prompt_id"] == record["prompt_id"] and row["method_key"] == "vanilla":
                row["measured_speedup"] = 1.0
    summary = summarize(raw_rows, args.seed)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"DONE: {out}")


if __name__ == "__main__":
    main()
