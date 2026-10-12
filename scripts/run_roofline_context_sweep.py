#!/usr/bin/env python3
"""Controlled held-out context-length sweep for the frozen roofline model.

Calibration uses GSM8K + HumanEval vanilla runs. The test workload is CNN/DM.
Context lengths above each source prompt are formed by repeating its token IDs;
those rows isolate context-shape effects and are synthetic, not natural documents.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_heldout_roofline_validation import (  # noqa: E402
    ACTIONS, BENCHMARKS, K_VALUES, append_jsonl, context_token_ids,
    gpu_snapshot, load_records, measure_vanilla, run_action, sha256,
)
from zassd.models.layer_manager import LayerManager  # noqa: E402
from zassd.models.model_adapter import ModelAdapter  # noqa: E402
from zassd.profiling.action_cost_model import MeasuredActionCostModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples-per-task", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    if args.samples_per_task < 1 or args.max_new_tokens < 2:
        ap.error("samples-per-task must be positive and max-new-tokens must be at least 2")
    if not args.context_lengths or any(n < 1 for n in args.context_lengths):
        ap.error("context-lengths must contain positive values")
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA device is required")
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    (out / "heldout").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    calibration = []
    for task in ("gsm8k", "humaneval"):
        calibration.extend({"task": task, **row}
                           for row in load_records(BENCHMARKS[task], args.samples_per_task))
    heldout = [{"task": "cnndm", **row}
               for row in load_records(BENCHMARKS["cnndm"], args.samples_per_task)]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=bnb,
        device_map=args.device, attn_implementation="sdpa").eval()
    manager = LayerManager(ModelAdapter(model))
    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    cfg = {
        "model": args.model, "model_revision": args.revision,
        "quantization": "NF4", "compute_dtype": "float16", "attention": "SDPA",
        "max_new_tokens": args.max_new_tokens, "samples_per_task": args.samples_per_task,
        "calibration_tasks": ["gsm8k", "humaneval"], "heldout_task": "cnndm",
        "context_lengths": args.context_lengths,
        "context_construction": "token-exact truncation/repetition of source prompt IDs",
        "actions": [{"config": x["config"], "kept_layers": x["kept_layers"], "k": list(K_VALUES)} for x in ACTIONS],
        "predictor": "Frozen MeasuredActionCostModel action parameters; only vanilla decode baseline recalibrated per context using calibration prompts",
        "seed": args.seed, "execution_commit": base_sha,
        "dataset_sha256": {name: sha256(path) for name, path in BENCHMARKS.items()},
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "environment.json").write_text(json.dumps({
        "commit_sha": base_sha, "gpu_name": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "initial_telemetry": gpu_snapshot(args.device),
    }, indent=2))

    rng = random.Random(args.seed)
    cal_path = out / "calibration/raw_measurements.jsonl"
    held_path = out / "heldout/raw_measurements.jsonl"
    warm_prompt = calibration[0]["prompt"]
    for length in args.context_lengths:
        warm_ids = context_token_ids(tokenizer, warm_prompt, length)
        measure_vanilla(model, tokenizer, warm_ids, min(8, args.max_new_tokens), args.device)
        for action in ACTIONS:
            run_action(model, tokenizer, manager, warm_ids, action, 2,
                       min(8, args.max_new_tokens), args.device)
    torch.cuda.synchronize(args.device)
    baseline_by_context = {}
    for length in args.context_lengths:
        rows = []
        for record in calibration:
            ids = context_token_ids(tokenizer, record["prompt"], length)
            before = gpu_snapshot(args.device)
            measured = measure_vanilla(model, tokenizer, ids, args.max_new_tokens, args.device)
            after = gpu_snapshot(args.device)
            row = {"split": "calibration", "task": record["task"],
                   "prompt_id": str(record["id"]), "prompt_tokens": int(ids.shape[-1]),
                   "context_length": length, "method": "vanilla", **measured,
                   "telemetry_before": before, "telemetry_after": after}
            append_jsonl(cal_path, row)
            rows.append(row)
        baseline_by_context[str(length)] = (
            sum(r["decode_tokens"] for r in rows) /
            max(sum(r["decode_ms"] for r in rows) / 1000.0, 1e-9)
        )
    (out / "calibration/baseline.json").write_text(json.dumps({
        "baseline_tps_decode_by_context": baseline_by_context,
        "n_prompts_per_context": len(calibration),
        "fit_policy": "baseline only; draft, verify, acceptance and cache cost coefficients frozen",
        "prompt_ids": [{"task": r["task"], "id": str(r["id"])} for r in calibration],
    }, indent=2))

    for record in heldout:
        for length in args.context_lengths:
            ids = context_token_ids(tokenizer, record["prompt"], length)
            cm = MeasuredActionCostModel.for_model(
                "qwen25_3b", baseline_tps=baseline_by_context[str(length)])
            jobs = [("action", action, k) for action in ACTIONS for k in K_VALUES]
            rng.shuffle(jobs)
            jobs.insert(rng.randrange(len(jobs) + 1), ("vanilla", None, None))
            vanilla = None
            action_rows = []
            for kind, action, k in jobs:
                if kind == "vanilla":
                    before = gpu_snapshot(args.device)
                    vanilla = measure_vanilla(model, tokenizer, ids, args.max_new_tokens, args.device)
                    after = gpu_snapshot(args.device)
                    append_jsonl(held_path, {
                        "split": "heldout", "task": record["task"],
                        "prompt_id": str(record["id"]), "context_length": length,
                        "prompt_tokens": int(ids.shape[-1]), "method": "vanilla", **vanilla,
                        "telemetry_before": before, "telemetry_after": after,
                    })
                    continue
                pred = cm.evaluate_action(action["config"], k)
                before = gpu_snapshot(args.device)
                actual = run_action(model, tokenizer, manager, ids, action, k,
                                    args.max_new_tokens, args.device)
                after = gpu_snapshot(args.device)
                action_rows.append({
                    "split": "heldout", "task": record["task"],
                    "prompt_id": str(record["id"]), "context_length": length,
                    "prompt_tokens": int(ids.shape[-1]), "method": "layer_skip",
                    "action": action["config"], "kept_layers": action["kept_layers"], "k": k,
                    "calibrated_baseline_tps": baseline_by_context[str(length)],
                    "prediction": {
                        "draft_ms_per_cycle": pred.draft_ms,
                        "verify_ms_per_cycle": pred.verify_ms,
                        "cache_ms_per_cycle": cm.default_cache_ms,
                        "total_cycle_ms": pred.total_cycle_ms,
                        "acceptance_rate": pred.expected_acceptance,
                        "tokens_per_cycle": pred.expected_tokens_per_step,
                        "decode_tps": pred.expected_tps,
                        "speedup_vs_calibration_vanilla": pred.expected_speedup,
                    },
                    "measured": actual, "measured_speedup_vs_prompt_vanilla": None,
                    "telemetry_before": before, "telemetry_after": after,
                })
                print(f"heldout {record['id']} ctx={length} {action['config']} K={k}: "
                      f"pred {pred.expected_tps:.1f}, measured {actual['decode_tps']:.1f} tok/s",
                      flush=True)
            if vanilla is None:
                raise RuntimeError(f"No vanilla control for {record['id']} context {length}")
            for row in action_rows:
                row["measured_speedup_vs_prompt_vanilla"] = (
                    row["measured"]["decode_tps"] / max(vanilla["decode_tps"], 1e-9))
                append_jsonl(held_path, row)
    print(f"DONE: {out}")


if __name__ == "__main__":
    main()
