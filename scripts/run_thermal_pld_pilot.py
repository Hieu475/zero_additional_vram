#!/usr/bin/env python3
"""Thermal telemetry pilot for paired vanilla/PLD short-context decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_heldout_roofline_validation import gpu_snapshot, measure_vanilla, package_version, append_jsonl  # noqa: E402
from run_independent_pld_short_context import measure_pld  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_prompts(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(record["source_task"], []).append(record)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        progressed = False
        for task in sorted(buckets):
            if buckets[task] and len(chosen) < count:
                chosen.append(buckets[task].pop(rng.randrange(len(buckets[task]))))
                progressed = True
        if not progressed:
            raise ValueError(f"Only {len(chosen)} unique prompts available")
    return chosen


def summarize(rows: list[dict[str, Any]], draws: int = 10000) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(row["prompt_id"], {})[row["method_key"]] = row
    rng = random.Random(4319)
    output: dict[str, Any] = {}
    for method in ("pld_forced_miss_k4", "pld_k4", "pld_k8"):
        ratios, exact = [], []
        for pair in pairs.values():
            if "vanilla" not in pair or method not in pair:
                continue
            ratios.append(pair[method]["measured"]["decode_tps"] /
                          max(pair["vanilla"]["measured"]["decode_tps"], 1e-9))
            exact.append(pair[method]["measured"]["token_ids"] ==
                         pair["vanilla"]["measured"]["token_ids"])
        if ratios:
            sampled = sorted(sum(ratios[rng.randrange(len(ratios))] for _ in ratios) / len(ratios)
                             for _ in range(draws))
            output[method] = {
                "n_prompts": len(ratios), "mean_decode_speedup": sum(ratios) / len(ratios),
                "prompt_bootstrap_95ci": [sampled[int(.025 * draws)], sampled[int(.975 * draws)]],
                "median_speedup": sorted(ratios)[len(ratios) // 2],
                "exact_match_count": sum(exact), "exact_match_rate": sum(exact) / len(exact),
            }
    by_temp: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        temp = row["telemetry_before"].get("temperature_c")
        if temp is not None:
            label = "<70C" if temp < 70 else ("70-79C" if temp < 80 else ">=80C")
            by_temp.setdefault(label, []).append(row)
    thermal = {}
    for label, group in by_temp.items():
        thermal[label] = {
            "n_method_runs": len(group),
            "temperature_mean_c": sum(r["telemetry_before"]["temperature_c"] for r in group) / len(group),
            "sm_clock_mean_mhz": sum(r["telemetry_before"].get("sm_clock_mhz") or 0 for r in group) / len(group),
            "methods": {m: sum(r["measured"]["decode_ms"] for r in group if r["method_key"] == m) /
                        max(1, sum(r["method_key"] == m for r in group))
                        for m in sorted({r["method_key"] for r in group})},
        }
    return {"n_prompts": len(pairs), "paired_speedup": output, "thermal_bins_diagnostic": thermal}


class TelemetrySampler:
    def __init__(self, device: str, path: Path, interval_s: float):
        self.device, self.path, self.interval_s = device, path, interval_s
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        def sample() -> None:
            while not self.stop_event.is_set():
                append_jsonl(self.path, gpu_snapshot(self.device))
                self.stop_event.wait(self.interval_s)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-inputs", type=Path, default=ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--order-seed", type=int, default=20261014)
    parser.add_argument("--telemetry-interval", type=float, default=1.0)
    parser.add_argument("--preheat-to-c", type=float, default=None,
                        help="Warm the GPU with vanilla decoding until this temperature, capped at 240 seconds")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out = out if out.is_absolute() else ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source = args.prepared_inputs if args.prepared_inputs.is_absolute() else ROOT / args.prepared_inputs
    all_records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    records = choose_prompts(all_records, args.prompts, args.seed)
    (out / "prompt_subset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    config = {"commit_sha": commit, "model": args.model, "model_revision": args.revision,
              "precision": "NF4 / FP16 compute", "attention": "SDPA", "kv_backend": "static",
              "prompt_source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
              "n_prompts": len(records), "prompt_ids": [r["prompt_id"] for r in records],
              "max_new_tokens": args.max_new_tokens,
              "methods": ["vanilla", "pld_forced_miss_k4", "pld_k4", "pld_k8"],
              "order_seed": args.order_seed, "telemetry_interval_s": args.telemetry_interval,
              "power_or_clock_intervention": False,
              "design": "paired prompt measurements, randomized method order within prompt; temperature bins diagnostic only"}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        quantization_config=quant, device_map=args.device, attn_implementation="sdpa").eval()
    env = {"commit_sha": commit, "python": platform.python_version(), "torch": torch.__version__,
           "transformers": package_version("transformers"), "bitsandbytes": package_version("bitsandbytes"),
           "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(args.device),
           "initial_telemetry": gpu_snapshot(args.device)}
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    sample_path = out / "telemetry_samples.jsonl"
    sampler = TelemetrySampler(args.device, sample_path, args.telemetry_interval)
    sampler.start()
    try:
        warm = torch.tensor([records[0]["input_ids"]], dtype=torch.long, device=args.device)
        measure_vanilla(model, tokenizer, warm, min(8, args.max_new_tokens), args.device)
        measure_pld(model, tokenizer, warm, 4, min(8, args.max_new_tokens), args.device, force_miss=True)
        measure_pld(model, tokenizer, warm, 4, min(8, args.max_new_tokens), args.device)
        measure_pld(model, tokenizer, warm, 8, min(8, args.max_new_tokens), args.device)
        preheat = {"target_c": args.preheat_to_c, "start": gpu_snapshot(args.device),
                   "elapsed_s": 0.0, "reached_target": None, "capped": False}
        if args.preheat_to_c is not None:
            started = time.perf_counter()
            while time.perf_counter() - started < 240:
                snap = gpu_snapshot(args.device)
                if snap.get("temperature_c") is not None and snap["temperature_c"] >= args.preheat_to_c:
                    preheat["reached_target"] = True
                    break
                measure_vanilla(model, tokenizer, warm, args.max_new_tokens, args.device)
            preheat["elapsed_s"] = time.perf_counter() - started
            preheat["end"] = gpu_snapshot(args.device)
            preheat["capped"] = preheat["reached_target"] is not True
            (out / "preheat.json").write_text(json.dumps(preheat, indent=2) + "\n")
            print(f"preheat: {json.dumps(preheat)}", flush=True)
        rng = random.Random(args.order_seed)
        raw, rows = out / "raw_measurements.jsonl", []
        methods = ["vanilla", "pld_forced_miss_k4", "pld_k4", "pld_k8"]
        for record in records:
            prompt = torch.tensor([record["input_ids"]], dtype=torch.long, device=args.device)
            order = methods.copy()
            rng.shuffle(order)
            for method in order:
                before = gpu_snapshot(args.device)
                if method == "vanilla":
                    measured = measure_vanilla(model, tokenizer, prompt, args.max_new_tokens, args.device)
                    measured["total_ms"] = measured["prefill_ms"] + measured["decode_ms"]
                else:
                    k = 8 if method == "pld_k8" else 4
                    measured = measure_pld(model, tokenizer, prompt, k, args.max_new_tokens,
                                           args.device, force_miss=method == "pld_forced_miss_k4")
                row = {"prompt_id": record["prompt_id"], "source_task": record["source_task"],
                       "source_id": record["source_id"], "prompt_tokens": record["prompt_tokens"],
                       "method_key": method, "measured": measured, "telemetry_before": before,
                       "telemetry_after": gpu_snapshot(args.device), "order_index": order.index(method),
                       "order_seed": args.order_seed}
                append_jsonl(raw, row)
                rows.append(row)
            print(f"completed {record['prompt_id']} at {gpu_snapshot(args.device).get('temperature_c')} C", flush=True)
        summary = summarize(rows)
        summary["telemetry_sample_count"] = sum(1 for _ in sample_path.open())
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        sampler.close()
    print(f"DONE: {out}")


if __name__ == "__main__":
    main()
