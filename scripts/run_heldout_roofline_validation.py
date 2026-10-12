#!/usr/bin/env python3
"""Run a frozen-cost-model validation with a workload-level held-out split.

Calibration prompts come from GSM8K and HumanEval and are used only to measure
the vanilla decode baseline. Existing draft, verify, acceptance, and cache cost
parameters remain frozen. CNN/DailyMail prompts are a separate held-out
workload. Per-prompt raw observations are written incrementally.
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel


BENCHMARKS = {
    "gsm8k": ROOT / "data/benchmarks/gsm8k_eval.jsonl",
    "humaneval": ROOT / "data/benchmarks/humaneval_eval.jsonl",
    "cnndm": ROOT / "data/benchmarks/cnndm_eval.jsonl",
}

# Layer masks are the repository's established CKA configurations. All three
# and K values below use the existing model's parametric fallback (no exact
# action-latency lookup), so the validation exercises its generalization path.
ACTIONS = [
    {"config": "cka_60", "kept_layers": 22,
     "skip_indices": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 21]},
    {"config": "cka_67", "kept_layers": 24,
     "skip_indices": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16]},
    {"config": "cka_90", "kept_layers": 32,
     "skip_indices": [4, 5, 6, 7]},
]
K_VALUES = (2, 4)


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No prompts found in {path}")
    # Evenly sample file order, avoiding a head-only slice when a dataset is large.
    if len(records) > limit:
        indices = [round(i * (len(records) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
        records = [records[i] for i in indices]
    return records


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def context_token_ids(tokenizer, prompt: str, target_tokens: int) -> torch.Tensor:
    """Create token-exact context; extension by repetition is synthetic."""
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if ids.shape[-1] >= target_tokens:
        return ids[:, :target_tokens].contiguous()
    body = ids[:, 1:] if ids.shape[-1] > 1 else ids
    repeats = (target_tokens + body.shape[-1] - 1) // body.shape[-1]
    extended = torch.cat([ids[:, :1], body.repeat(1, repeats)], dim=-1)
    return extended[:, :target_tokens].contiguous()


def gpu_snapshot(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {"gpu_name": None, "sm_clock_mhz": None,
                              "temperature_c": None, "power_w_snapshot": None,
                              "utilization_pct": None, "memory_used_mb": None,
                              "power_limit_w": None, "timestamp_unix_s": time.time()}
    try:
        import pynvml
        pynvml.nvmlInit()
        index = torch.device(device).index or 0
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        result.update({
            "gpu_name": pynvml.nvmlDeviceGetName(h),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "temperature_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "power_w_snapshot": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "utilization_pct": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
            "memory_used_mb": pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024**2),
        })
        try:
            result["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        except Exception:
            pass
    except Exception:
        pass
    return result


def measure_vanilla(model, tokenizer, prompt: str | torch.Tensor, max_new_tokens: int,
                    device: str) -> dict[str, Any]:
    """Vanilla reference using the same static KV path; separates prefill/decode."""
    from zassd.cache.kv_cache import TargetKVCache

    ids = (prompt if isinstance(prompt, torch.Tensor)
           else tokenizer(prompt, return_tensors="pt")["input_ids"]).to(device)
    cache = TargetKVCache(backend="static", max_capacity=max(2048, int(ids.shape[-1]) + max_new_tokens + 8))
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model(ids, past_key_values=cache.cache, use_cache=True)
    torch.cuda.synchronize(device)
    prefill_s = time.perf_counter() - start
    decode_start = time.perf_counter()
    emitted: list[int] = []
    token = out.logits[:, -1:, :].argmax(dim=-1)
    token_id = int(token.item())
    emitted.append(token_id)
    while len(emitted) < max_new_tokens and token_id != tokenizer.eos_token_id:
        with torch.no_grad():
            out = model(token, past_key_values=cache.cache, use_cache=True)
        token = out.logits[:, -1:, :].argmax(dim=-1)
        token_id = int(token.item())
        emitted.append(token_id)
    torch.cuda.synchronize(device)
    decode_s = time.perf_counter() - decode_start
    return {
        "output_tokens": len(emitted),
        "decode_tokens": max(0, len(emitted) - 1),
        "prefill_ms": prefill_s * 1000.0,
        "decode_ms": decode_s * 1000.0,
        "decode_tps": max(0, len(emitted) - 1) / max(decode_s, 1e-9),
        "token_ids": emitted,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_action(model, tokenizer, manager, prompt: str | torch.Tensor, action: dict[str, Any],
               k: int, max_new_tokens: int, device: str) -> dict[str, Any]:
    _, metrics = self_speculative_generate(
        model=model, tokenizer=tokenizer, layer_mgr=manager,
        skip_indices=action["skip_indices"], prompt=prompt, k=k,
        max_new_tokens=max_new_tokens, temperature=0.0, device=device,
        kv_cache_backend="static", config_name=action["config"],
        collect_acceptance_cycles=True,
        kv_cache_max_capacity=max(2048, int(prompt.shape[-1]) + max_new_tokens + max(K_VALUES) + 8)
        if isinstance(prompt, torch.Tensor) else 2048,
    )
    decode_s = max(1e-9, metrics.total_time_s - metrics.prefill_time_s)
    cycles = max(1, metrics.num_verification_cycles)
    emitted = max(1, metrics.total_tokens)
    return {
        "output_tokens": metrics.total_tokens,
        "decode_tokens": max(0, metrics.total_tokens - 1),
        "prefill_ms": metrics.prefill_time_s * 1000.0,
        "decode_ms": decode_s * 1000.0,
        "decode_tps": max(0, metrics.total_tokens - 1) / decode_s,
        "generated_tokens_per_cycle": emitted / cycles,
        "verification_cycles": metrics.num_verification_cycles,
        "draft_ms_per_cycle": metrics.draft_time_s * 1000.0 / cycles,
        "verify_ms_per_cycle": metrics.verify_time_s * 1000.0 / cycles,
        "cache_ms_per_cycle": metrics.cache_time_s * 1000.0 / cycles,
        "other_ms_per_cycle": metrics.other_time_s * 1000.0 / cycles,
        "decode_ms_per_cycle": decode_s * 1000.0 / cycles,
        "acceptance_rate": metrics.acceptance_rate,
        "peak_vram_mb": metrics.peak_vram_mb,
        "token_ids": metrics.generated_token_ids,
        "acceptance_cycles": metrics.per_iteration_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA device is required")

    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    run_tag = time.strftime("%Y%m%d_%H%M%S") + "_" + base_sha[:7]
    out = Path(args.output_dir) if args.output_dir else ROOT / "experiments/roofline_validation" / f"run_{run_tag}"
    out = out if out.is_absolute() else ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    for split in ("calibration", "heldout"):
        (out / split).mkdir(parents=True, exist_ok=True)
    (out / "acceptance_cycles").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    calibration = []
    for task in ("gsm8k", "humaneval"):
        calibration.extend({"task": task, **row} for row in load_records(BENCHMARKS[task], args.samples_per_task))
    heldout = [{"task": "cnndm", **row} for row in load_records(BENCHMARKS["cnndm"], args.samples_per_task)]

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=bnb,
        device_map=args.device, attn_implementation="sdpa").eval()
    manager = LayerManager(ModelAdapter(model))

    config = {
        "model": args.model, "model_revision": args.revision,
        "quantization": "NF4", "compute_dtype": "float16",
        "attention_implementation": "sdpa", "max_new_tokens": args.max_new_tokens,
        "samples_per_task": args.samples_per_task,
        "calibration_tasks": ["gsm8k", "humaneval"],
        "heldout_tasks": ["cnndm"],
        "actions": [{"config": a["config"], "kept_layers": a["kept_layers"], "k": list(K_VALUES)} for a in ACTIONS],
        "predictor_policy": "Frozen MeasuredActionCostModel profile; only baseline_tps is calibrated from calibration vanilla decode throughput. No heldout fitting.",
        "seed": args.seed, "execution_commit": base_sha,
        "dataset_sha256": {name: sha256(path) for name, path in BENCHMARKS.items()},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    env = {
        "commit_sha": base_sha, "python": platform.python_version(),
        "torch": torch.__version__, "transformers": package_version("transformers"),
        "bitsandbytes": package_version("bitsandbytes"), "cuda": torch.version.cuda,
        "device": args.device, "gpu_name": torch.cuda.get_device_name(args.device),
        "initial_telemetry": gpu_snapshot(args.device),
    }
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    rng = random.Random(args.seed)
    cal_raw = out / "calibration/raw_measurements.jsonl"
    held_raw = out / "heldout/raw_measurements.jsonl"
    cycle_paths = {"calibration": out / "acceptance_cycles/calibration.jsonl",
                   "heldout": out / "acceptance_cycles/heldout.jsonl"}
    cal_vanilla: list[dict[str, Any]] = []

    # Warm the exact decode paths and action shapes used below. These runs are
    # discarded and never enter calibration or heldout statistics.
    warm_prompt = calibration[0]["prompt"]
    measure_vanilla(model, tokenizer, warm_prompt, min(args.max_new_tokens, 8), args.device)
    for action in ACTIONS:
        run_action(model, tokenizer, manager, warm_prompt, action, 2,
                   min(args.max_new_tokens, 8), args.device)
    torch.cuda.synchronize(args.device)
    env["post_warmup_telemetry"] = gpu_snapshot(args.device)
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    # Measure the calibration workload's vanilla decode baseline. Cost-profile
    # coefficients are unchanged; this is the only newly estimated scalar.
    for record in calibration:
        prompt = record["prompt"]
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        telemetry_before = gpu_snapshot(args.device)
        measured = measure_vanilla(model, tokenizer, prompt, args.max_new_tokens, args.device)
        telemetry_after = gpu_snapshot(args.device)
        row = {"split": "calibration", "task": record["task"], "prompt_id": str(record["id"]),
               "prompt_tokens": int(ids.shape[-1]), "method": "vanilla", **measured,
               "telemetry_before": telemetry_before, "telemetry_after": telemetry_after}
        append_jsonl(cal_raw, row)
        cal_vanilla.append(row)
    total_decode_tokens = sum(r["decode_tokens"] for r in cal_vanilla)
    total_decode_seconds = sum(r["decode_ms"] for r in cal_vanilla) / 1000.0
    calibrated_baseline_tps = total_decode_tokens / max(total_decode_seconds, 1e-9)
    cost_model = MeasuredActionCostModel.for_model("qwen25_3b", baseline_tps=calibrated_baseline_tps)
    (out / "calibration/baseline.json").write_text(json.dumps({
        "baseline_tps_decode": calibrated_baseline_tps,
        "n_prompts": len(cal_vanilla),
        "calibration_prompt_ids": [{"task": r["task"], "id": r["prompt_id"]} for r in cal_vanilla],
        "cost_model_profile": "Qwen25_3B_CostProfile defaults from src/zassd/profiling/action_cost_model.py",
        "fit_policy": "baseline throughput only; all action cost/acceptance coefficients frozen",
    }, indent=2))

    # Record predictions before action execution. Calibration action rows are a
    # diagnostic; heldout metrics are evaluated only on CNN/DailyMail prompts.
    for split_name, records, raw_path in (
        ("calibration", calibration, cal_raw), ("heldout", heldout, held_raw)
    ):
        for record_index, record in enumerate(records):
            prompt = record["prompt"]
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            vanilla: dict[str, Any] | None = None
            action_jobs = [(action, k) for action in ACTIONS for k in K_VALUES]
            rng.shuffle(action_jobs)
            if split_name == "heldout":
                # Rotate vanilla's position among actions to reduce order bias.
                jobs: list[tuple[str, Any, int | None]] = [("action", a, k) for a, k in action_jobs]
                jobs.insert(rng.randrange(len(jobs) + 1), ("vanilla", None, None))
            else:
                jobs = [("action", a, k) for a, k in action_jobs]
            pending_action_rows: list[dict[str, Any]] = []
            for kind, action, k in jobs:
                if kind == "vanilla":
                    telemetry_before = gpu_snapshot(args.device)
                    vanilla = measure_vanilla(model, tokenizer, prompt, args.max_new_tokens, args.device)
                    telemetry_after = gpu_snapshot(args.device)
                    append_jsonl(raw_path, {
                        "split": split_name, "task": record["task"], "prompt_id": str(record["id"]),
                        "prompt_tokens": int(ids.shape[-1]), "method": "vanilla", **vanilla,
                        "telemetry_before": telemetry_before, "telemetry_after": telemetry_after,
                    })
                    continue
                prediction = cost_model.evaluate_action(action["config"], k)
                telemetry_before = gpu_snapshot(args.device)
                actual = run_action(model, tokenizer, manager, prompt, action, k,
                                    args.max_new_tokens, args.device)
                for cycle_index, cycle in enumerate(actual.pop("acceptance_cycles")):
                    accepted = int(cycle.get("accepted", 0))
                    proposed = int(cycle.get("k", 0))
                    append_jsonl(cycle_paths[split_name], {
                        "split": split_name, "task": record["task"],
                        "prompt_id": str(record["id"]), "action": action["config"], "k": k,
                        "cycle_index": cycle_index,
                        "context_length": cycle.get("cycle_context_length", int(ids.shape[-1])),
                        "draft_tokens": proposed, "accepted_tokens": accepted,
                        "emitted_tokens": int(cycle.get("emitted_count", 0)),
                        "accepted_survival": [int(accepted >= i) for i in range(1, proposed + 1)],
                        "draft_ms": float(cycle.get("draft_time_ms", 0.0)),
                        "verify_ms": float(cycle.get("verify_time_ms", 0.0)),
                        "cache_ms": float(cycle.get("cache_time_ms", 0.0)),
                        "entropy": cycle.get("cycle_entropy"),
                        "logit_margin": cycle.get("cycle_logit_margin"),
                        "source": cycle.get("source", "layer_skip"),
                    })
                telemetry_after = gpu_snapshot(args.device)
                row = {
                    "split": split_name, "task": record["task"], "prompt_id": str(record["id"]),
                    "prompt_tokens": int(ids.shape[-1]), "context_bucket": (
                        "short" if ids.shape[-1] < 512 else "medium" if ids.shape[-1] < 2048 else "long"),
                    "method": "layer_skip", "action": action["config"],
                    "kept_layers": action["kept_layers"], "k": k,
                    "prediction": {
                        "draft_ms_per_cycle": prediction.draft_ms,
                        "verify_ms_per_cycle": prediction.verify_ms,
                        "cache_ms_per_cycle": cost_model.default_cache_ms,
                        "total_cycle_ms": prediction.total_cycle_ms,
                        "acceptance_rate": prediction.expected_acceptance,
                        "tokens_per_cycle": prediction.expected_tokens_per_step,
                        "decode_tps": prediction.expected_tps,
                        "speedup_vs_calibration_vanilla": prediction.expected_speedup,
                    },
                    "measured": actual,
                    "measured_speedup_vs_prompt_vanilla": None,
                    "telemetry_before": telemetry_before, "telemetry_after": telemetry_after,
                }
                if split_name == "heldout":
                    pending_action_rows.append(row)
                else:
                    append_jsonl(raw_path, row)
                print(f"[{split_name}] {record['task']}:{record['id']} {action['config']} K={k} "
                      f"pred={prediction.expected_tps:.2f} measured={actual['decode_tps']:.2f} tok/s",
                      flush=True)
            if split_name == "heldout":
                if vanilla is None:
                    raise RuntimeError(f"Vanilla control missing for heldout prompt {record['id']}")
                for row in pending_action_rows:
                    if vanilla["decode_tps"] > 0:
                        row["measured_speedup_vs_prompt_vanilla"] = (
                            row["measured"]["decode_tps"] / vanilla["decode_tps"])
                    append_jsonl(raw_path, row)

    print(f"DONE: {out}")


if __name__ == "__main__":
    main()
