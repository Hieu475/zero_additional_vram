#!/usr/bin/env python3
"""Controlled 2x2 replay of NF4 double quantization and static KV capacity.

The timing pass is uninstrumented: it records only wall-clock and CUDA peak
memory. A separate correctness pass records prompt-prefill logits/KV and
logits only while each variant still has the same token prefix as the
double-quantized, capacity-436 reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zassd.cache.kv_cache import TargetKVCache  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
INPUTS = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl"
TRACE_CONFIG = ROOT / "experiments/numerical_causality/traces/config.json"
OUT = ROOT / "experiments/backend_validation/capacity_control"
CAPACITIES = (436, 2048)
QUANT_FLAGS = (True, False)


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    delta = (actual.float() - reference.float()).abs()
    ref = reference.float()
    return {
        "mae": float(delta.mean()),
        "max_abs": float(delta.max()),
        "relative_rms": float(delta.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-12)),
        "bitwise_equal": bool(torch.equal(actual, reference)),
    }


def snapshot_active_kv(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    snapshots = []
    for layer in cache.layers:
        n = int(layer.get_seq_length())
        if n and getattr(layer, "is_initialized", False):
            snapshots.append((layer.keys[:, :, :n].detach().cpu().clone(),
                              layer.values[:, :, :n].detach().cpu().clone(), n))
        else:
            snapshots.append((torch.empty(0), torch.empty(0), n))
    return snapshots


def compare_kv(actual: list[tuple[torch.Tensor, torch.Tensor, int]],
               reference: list[tuple[torch.Tensor, torch.Tensor, int]]) -> dict[str, Any]:
    layers = []
    for idx, (a, r) in enumerate(zip(actual, reference)):
        ak, av, an = a
        rk, rv, rn = r
        entry: dict[str, Any] = {"layer": idx, "actual_length": an, "reference_length": rn}
        if an != rn or ak.shape != rk.shape:
            entry["shape_or_length_mismatch"] = True
        elif an:
            entry["key"] = error_metrics(ak, rk)
            entry["value"] = error_metrics(av, rv)
        layers.append(entry)
    numeric = [x for x in layers if "key" in x]
    return {
        "logical_length_equal": all(x["actual_length"] == x["reference_length"] for x in layers),
        "all_active_kv_bitwise_equal": bool(numeric) and all(
            x["key"]["bitwise_equal"] and x["value"]["bitwise_equal"] for x in numeric
        ),
        "max_key_abs_error": max((x["key"]["max_abs"] for x in numeric), default=None),
        "max_value_abs_error": max((x["value"]["max_abs"] for x in numeric), default=None),
        "layers": layers,
    }


def compare_logits_on_shared_prefix(actual_logits: list[torch.Tensor],
                                    reference_logits: list[torch.Tensor],
                                    actual_tokens: list[int],
                                    reference_tokens: list[int]) -> list[dict[str, Any]]:
    """Compare rows through and including the first differing next-token choice."""
    first = next((j for j, (a, b) in enumerate(zip(actual_tokens, reference_tokens)) if a != b),
                 min(len(actual_tokens), len(reference_tokens))
                 if len(actual_tokens) != len(reference_tokens) else None)
    rows = []
    for pos, (a, b) in enumerate(zip(actual_logits, reference_logits)):
        if first is not None and pos > first:
            break
        rows.append({"generated_prefix_length": pos,
                     **error_metrics(a, b.to(dtype=a.dtype))})
    return rows


def load_diagnostic_rows(inputs_path: Path, config_path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in inputs_path.read_text().splitlines() if x.strip()]
    ids = json.loads(config_path.read_text())["ids"]
    by_id = {r["prompt_id"]: r for r in rows}
    missing = [pid for pid in ids if pid not in by_id]
    if missing:
        raise ValueError(f"Missing configured diagnostic prompt IDs: {missing}")
    return [by_id[pid] for pid in ids]


def run_one(model: Any, ids: list[int], capacity: int, budget: int,
            eos_id: int | None, device: str, collect_correctness: bool) -> dict[str, Any]:
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    if collect_correctness:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=x, past_key_values=cache.cache, use_cache=True)
        prefill_logits = out.logits[0, -1].detach().cpu() if collect_correctness else None
        next_id = int(out.logits[0, -1].argmax().item())
        initial_kv = snapshot_active_kv(cache.cache) if collect_correctness else None
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0
        tokens: list[int] = []
        decode_logits: list[torch.Tensor] = []
        if collect_correctness:
            decode_logits.append(prefill_logits)
        t_decode_start = time.perf_counter()
        while len(tokens) < budget:
            tokens.append(next_id)
            if next_id == eos_id or len(tokens) >= budget:
                break
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(input_ids=x, past_key_values=cache.cache, use_cache=True)
            if collect_correctness:
                decode_logits.append(out.logits[0, -1].detach().cpu().clone())
            next_id = int(out.logits[0, -1].argmax().item())
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t_decode_start
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2) if not collect_correctness else None
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2) if not collect_correctness else None
    return {
        "token_ids": tokens,
        "first_divergence_vs_reference": None,
        "eos_position": tokens.index(eos_id) if eos_id in tokens else None,
        "prompt_prefill_logits": prefill_logits,
        "prompt_prefill_kv": initial_kv,
        "same_prefix_logits": decode_logits if collect_correctness else None,
        "prefill_latency_s": t_prefill,
        "decode_latency_s": decode_s,
        "end_to_end_latency_s": t_prefill + decode_s,
        "tokens_per_second_e2e": len(tokens) / max(t_prefill + decode_s, 1e-12),
        "peak_allocated_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
        "final_kv_length": cache.get_seq_length(),
    }


def build_model(double_quant: bool, device: str) -> Any:
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=double_quant)
    return AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, quantization_config=q, device_map=device,
        attn_implementation="sdpa").eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--trace-config", type=Path, default=TRACE_CONFIG)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ids", nargs="*", help="Optional subset of configured prompt IDs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NF4 capacity control")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite nonempty {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_diagnostic_rows(args.inputs, args.trace_config)
    if args.ids:
        by_id = {r["prompt_id"]: r for r in prompts}
        missing = [pid for pid in args.ids if pid not in by_id]
        if missing:
            raise ValueError(f"Requested IDs are not in the diagnostic cohort: {missing}")
        prompts = [by_id[pid] for pid in args.ids]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()

    # Reference A: double quantization enabled, capacity 436.
    refs_by_capacity: dict[tuple[str, int], dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    for double_quant in QUANT_FLAGS:
        model = build_model(double_quant, args.device)
        for idx, row in enumerate(prompts):
            pid = row["prompt_id"]
            ids = list(row["input_ids"])
            cap_order = list(CAPACITIES) if idx % 2 == 0 else list(reversed(CAPACITIES))
            captures: dict[int, dict[str, Any]] = {}
            timings: dict[int, dict[str, Any]] = {}
            # Correctness capture is isolated from timing; no trace hook is installed.
            for capacity in cap_order:
                correctness = run_one(model, ids, capacity, args.max_new_tokens,
                                      tok.eos_token_id, args.device, True)
                captures[capacity] = correctness
                if double_quant:
                    refs_by_capacity[(pid, capacity)] = correctness
            for capacity in cap_order:
                timing = run_one(model, ids, capacity, args.max_new_tokens,
                                 tok.eos_token_id, args.device, False)
                timings[capacity] = timing
            for capacity in CAPACITIES:
                result = captures[capacity]
                reference = refs_by_capacity[(pid, capacity)]
                other_capacity = next(c for c in CAPACITIES if c != capacity)
                capacity_reference = captures[other_capacity]
                ref_tokens = reference["token_ids"]
                current = result["token_ids"]
                first = next((j for j, (a, b) in enumerate(zip(current, ref_tokens)) if a != b),
                             min(len(current), len(ref_tokens)) if len(current) != len(ref_tokens) else None)
                logit_diffs = compare_logits_on_shared_prefix(
                    result["same_prefix_logits"] or [],
                    reference["same_prefix_logits"] or [], current, ref_tokens)
                capacity_logit_diffs = compare_logits_on_shared_prefix(
                    result["same_prefix_logits"] or [],
                    capacity_reference["same_prefix_logits"] or [],
                    current, capacity_reference["token_ids"])
                timing = timings[capacity]
                row_result = {
                    "prompt_id": pid,
                    "prompt_token_ids_sha256": sha(ids),
                    "prompt_tokens": len(ids),
                    "double_quant": double_quant,
                    "kv_capacity": capacity,
                    "token_ids": current,
                    "same_capacity_double_quant_reference_token_ids": ref_tokens,
                    "sequence_exact_match_vs_double_quant_same_capacity": current == ref_tokens,
                    "first_divergence_position_vs_double_quant_same_capacity": first,
                    "eos_position": result["eos_position"],
                    "logical_prefix_logits_vs_double_quant_same_capacity": logit_diffs,
                    "max_logit_abs_error_same_prefix_vs_double_quant_same_capacity": max((x["max_abs"] for x in logit_diffs), default=None),
                    "active_kv_at_prompt_prefill_vs_double_quant_same_capacity": compare_kv(result["prompt_prefill_kv"], reference["prompt_prefill_kv"]),
                    "capacity_pair_comparison": {
                        "other_capacity": other_capacity,
                        "token_ids_equal": current == capacity_reference["token_ids"],
                        "first_divergence_position": next((j for j, (a, b) in enumerate(zip(current, capacity_reference["token_ids"])) if a != b),
                                                          min(len(current), len(capacity_reference["token_ids"])) if len(current) != len(capacity_reference["token_ids"]) else None),
                        "same_prefix_logits": capacity_logit_diffs,
                        "max_logit_abs_error_same_prefix": max((x["max_abs"] for x in capacity_logit_diffs), default=None),
                        "active_kv_at_prompt_prefill": compare_kv(result["prompt_prefill_kv"], capacity_reference["prompt_prefill_kv"]),
                    },
                    "prefill_latency_s": timing["prefill_latency_s"],
                    "decode_latency_s": timing["decode_latency_s"],
                    "end_to_end_latency_s": timing["end_to_end_latency_s"],
                    "tokens_per_second_e2e": timing["tokens_per_second_e2e"],
                    "peak_allocated_mb": timing["peak_allocated_mb"],
                    "peak_reserved_mb": timing["peak_reserved_mb"],
                    "final_kv_length": result["final_kv_length"],
                }
                output_rows.append(row_result)
            print(f"dq={double_quant} {idx+1}/{len(prompts)} {pid} complete", flush=True)
        del model
        torch.cuda.empty_cache()

    raw = args.output_dir / "raw_measurements.jsonl"
    with raw.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "execution_commit": commit,
        "model": MODEL,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "hardware": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "attention": "sdpa", "compute_dtype": "float16", "quantization": "NF4",
        "variants": [{"double_quant": dq, "kv_capacity": cap}
                     for dq in QUANT_FLAGS for cap in CAPACITIES],
        "reference_variant": {"double_quant": True, "kv_capacity": 436},
        "max_new_tokens": args.max_new_tokens,
        "prompt_source": str(args.inputs.relative_to(ROOT)),
        "prompt_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "diagnostic_ids": [r["prompt_id"] for r in prompts],
        "measurement_note": "Correctness replay captures prompt-prefill active KV and same-prefix logits. Timing is a separate direct greedy model-forward pass without tracing or per-step snapshots.",
        "raw_measurements": str(raw.relative_to(ROOT)) if raw.is_relative_to(ROOT) else str(raw),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary: dict[str, Any] = {}
    for dq in QUANT_FLAGS:
        for cap in CAPACITIES:
            key = f"double_quant={dq},capacity={cap}"
            rows = [r for r in output_rows if r["double_quant"] == dq and r["kv_capacity"] == cap]
            summary[key] = {
                "n": len(rows),
                "exact_vs_double_quant_same_capacity": sum(r["sequence_exact_match_vs_double_quant_same_capacity"] for r in rows),
                "mean_prefill_s": sum(r["prefill_latency_s"] for r in rows) / len(rows),
                "mean_decode_s": sum(r["decode_latency_s"] for r in rows) / len(rows),
                "mean_tps": sum(r["tokens_per_second_e2e"] for r in rows) / len(rows),
                "mean_peak_allocated_mb": sum(r["peak_allocated_mb"] for r in rows) / len(rows),
                "mean_peak_reserved_mb": sum(r["peak_reserved_mb"] for r in rows) / len(rows),
                "mean_max_logit_abs_error_same_prefix_vs_double_quant_same_capacity": sum(
                    r["max_logit_abs_error_same_prefix_vs_double_quant_same_capacity"] or 0.0 for r in rows) / len(rows),
                "active_kv_prefill_exact_vs_double_quant_same_capacity": sum(r["active_kv_at_prompt_prefill_vs_double_quant_same_capacity"]["all_active_kv_bitwise_equal"] for r in rows),
                "capacity_pair_token_ids_exact": sum(r["capacity_pair_comparison"]["token_ids_equal"] for r in rows),
                "capacity_pair_kv_prefill_exact": sum(r["capacity_pair_comparison"]["active_kv_at_prompt_prefill"]["all_active_kv_bitwise_equal"] for r in rows),
            }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
