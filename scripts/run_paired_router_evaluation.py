#!/usr/bin/env python3
"""Paired WP4 evaluation on the frozen WP3 token-ID cohort.

Methods are run in a deterministic randomized order per prompt with tracing
disabled. The development subset selects a single best fixed method; the
held-out subset is used only for final evaluation.
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
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zassd.baselines.prompt_lookup import prompt_lookup_generate  # noqa: E402
from zassd.cache.kv_cache import TargetKVCache  # noqa: E402
from zassd.decoding.speculative import self_speculative_generate  # noqa: E402
from zassd.models.layer_manager import LayerManager  # noqa: E402
from zassd.models.model_adapter import ModelAdapter  # noqa: E402
from zassd.profiling.action_cost_model import MeasuredActionCostModel  # noqa: E402
from zassd.routing.hybrid_router import HybridDraftRouter  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
INPUTS = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl"
TRACE_CFG = ROOT / "experiments/numerical_causality/traces/config.json"
WP3_DIR = ROOT / "experiments/numerical_causality/precision_validation_final"
DEFAULT_OUT = ROOT / "experiments/numerical_causality/unified_router_evaluation"
CKA90_SKIP = [4, 5, 6, 7]
FIXED_METHODS = ("pld_k2", "pld_k4", "pld_k8", "cka90_k2")
ALL_METHODS = ("direct_vanilla", "vanilla_speculative_core_k0", *FIXED_METHODS, "hybrid_router")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def select_cohort() -> list[tuple[dict, str]]:
    inputs = [json.loads(line) for line in INPUTS.read_text().splitlines() if line.strip()]
    config = json.loads(TRACE_CFG.read_text())
    diag_ids = set(config["ids"])
    wp3_records = [json.loads(line) for line in (WP3_DIR / "sdpa_static.jsonl").read_text().splitlines() if line.strip()]
    labels = {r["prompt_id"]: r["cohort"] for r in wp3_records}
    rows = [(r, labels[r["prompt_id"]]) for r in inputs if r["prompt_id"] in labels]
    if len(rows) != 30 or sum(label == "diagnostic" for _, label in rows) != 16:
        raise RuntimeError("WP3 cohort mismatch: expected 16 diagnostic + 14 heldout prompts")
    if {r["prompt_id"] for r, label in rows if label == "diagnostic"} != diag_ids:
        raise RuntimeError("WP3 diagnostic prompt IDs differ from the frozen trace cohort")
    return rows


def direct_vanilla(model, prompt_ids: torch.Tensor, device: str, budget: int,
                   capacity: int) -> dict[str, Any]:
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    generated: list[int] = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        prefill_start = time.perf_counter()
        out = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
        torch.cuda.synchronize(device)
        prefill_s = time.perf_counter() - prefill_start
        current = int(out.logits[0, -1].argmax())
        while len(generated) < budget:
            generated.append(current)
            if current == tokenizer_eos:
                break
            if len(generated) == budget:
                break
            token = torch.tensor([[current]], dtype=torch.long, device=device)
            out = model(token, past_key_values=cache.cache, use_cache=True)
            current = int(out.logits[0, -1].argmax())
    torch.cuda.synchronize(device)
    total_s = time.perf_counter() - t0
    decode_s = max(total_s - prefill_s, 1e-9)
    return {"generated_token_ids": generated, "total_time_s": total_s,
            "prefill_time_s": prefill_s, "decode_time_s": decode_s,
            "decode_tps": max(0, len(generated) - 1) / decode_s,
            "tokens_per_second": len(generated) / max(total_s, 1e-9),
            "verification_time_s": decode_s, "acceptance_rate": None,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "action_counts": {"vanilla": 1}, "verification_cycles": max(0, len(generated) - 1)}


def speculative_metrics(metrics, device: str) -> dict[str, Any]:
    total = float(metrics.total_time_s)
    prefill = float(metrics.prefill_time_s)
    decode = max(total - prefill, 1e-9)
    token_ids = list(metrics.generated_token_ids)
    counts: dict[str, int] = {}
    for stat in metrics.per_iteration_stats:
        source = stat.get("router_source") or stat.get("source")
        if source:
            counts[source] = counts.get(source, 0) + 1
    return {"generated_token_ids": token_ids, "total_time_s": total,
            "prefill_time_s": prefill, "decode_time_s": decode,
            "decode_tps": max(0, len(token_ids) - 1) / decode,
            "tokens_per_second": len(token_ids) / max(total, 1e-9),
            "verification_time_s": float(metrics.verify_time_s),
            "acceptance_rate": float(metrics.acceptance_rate),
            "peak_vram_mb": float(metrics.peak_vram_mb), "action_counts": counts,
            "verification_cycles": int(metrics.num_verification_cycles),
            "draft_tokens": int(metrics.total_draft_tokens),
            "accepted_tokens": int(metrics.total_accepted_tokens)}


def execute_method(name: str, model, tokenizer, manager, prompt_ids: torch.Tensor,
                   device: str, budget: int, capacity: int, cost_model) -> dict[str, Any]:
    if name == "direct_vanilla":
        return direct_vanilla(model, prompt_ids, device, budget, capacity)
    if name.startswith("pld_k"):
        k = int(name.removeprefix("pld_k"))
        _, metrics = prompt_lookup_generate(model, tokenizer, prompt_ids, k=k,
            ngram_size=3, max_new_tokens=budget, temperature=0.0, device=device,
            kv_cache_backend="static", k_policy="fixed",
            candidate_source="prompt+generated", kv_cache_max_capacity=capacity)
        return speculative_metrics(metrics, device)
    if name == "vanilla_speculative_core_k0":
        _, metrics = self_speculative_generate(model, tokenizer, manager, [], prompt_ids,
            k=0, max_new_tokens=budget, temperature=0.0, device=device,
            kv_cache_backend="static", kv_cache_max_capacity=capacity,
            draft_mode="layer_skip", config_name="vanilla_k0")
        return speculative_metrics(metrics, device)
    if name == "cka90_k2":
        _, metrics = self_speculative_generate(model, tokenizer, manager, CKA90_SKIP,
            prompt_ids, k=2, max_new_tokens=budget, temperature=0.0, device=device,
            kv_cache_backend="static", kv_cache_max_capacity=capacity,
            draft_mode="layer_skip", config_name="cka_90")
        return speculative_metrics(metrics, device)
    if name == "hybrid_router":
        router = HybridDraftRouter(cost_model=cost_model, pld_k=4)
        _, metrics = self_speculative_generate(model, tokenizer, manager, CKA90_SKIP,
            prompt_ids, k=4, router=router, config_name="cka_90",
            max_new_tokens=budget, temperature=0.0, device=device,
            kv_cache_backend="static", kv_cache_max_capacity=capacity,
            draft_mode="routed")
        result = speculative_metrics(metrics, device)
        result["router_stats"] = router.stats.to_dict()
        return result
    raise ValueError(name)


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    global tokenizer_eos
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--order-seed", type=int, default=20261015)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pilot", type=int, default=0, help="Run only the first N prompts for a schema/runtime pilot")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA device is required")
    outdir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    cohort = select_cohort()
    if args.pilot:
        cohort = cohort[:args.pilot]
    diagnostic_ids = sorted(r["prompt_id"] for r, label in cohort if label == "diagnostic")
    development_ids = set(sorted(diagnostic_ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:8])
    ids_hash = digest(INPUTS.read_bytes())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    config = {"execution_commit": commit, "model": MODEL, "model_revision": REVISION,
              "tokenizer_revision": REVISION, "quantization": {"type": "NF4", "compute_dtype": "float16", "double_quant": True},
              "attention": "sdpa", "kv_backend": "static", "max_new_tokens": args.max_new_tokens,
              "K_values": {"pld": [2, 4, 8], "cka90": 2, "router_pld_max": 4},
              "candidate_source": "prompt+generated", "methods": list(ALL_METHODS),
              "method_order_seed": args.order_seed, "randomized_order": "per prompt, seeded Python RNG",
              "tracing": False, "input_path": str(INPUTS.relative_to(ROOT)), "input_sha256": ids_hash,
              "diagnostic_ids": [r["prompt_id"] for r, label in cohort if label == "diagnostic"],
              "cohort_counts": {c: sum(label == c for _, label in cohort) for c in ("diagnostic", "heldout")},
              "development_split": sorted(development_ids),
              "selection_rule": "choose fixed method by mean decode TPS on the 8 diagnostic development IDs; held-out is never used to select",
              "fixed_method_candidates": list(FIXED_METHODS),
              "limitations": "WP3 prepared-input heldout subset is 14 prompts; development prompt IDs are mismatch-diagnostic and may not represent natural traffic."}
    (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "bitsandbytes": package_version("bitsandbytes"),
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device)}
    (outdir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer_eos = tokenizer.eos_token_id
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        quantization_config=quant, device_map=args.device, attn_implementation="sdpa").eval()
    manager = LayerManager(ModelAdapter(model))
    cost_model = MeasuredActionCostModel.for_model("qwen25_3b")
    raw_path = outdir / "raw_measurements.jsonl"
    rng = random.Random(args.order_seed)
    base_order = list(ALL_METHODS)
    rng.shuffle(base_order)
    # Warm every path outside the measured cohort. The shared randomized base
    # order is rotated per prompt, balancing each method's ordinal position.
    warm_prompt = torch.tensor([cohort[0][0]["input_ids"][:32]], dtype=torch.long, device=args.device)
    warm_capacity = int(warm_prompt.shape[-1]) + 12
    for name in base_order:
        execute_method(name, model, tokenizer, manager, warm_prompt, args.device, 4,
                       warm_capacity, cost_model)
    torch.cuda.synchronize(args.device)
    all_records = []
    for idx, (row, cohort_label) in enumerate(cohort):
        prompt_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=args.device)
        capacity = len(row["input_ids"]) + args.max_new_tokens + 24
        offset = idx % len(base_order)
        order = base_order[offset:] + base_order[:offset]
        measurements = {}
        for name in order:
            # Reset all device-side timing/peak stats before each paired method.
            torch.cuda.synchronize(args.device)
            data = execute_method(name, model, tokenizer, manager, prompt_ids,
                                  args.device, args.max_new_tokens, capacity, cost_model)
            measurements[name] = data
        reference = measurements["direct_vanilla"]["generated_token_ids"]
        for name in ALL_METHODS:
            data = measurements[name]
            generated = data["generated_token_ids"]
            data["sequence_exact_match"] = generated == reference
            data["first_divergence_position"] = first_diff(reference, generated)
            data["eos_position"] = generated.index(tokenizer_eos) if tokenizer_eos in generated else None
            data["paired_speedup_vs_vanilla"] = data["decode_tps"] / max(measurements["direct_vanilla"]["decode_tps"], 1e-9)
            data["slowdown_vs_vanilla"] = data["paired_speedup_vs_vanilla"] < 1.0
            record = {"prompt_id": row["prompt_id"], "cohort": cohort_label,
                      "selection_role": ("development" if row["prompt_id"] in development_ids
                                         else ("heldout_evaluation" if cohort_label == "heldout" else "diagnostic_evaluation")),
                      "source_task": row.get("source_task"), "source_id": row.get("source_id"),
                      "prompt_token_ids_sha256": digest(json.dumps(row["input_ids"], separators=(",", ":")).encode()),
                      "prompt_tokens": len(row["input_ids"]), "method_order": order,
                      "method": name, "measurement": data}
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
            all_records.append(record)
        print(f"completed {idx+1}/{len(cohort)} {row['prompt_id']} ({cohort_label}) order={','.join(order)}", flush=True)
    manifest = {**config, **env, "raw_file": str(raw_path),
                "raw_sha256": digest(raw_path.read_bytes()), "record_count": len(all_records)}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"WP4 paired run saved: {outdir}", flush=True)


if __name__ == "__main__":
    main()
