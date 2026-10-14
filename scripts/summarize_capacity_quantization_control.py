#!/usr/bin/env python3
"""Analyze the controlled KV-capacity × NF4 double-quantization replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_mean(values: list[float], seed: int = 20261017, n: int = 5000) -> list[float]:
    rng = random.Random(seed)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(n)]
    samples.sort()
    return [samples[int(0.025 * n)], samples[int(0.975 * n)]]


def pair_result(left: list[dict[str, Any]], right: list[dict[str, Any]],
                label: str, kv_key: str, logits_key: str,
                cap_pair: bool = False) -> dict[str, Any]:
    lmap = {r["prompt_id"]: r for r in left}
    rmap = {r["prompt_id"]: r for r in right}
    if set(lmap) != set(rmap):
        raise ValueError(f"Prompt mismatch in contrast {label}")
    token_same, first_diffs, decode_tps_ratios, e2e_tps_ratios = [], [], [], []
    decode_ratios, prefill_diffs, allocated_diffs, reserved_diffs = [], [], [], []
    logit_max, kv_key_max, kv_value_max, kv_equal = [], [], [], []
    for pid, a in lmap.items():
        b = rmap[pid]
        token_same.append(a["token_ids"] == b["token_ids"])
        first = next((i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y),
                     min(len(a["token_ids"]), len(b["token_ids"]))
                     if len(a["token_ids"]) != len(b["token_ids"]) else None)
        if first is not None:
            first_diffs.append(first)
        decode_tps_a = max(0, len(a["token_ids"]) - 1) / max(a["decode_latency_s"], 1e-12)
        decode_tps_b = max(0, len(b["token_ids"]) - 1) / max(b["decode_latency_s"], 1e-12)
        decode_tps_ratios.append(decode_tps_b / max(decode_tps_a, 1e-12))
        e2e_tps_ratios.append(b["tokens_per_second_e2e"] / max(a["tokens_per_second_e2e"], 1e-12))
        decode_ratios.append(b["decode_latency_s"] / max(a["decode_latency_s"], 1e-12))
        prefill_diffs.append(b["prefill_latency_s"] - a["prefill_latency_s"])
        allocated_diffs.append(b["peak_allocated_mb"] - a["peak_allocated_mb"])
        reserved_diffs.append(b["peak_reserved_mb"] - a["peak_reserved_mb"])
        if cap_pair:
            kv = a["capacity_pair_comparison"]["active_kv_at_prompt_prefill"]
            logit = a["capacity_pair_comparison"]["max_logit_abs_error_same_prefix"]
        else:
            kv = b[kv_key]
            logit = b[logits_key]
        kv_equal.append(kv["all_active_kv_bitwise_equal"])
        kv_key_max.append(kv["max_key_abs_error"] or 0.0)
        kv_value_max.append(kv["max_value_abs_error"] or 0.0)
        logit_max.append(logit or 0.0)
    return {
        "contrast": label,
        "n": len(left),
        "token_sequences_equal": sum(token_same),
        "token_sequences_changed": len(token_same) - sum(token_same),
        "first_divergence_positions": first_diffs,
        "mean_decode_tps_ratio_right_over_left": mean(decode_tps_ratios),
        "decode_tps_ratio_bootstrap_95_ci": bootstrap_mean(decode_tps_ratios),
        "mean_e2e_tps_ratio_right_over_left": mean(e2e_tps_ratios),
        "mean_decode_latency_ratio_right_over_left": mean(decode_ratios),
        "mean_prefill_latency_delta_s_right_minus_left": mean(prefill_diffs),
        "mean_peak_allocated_delta_mb_right_minus_left": mean(allocated_diffs),
        "mean_peak_reserved_delta_mb_right_minus_left": mean(reserved_diffs),
        "active_kv_prefill_bitwise_equal": sum(kv_equal),
        "mean_max_key_abs_error": mean(kv_key_max),
        "mean_max_value_abs_error": mean(kv_value_max),
        "mean_max_logit_abs_error_same_prefix": mean(logit_max),
        "max_logit_abs_error_same_prefix": max(logit_max, default=0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, nargs="?",
                    default=Path("experiments/backend_validation/capacity_control"))
    args = ap.parse_args()
    raw_path = args.run_dir / "raw_measurements.jsonl"
    rows = [json.loads(x) for x in raw_path.read_text().splitlines() if x.strip()]
    keyed = {(r["prompt_id"], r["double_quant"], r["kv_capacity"]): r for r in rows}
    prompts = sorted({r["prompt_id"] for r in rows})
    expected = {(pid, dq, cap) for pid in prompts for dq in (True, False) for cap in (436, 2048)}
    if set(keyed) != expected or len(rows) != len(expected):
        raise ValueError(f"Expected {len(expected)} unique rows; got {len(rows)}")
    g = lambda dq, cap: [keyed[(pid, dq, cap)] for pid in prompts]
    contrasts = [
        pair_result(g(True, 436), g(True, 2048), "capacity 436→2048, double_quant=True",
                    "", "", cap_pair=True),
        pair_result(g(False, 436), g(False, 2048), "capacity 436→2048, double_quant=False",
                    "", "", cap_pair=True),
        pair_result(g(True, 436), g(False, 436), "double_quant True→False, capacity=436",
                    "active_kv_at_prompt_prefill_vs_double_quant_same_capacity",
                    "max_logit_abs_error_same_prefix_vs_double_quant_same_capacity"),
        pair_result(g(True, 2048), g(False, 2048), "double_quant True→False, capacity=2048",
                    "active_kv_at_prompt_prefill_vs_double_quant_same_capacity",
                    "max_logit_abs_error_same_prefix_vs_double_quant_same_capacity"),
    ]
    summary = {"n_prompts": len(prompts), "variants": 4, "rows": len(rows), "contrasts": contrasts}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sha256 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    runner = ROOT / "scripts/run_capacity_quantization_control.py"
    trace_config = ROOT / "experiments/numerical_causality/traces/config.json"
    manifest["trace_config_sha256"] = sha256(trace_config)
    manifest["runner_sha256"] = sha256(runner)
    manifest["analysis_script_sha256"] = sha256(Path(__file__))
    manifest["raw_measurements_sha256"] = sha256(raw_path)
    manifest["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
    manifest["worktree_was_dirty_during_run"] = True
    manifest["run_scope"] = "single session; diagnostic prompts; descriptive performance only"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# KV capacity × NF4 double-quantization control", "",
             f"Cohort: {len(prompts)} diagnostic prompts, 384 prompt tokens, 32-token greedy budget. ",
             "The reference protocol uses Qwen2.5-3B-Instruct, SDPA, FP16 compute, NF4 and a static cache. ",
             "Correctness capture and timing were separate passes; timing used direct model forwards without trace instrumentation.", "",
             "## Paired factorial contrasts", "",
             "| Contrast (right vs left) | Sequences equal | Mean decode TPS ratio [bootstrap 95% CI] | Prefill KV exact | Mean max logit abs error | Allocated VRAM delta |",
             "|---|---:|---:|---:|---:|---:|"]
    for c in contrasts:
        ci = c["decode_tps_ratio_bootstrap_95_ci"]
        lines.append(f"| {c['contrast']} | {c['token_sequences_equal']}/{c['n']} | {c['mean_decode_tps_ratio_right_over_left']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {c['active_kv_prefill_bitwise_equal']}/{c['n']} | {c['mean_max_logit_abs_error_same_prefix']:.6g} | {c['mean_peak_allocated_delta_mb_right_minus_left']:+.1f} MB |")
    lines += ["", "## Reading the result", "",
              "The capacity contrast is direct within each quantization setting. Prefill cache and logit comparisons use the same prompt token prefix; generated-prefix logits are compared only through the first position where the predicted token differs.",
              "Quantization contrasts compare outputs from the same capacity and therefore do not mix in the cache allocation difference.",
              "Latency and VRAM are descriptive for one run on one RTX 4050. Quantization order was fixed (double-quant on before off), so timing deltas are not treated as causal performance evidence. The unusually high peak allocated memory with double quantization disabled requires kernel-level profiling before assigning a cause.",
              "Capacity changes reserve additional static KV backing memory but do not alter the logical cache length. This cohort is diagnostic and does not support population-level generalization.", ""]
    (args.run_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
