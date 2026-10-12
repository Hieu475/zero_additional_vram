#!/usr/bin/env python3
"""Analyze held-out roofline measurements and write metrics/figures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mae_mape(pred: list[float], actual: list[float]) -> dict[str, float | None]:
    if not pred:
        return {"mae": None, "mape_pct": None}
    p, a = np.asarray(pred), np.asarray(actual)
    return {"mae": float(np.mean(np.abs(p - a))),
            "mape_pct": float(np.mean(np.abs(p - a) / np.maximum(np.abs(a), 1e-9)) * 100.0)}


def feasibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = np.asarray([r["measured_speedup_vs_prompt_vanilla"] > 1.0 for r in rows], dtype=bool)
    pred = np.asarray([r["prediction"]["speedup_vs_calibration_vanilla"] > 1.0 for r in rows], dtype=bool)
    tp = int(np.sum(pred & actual)); fp = int(np.sum(pred & ~actual))
    fn = int(np.sum(~pred & actual)); tn = int(np.sum(~pred & ~actual))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    specificity = float(tn / max(1, tn + fp))
    return {
        "n_action_prompt_pairs": int(len(rows)), "threshold": "speedup > 1.0",
        "accuracy": float((tp + tn) / max(1, len(rows))),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision_profitable": precision,
        "recall_profitable": recall,
        "specificity": specificity,
        "predicted_profitable_fraction": float(np.mean(pred)) if len(pred) else None,
        "measured_profitable_fraction": float(np.mean(actual)) if len(actual) else None,
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def prompt_bootstrap_ci(rows: list[dict[str, Any]], metric_fn, seed: int = 20261012,
                        n_boot: int = 4000) -> list[float] | None:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[f"{row['task']}:{row['prompt_id']}"].append(row)
    keys = list(by_prompt)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        sample_rows = [row for key in sampled for row in by_prompt[str(key)]]
        value = metric_fn(sample_rows)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def summarize(rows: list[dict[str, Any]], split: str, baseline_tps: float) -> dict[str, Any]:
    actions = [r for r in rows if r.get("method") == "layer_skip"]
    draft_p = [r["prediction"]["draft_ms_per_cycle"] for r in actions]
    draft_a = [r["measured"]["draft_ms_per_cycle"] for r in actions]
    verify_p = [r["prediction"]["verify_ms_per_cycle"] for r in actions]
    verify_a = [r["measured"]["verify_ms_per_cycle"] for r in actions]
    cycle_p = [r["prediction"]["total_cycle_ms"] for r in actions]
    cycle_a = [r["measured"]["decode_ms_per_cycle"] for r in actions]
    tpc_p = [r["prediction"]["tokens_per_cycle"] for r in actions]
    tpc_a = [r["measured"]["generated_tokens_per_cycle"] for r in actions]
    tps_p = [r["prediction"]["decode_tps"] for r in actions]
    tps_a = [r["measured"]["decode_tps"] for r in actions]
    acc_p = [r["prediction"]["acceptance_rate"] for r in actions]
    acc_a = [r["measured"]["acceptance_rate"] for r in actions]
    result: dict[str, Any] = {
        "split": split,
        "n_prompts": len({(r["task"], r["prompt_id"]) for r in actions}),
        "n_action_prompt_pairs": len(actions),
        "baseline_tps_calibration": baseline_tps,
        "component_errors": {
            "draft_ms_per_cycle": mae_mape(draft_p, draft_a),
            "verify_ms_per_cycle": mae_mape(verify_p, verify_a),
            "total_cycle_ms": mae_mape(cycle_p, cycle_a),
            "tokens_per_cycle": mae_mape(tpc_p, tpc_a),
            "acceptance_rate": mae_mape(acc_p, acc_a),
            "decode_tps": mae_mape(tps_p, tps_a),
        },
        "feasibility": feasibility(actions) if actions and all(
            r.get("measured_speedup_vs_prompt_vanilla") is not None for r in actions) else None,
        "actual_profitable_fraction": (float(np.mean([
            r["measured_speedup_vs_prompt_vanilla"] > 1.0 for r in actions
        ])) if actions and all(r.get("measured_speedup_vs_prompt_vanilla") is not None for r in actions) else None),
    }
    if actions and all(r.get("measured_speedup_vs_prompt_vanilla") is not None for r in actions):
        result["speedup_error_mae"] = float(np.mean(np.abs(
            np.asarray([r["prediction"]["speedup_vs_calibration_vanilla"] for r in actions]) -
            np.asarray([r["measured_speedup_vs_prompt_vanilla"] for r in actions])
        )))
        result["speedup_error_mae_ci95_prompt_bootstrap"] = prompt_bootstrap_ci(
            actions, lambda rs: float(np.mean([abs(
                r["prediction"]["speedup_vs_calibration_vanilla"] -
                r["measured_speedup_vs_prompt_vanilla"]
            ) for r in rs]))
        )
        result["mean_measured_speedup_ci95_prompt_bootstrap"] = prompt_bootstrap_ci(
            actions, lambda rs: float(np.mean([r["measured_speedup_vs_prompt_vanilla"] for r in rs]))
        )
        by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in actions:
            by_action[f"{row['action']}_K{row['k']}"].append(row)
            by_context[str(row.get("context_length", row.get("context_bucket", "unknown")))].append(row)
        result["by_action"] = {
            name: {
                "n_pairs": len(group),
                "draft_ms_mae": mae_mape(
                    [r["prediction"]["draft_ms_per_cycle"] for r in group],
                    [r["measured"]["draft_ms_per_cycle"] for r in group],
                )["mae"],
                "verify_ms_mae": mae_mape(
                    [r["prediction"]["verify_ms_per_cycle"] for r in group],
                    [r["measured"]["verify_ms_per_cycle"] for r in group],
                )["mae"],
                "predicted_decode_tps": float(np.mean([r["prediction"]["decode_tps"] for r in group])),
                "measured_decode_tps": float(np.mean([r["measured"]["decode_tps"] for r in group])),
                "mean_paired_speedup": float(np.mean([r["measured_speedup_vs_prompt_vanilla"] for r in group])),
                "measured_profitable_fraction": float(np.mean([
                    r["measured_speedup_vs_prompt_vanilla"] > 1.0 for r in group
                ])),
            } for name, group in by_action.items()
        }
        result["by_context_bucket"] = {
            name: {
                "n_pairs": len(group),
                "n_prompts": len({r["prompt_id"] for r in group}),
                "context_tokens_median": float(np.median([r["prompt_tokens"] for r in group])),
                "mean_paired_speedup": float(np.mean([r["measured_speedup_vs_prompt_vanilla"] for r in group])),
                "measured_profitable_fraction": float(np.mean([
                    r["measured_speedup_vs_prompt_vanilla"] > 1.0 for r in group
                ])),
            } for name, group in by_context.items()
        }
    return result


def make_figures(rows: list[dict[str, Any]], baseline_tps: float, figures: Path) -> None:
    actions = [r for r in rows if r.get("method") == "layer_skip"]
    figures.mkdir(parents=True, exist_ok=True)
    pred_tps = np.asarray([r["prediction"]["decode_tps"] for r in actions])
    actual_tps = np.asarray([r["measured"]["decode_tps"] for r in actions])
    upper = max(float(np.max(pred_tps)), float(np.max(actual_tps)), 1.0) * 1.05
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(pred_tps, actual_tps, alpha=0.75, s=30, c="#3569a8")
    ax.plot([0, upper], [0, upper], "k--", linewidth=1, label="y = x")
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Predicted decode throughput (tokens/s)",
           ylabel="Measured decode throughput (tokens/s)", title="Frozen cost model on held-out CNN/DM")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "predicted_vs_measured_throughput.pdf"); plt.close(fig)

    x = np.asarray([r["prediction"]["speedup_vs_calibration_vanilla"] for r in actions])
    y = np.asarray([r["measured_speedup_vs_prompt_vanilla"] for r in actions])
    profitable = y > 1.0
    upper = max(float(x.max()), float(y.max()), 1.0) * 1.08
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(x[~profitable], y[~profitable], alpha=0.8, s=32, label="Measured slower", marker="x", c="#c44336")
    ax.scatter(x[profitable], y[profitable], alpha=0.8, s=32, label="Measured faster", marker="o", c="#27835c")
    ax.plot([0, upper], [0, upper], "k--", linewidth=1, label="Predicted = measured")
    ax.axvline(1.0, color="#777", linestyle=":", linewidth=1)
    ax.axhline(1.0, color="#777", linestyle=":", linewidth=1)
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Predicted speedup vs calibration vanilla",
           ylabel="Measured paired speedup vs prompt vanilla", title="Held-out feasibility map")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "heldout_feasibility_map.pdf"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory created by runner")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    baseline = json.loads((run_dir / "calibration/baseline.json").read_text())
    cal = load_jsonl(run_dir / "calibration/raw_measurements.jsonl")
    held = load_jsonl(run_dir / "heldout/raw_measurements.jsonl")
    if "baseline_tps_decode_by_context" in baseline:
        baselines = {str(k): float(v) for k, v in baseline["baseline_tps_decode_by_context"].items()}
        baseline_tps = float(np.mean(list(baselines.values())))
    else:
        baseline_tps = float(baseline["baseline_tps_decode"])
    result = {
        "calibration": summarize(cal, "calibration", baseline_tps),
        "heldout": summarize(held, "heldout", baseline_tps),
        "interpretation": {
            "primary_test_split": "CNN/DailyMail; prompt IDs do not overlap GSM8K/HumanEval calibration workloads",
            "frozen_components": ["draft cost", "verify cost", "acceptance model", "cache overhead"],
            "calibrated_component": "vanilla decode throughput from calibration prompts only",
            "feasibility_definition": "faster iff per-prompt paired decode TPS ratio > 1.0",
        },
    }
    if "baseline_tps_decode_by_context" in baseline:
        result["calibrated_baseline_tps_by_context"] = baselines
        result["heldout_by_context"] = {}
        for context, value in baselines.items():
            context_rows = [r for r in held if str(r.get("context_length")) == context]
            result["heldout_by_context"][context] = summarize(context_rows, "heldout", value)
    (run_dir / "prediction_errors.json").write_text(json.dumps(result, indent=2))
    make_figures(held, baseline_tps, run_dir / "figures")
    print(json.dumps(result["heldout"], indent=2))
    print(f"Figures and prediction errors written to {run_dir}")


if __name__ == "__main__":
    main()
