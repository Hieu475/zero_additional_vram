#!/usr/bin/env python3
"""Fit on train, select on dev, then report predictor and decision quality."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zassd.profiling.acceptance_survival import (  # noqa: E402
    AcceptanceSurvivalPredictor, context_bin, expected_emitted, survival_labels,
)
from zassd.profiling.action_cost_model import MeasuredActionCostModel  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pid(row: dict[str, Any]) -> str:
    return f"{row['task']}:{row['prompt_id']}"


def valid_tpc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("verify_complete") and int(r.get("prediction_horizon", 0)) > 0]


def prompt_bootstrap(values_by_prompt: dict[str, list[float]], seed: int = 913) -> list[float] | None:
    keys = sorted(values_by_prompt)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [float(np.mean(values_by_prompt[k])) for k in keys]
    draws = [float(np.mean(rng.choice(means, size=len(means), replace=True))) for _ in range(5000)]
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def confusion(labels: list[bool], guesses: list[bool]) -> dict[str, float | int]:
    tp = sum(a and p for a, p in zip(labels, guesses))
    fp = sum((not a) and p for a, p in zip(labels, guesses))
    fn = sum(a and (not p) for a, p in zip(labels, guesses))
    tn = sum((not a) and (not p) for a, p in zip(labels, guesses))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision,
            "recall": recall, "false_positive_rate": fp / max(1, fp + tn),
            "balanced_accuracy": (recall + specificity) / 2}


def predictions(rows: list[dict[str, Any]], predictor: AcceptanceSurvivalPredictor,
                online: bool = False) -> list[list[float]]:
    result: list[list[float] | None] = [None] * len(rows)
    if not online:
        return [predictor.predict(row) for row in rows]
    indexed: dict[tuple[str, str, str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for i, row in enumerate(rows):
        indexed[(row["task"], str(row["prompt_id"]), row["action"], int(row["k"]))].append((i, row))
    for group in indexed.values():
        ema = dict(predictor.global_rates)
        for i, row in sorted(group, key=lambda item: int(item[1]["cycle_index"])):
            result[i] = predictor.predict_online(row, ema)
            predictor.update_online(row, ema)
    return [p or [] for p in result]


def tpc_metrics(rows: list[dict[str, Any]], preds: list[list[float]]) -> dict[str, Any]:
    per_prompt: dict[str, list[float]] = defaultdict(list)
    complete_per_prompt: dict[str, list[float]] = defaultdict(list)
    for row, pred in zip(rows, preds):
        if not row.get("verify_complete") or not pred:
            continue
        err = abs(expected_emitted(pred) - float(row["emitted_tokens"]))
        per_prompt[pid(row)].append(err)
        if row.get("draft_complete") and not row.get("truncated_by_eos") and not row.get("truncated_by_budget"):
            complete_per_prompt[pid(row)].append(err)
    def summarize(groups: dict[str, list[float]]) -> dict[str, Any]:
        values = [x for group in groups.values() for x in group]
        return {"n_prompts": len(groups), "n_cycles": len(values),
                "tokens_per_cycle_mae": float(np.mean(values)) if values else None,
                "prompt_cluster_bootstrap_95ci": prompt_bootstrap(groups)}
    by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row, pred in zip(rows, preds):
        if not row.get("verify_complete") or not pred:
            continue
        group_names = {
            "action_k": f"{row['action']}_K{row['k']}",
            "context": context_bin(int(row.get("context_length", 0))),
        }
        error = abs(expected_emitted(pred) - float(row["emitted_tokens"]))
        for dimension, group_name in group_names.items():
            by_group[dimension][group_name].append(error)
    return {"all_verified_cycles": summarize(per_prompt),
            "complete_uncensored_cycles": summarize(complete_per_prompt),
            "tpc_mae_by_action_k_context": {
                dimension: {name: float(np.mean(errors)) for name, errors in sorted(groups.items())}
                for dimension, groups in by_group.items()
            }}


def survival_metrics(rows: list[dict[str, Any]], preds: list[list[float]]) -> tuple[dict[str, Any], list[dict[str, float]]]:
    records = []
    grouped_brier: dict[str, list[float]] = defaultdict(list)
    grouped_abs: dict[str, list[float]] = defaultdict(list)
    for row, pred in zip(rows, preds):
        labels = survival_labels(row)
        for p, actual in zip(pred, labels):
            error = float(p) - float(actual)
            records.append({"predicted": float(p), "observed": float(actual)})
            grouped_brier[pid(row)].append(error * error)
            grouped_abs[pid(row)].append(abs(error))
    if not records:
        return {"n_token_survival_observations": 0}, []
    bins = []
    ece = 0.0
    for idx in range(10):
        lo, hi = idx / 10, (idx + 1) / 10
        selected = [r for r in records if lo <= r["predicted"] < hi or (idx == 9 and r["predicted"] == 1)]
        if selected:
            mean_p = float(np.mean([r["predicted"] for r in selected]))
            mean_y = float(np.mean([r["observed"] for r in selected]))
            ece += len(selected) / len(records) * abs(mean_p - mean_y)
            bins.append({"bin_lower": lo, "bin_upper": hi, "count": len(selected),
                         "mean_predicted": mean_p, "observed_rate": mean_y})
    brier_prompt = {p: [float(np.mean(values))] for p, values in grouped_brier.items()}
    mae_prompt = {p: [float(np.mean(values))] for p, values in grouped_abs.items()}
    return {
        "n_token_survival_observations": len(records),
        "survival_brier": float(np.mean([v for values in grouped_brier.values() for v in values])),
        "survival_brier_prompt_bootstrap_95ci": prompt_bootstrap(brier_prompt),
        "survival_probability_mae": float(np.mean([v for values in grouped_abs.values() for v in values])),
        "survival_ece_10_bins": ece,
        "reliability_bins": bins,
    }, records


def feasibility(rows: list[dict[str, Any]], raw: list[dict[str, Any]], preds: list[list[float]],
                baseline_tps: float, threshold: float) -> dict[str, Any]:
    measured = {(r["task"], str(r["prompt_id"]), r.get("action"), int(r.get("k", 0))): r
                for r in raw if r.get("method") == "layer_skip"}
    vanilla = {(r["task"], str(r["prompt_id"])): r for r in raw if r.get("method") == "vanilla"}
    cycle_groups: dict[tuple[str, str, str, int], list[tuple[dict[str, Any], list[float]]]] = defaultdict(list)
    for row, pred in zip(rows, preds):
        if row.get("verify_complete") and pred:
            cycle_groups[(row["task"], str(row["prompt_id"]), row["action"], int(row["k"]))].append((row, pred))
    labels_actual, guesses_oracle, guesses_runtime = [], [], []
    for (task, prompt_id, action, k), group in cycle_groups.items():
        action_row = measured.get((task, prompt_id, action, k))
        vanilla_row = vanilla.get((task, prompt_id))
        if not action_row or not vanilla_row:
            continue
        actual_speedup = float(action_row["measured_speedup_vs_prompt_vanilla"])
        pred_tpc = float(np.mean([expected_emitted(p) for _, p in group]))
        cycles = [r for r, _ in group]
        measured_cycle_ms = float(np.mean([
            float(r["draft_ms"]) + float(r["verify_ms"]) + float(r["cache_ms"]) for r in cycles
        ]))
        measured_vanilla_tps = float(vanilla_row["measured"]["decode_tps"])
        oracle_timing_speedup = (pred_tpc * 1000 / max(measured_cycle_ms, 1e-9)) / max(measured_vanilla_tps, 1e-9)
        cost_prediction = MeasuredActionCostModel.for_model(
            "qwen25_3b", baseline_tps=baseline_tps).evaluate_action(action, k)
        runtime_speedup = (pred_tpc * 1000 / max(cost_prediction.total_cycle_ms, 1e-9)) / max(baseline_tps, 1e-9)
        labels_actual.append(actual_speedup > threshold)
        guesses_oracle.append(oracle_timing_speedup > threshold)
        guesses_runtime.append(runtime_speedup > threshold)
    return {"measured_latency_diagnostic_only": confusion(labels_actual, guesses_oracle),
            "frozen_cost_model_runtime_estimate": confusion(labels_actual, guesses_runtime),
            "n_prompt_action_labels": len(labels_actual),
            "actual_profitable_cases": sum(labels_actual)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()
    root = Path(args.run_dir)
    cycle_dir = root / "acceptance_cycles"
    split_rows = {split: read_jsonl(cycle_dir / f"{split}.jsonl")
                  for split in ("train", "development", "heldout_in_domain", "heldout_cross_task")}
    raw_rows = {split: read_jsonl(root / split / "raw_measurements.jsonl")
                for split in split_rows}
    train = valid_tpc_rows(split_rows["train"])
    dev = valid_tpc_rows(split_rows["development"])
    if not train or not dev:
        raise ValueError("Training and development cycles are required")
    dev_scores = {}
    for mode in AcceptanceSurvivalPredictor.MODES:
        predictor = AcceptanceSurvivalPredictor(mode).fit(train)
        dev_scores[mode] = tpc_metrics(dev, predictions(dev, predictor))["complete_uncensored_cycles"]["tokens_per_cycle_mae"]
    selected = min(dev_scores, key=lambda mode: float("inf") if dev_scores[mode] is None else dev_scores[mode])
    fit_rows = train + dev
    predictor = AcceptanceSurvivalPredictor(selected).fit(fit_rows)
    all_metrics = {}
    reliability_data = {}
    split_predictions = {}
    for split in ("heldout_in_domain", "heldout_cross_task"):
        rows = split_rows[split]
        scored_rows = valid_tpc_rows(rows)
        model_results = {}
        model_preds = {}
        for mode in (*AcceptanceSurvivalPredictor.MODES, "online_ema"):
            model = predictor if mode == selected or mode == "online_ema" else AcceptanceSurvivalPredictor(mode).fit(fit_rows)
            ps = predictions(scored_rows, model, online=(mode == "online_ema"))
            model_preds[mode] = ps
            model_results[mode] = {
                "tpc": tpc_metrics(scored_rows, ps),
                "survival": survival_metrics(scored_rows, ps)[0],
            }
        split_predictions[split] = model_preds
        reliability_data[split] = survival_metrics(scored_rows, model_preds[selected])[0]["reliability_bins"]
        baseline_tps = json.loads((root / "train/baseline.json").read_text())["baseline_decode_tps"]
        model_results[selected]["decision_quality_by_threshold"] = {
            f"{threshold:.2f}": feasibility(
                scored_rows, raw_rows[split], model_preds[selected], baseline_tps, threshold)
            for threshold in (1.0, 1.03, 1.05)
        }
        all_metrics[split] = {
            "n_prompts": len({pid(r) for r in rows}), "n_cycles": len(rows),
            "selected_predictor": selected, "predictors": model_results,
        }
    result = {
        "prompt_counts": json.loads((root / "config.json").read_text())["prompt_counts"],
        "train_cycles": len(split_rows["train"]), "development_cycles": len(split_rows["development"]),
        "development_complete_uncensored_tpc_mae": dev_scores,
        "selected_predictor": selected, "heldout": all_metrics,
        "limitations": [
            "Predictor modes and thresholds are selected on development prompts only; final empirical rates use train+development.",
            "Held-out decision metrics include measured-latency diagnostic and frozen-cost-model runtime estimates separately.",
            "Cycle-level observations are clustered by prompt for confidence intervals; prompt snapshots limit the attainable sample sizes.",
            "The measured-latency classifier is an offline diagnostic, not deployable policy information.",
        ],
    }
    out = root / "evaluation.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    plot_path = Path(args.plot) if args.plot else root / "figures/reliability.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for split, bins in reliability_data.items():
        ax.plot([b["mean_predicted"] for b in bins], [b["observed_rate"] for b in bins],
                marker="o", label=split.replace("heldout_", ""))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted survival probability",
           ylabel="Observed acceptance rate", title=f"Acceptance survival: {selected}")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plot_path, dpi=180); plt.close(fig)
    result["reliability_plot"] = str(plot_path)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
