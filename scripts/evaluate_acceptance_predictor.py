#!/usr/bin/env python3
"""Select an acceptance-survival predictor on calibration dev prompts and test held-out.

Feasibility numbers use measured cycle latency and prompt-matched vanilla TPS to
isolate acceptance-model quality. They are diagnostic and are not runtime oracle data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from zassd.profiling.acceptance_survival import (
    AcceptanceSurvivalPredictor,
    expected_emitted,
    survival_labels,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_key(row: dict[str, Any]) -> str:
    return f"{row['task']}:{row['prompt_id']}"


def dev_prompt(key: str) -> bool:
    value = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return value % 5 == 0


def tpc_mae(rows: list[dict[str, Any]], predictions: list[list[float]]) -> float:
    return float(np.mean([abs(expected_emitted(p) - (1 + int(r["accepted_tokens"])))
                          for r, p in zip(rows, predictions)])) if rows else float("inf")


def survival_errors(rows: list[dict[str, Any]], predictions: list[list[float]]) -> dict[str, float]:
    errors = [probability - observed
              for row, pred in zip(rows, predictions)
              for probability, observed in zip(pred, survival_labels(row))]
    return {
        "survival_probability_mae": float(np.mean(np.abs(errors))) if errors else float("nan"),
        "survival_probability_brier": float(np.mean(np.square(errors))) if errors else float("nan"),
    }


def bootstrap_prompt_ci(rows: list[dict[str, Any]], errors: list[float], seed: int = 17) -> list[float] | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, errors):
        grouped[prompt_key(row)].append(error)
    keys = list(grouped)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(3000):
        selected = rng.choice(keys, size=len(keys), replace=True)
        draws.append(float(np.mean([e for key in selected for e in grouped[str(key)]])))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def confusion(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(actual & predicted)); fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted)); tn = int(np.sum(~actual & ~predicted))
    recall = tp / max(1, tp + fn); specificity = tn / max(1, tn + fp)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / max(1, tp + fp), "recall": recall,
            "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    calibration = read_jsonl(root / "acceptance_cycles/calibration.jsonl")
    heldout = read_jsonl(root / "acceptance_cycles/heldout.jsonl")
    raw_heldout = read_jsonl(root / "heldout/raw_measurements.jsonl")
    vanilla = {(r["task"], str(r["prompt_id"])): r for r in raw_heldout if r.get("method") == "vanilla"}
    action_rows = {(r["task"], str(r["prompt_id"]), r["action"], int(r["k"])): r
                   for r in raw_heldout if r.get("method") == "layer_skip"}
    train = [r for r in calibration if not dev_prompt(prompt_key(r))]
    dev = [r for r in calibration if dev_prompt(prompt_key(r))]
    if not train or not dev or not heldout:
        raise ValueError(f"Need calibration train/dev and heldout cycles (got {len(train)}/{len(dev)}/{len(heldout)})")

    dev_scores = {}
    for mode in AcceptanceSurvivalPredictor.MODES:
        predictor = AcceptanceSurvivalPredictor(mode).fit(train)
        dev_scores[mode] = tpc_mae(dev, [predictor.predict(r) for r in dev])
    selected = min(dev_scores, key=dev_scores.get)
    predictor = AcceptanceSurvivalPredictor(selected).fit(calibration)

    heldout = sorted(heldout, key=lambda r: (r["task"], str(r["prompt_id"]), r["action"], int(r["k"]), int(r["cycle_index"])))
    by_prompt: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in heldout:
        by_prompt[(row["task"], str(row["prompt_id"]), row["action"], int(row["k"]))].append(row)
    predictions: dict[str, list[list[float]]] = {m: [] for m in AcceptanceSurvivalPredictor.MODES}
    observed_tpc: list[float] = []
    online_pred: list[list[float]] = []
    models = {mode: (predictor if mode == selected else AcceptanceSurvivalPredictor(mode).fit(calibration))
              for mode in AcceptanceSurvivalPredictor.MODES}
    for _key, group in by_prompt.items():
        ema: dict[int, float] = dict(predictor.global_rates)
        for row in group:
            for mode in AcceptanceSurvivalPredictor.MODES:
                predictions[mode].append(models[mode].predict(row))
            online = predictor.predict_online(row, ema)
            online_pred.append(online)
            predictor.update_online(row, ema)
            observed_tpc.append(1 + int(row["accepted_tokens"]))

    errors_by_model = {}
    for mode, ps in predictions.items():
        errors = [abs(expected_emitted(p) - actual) for p, actual in zip(ps, observed_tpc)]
        errors_by_model[mode] = {"tokens_per_cycle_mae": float(np.mean(errors)),
                                 **survival_errors(heldout, ps),
                                 "ci95_prompt_bootstrap": bootstrap_prompt_ci(heldout, errors)}
    online_errors = [abs(expected_emitted(p) - actual) for p, actual in zip(online_pred, observed_tpc)]
    errors_by_model["online_ema"] = {"tokens_per_cycle_mae": float(np.mean(online_errors)),
                                      **survival_errors(heldout, online_pred),
                                      "ci95_prompt_bootstrap": bootstrap_prompt_ci(heldout, online_errors)}

    chosen = predictions[selected]
    aggregate: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row, pred in zip(heldout, chosen):
        aggregate[(row["task"], str(row["prompt_id"]), row["action"], int(row["k"]))].append(
            {"row": row, "tpc": expected_emitted(pred)})
    feasibility: dict[str, Any] = {}
    for threshold in (1.0, 1.03, 1.05):
        labels, guesses = [], []
        for (task, pid, action, k), cycles in aggregate.items():
            ar = action_rows.get((task, pid, action, k)); vr = vanilla.get((task, pid))
            if not ar or not vr or not vr.get("decode_tps"):
                continue
            mean_tpc = float(np.mean([c["tpc"] for c in cycles]))
            mean_ms = float(ar["measured"].get("decode_ms_per_cycle", 0))
            predicted_speedup = (mean_tpc * 1000.0 / max(mean_ms, 1e-9)) / float(vr["decode_tps"])
            actual_speedup = float(ar["measured_speedup_vs_prompt_vanilla"])
            labels.append(actual_speedup > threshold)
            guesses.append(predicted_speedup > threshold)
        feasibility[str(threshold)] = confusion(np.asarray(labels, dtype=bool), np.asarray(guesses, dtype=bool))

    result = {
        "calibration_prompts": len({prompt_key(r) for r in calibration}),
        "calibration_train_prompts": len({prompt_key(r) for r in train}),
        "calibration_dev_prompts": len({prompt_key(r) for r in dev}),
        "heldout_prompts": len({prompt_key(r) for r in heldout}),
        "n_heldout_cycles": len(heldout),
        "development_tokens_per_cycle_mae": dev_scores,
        "selected_predictor": selected,
        "heldout_predictor_errors": errors_by_model,
        "feasibility_using_measured_latency_diagnostic_only": feasibility,
        "limitations": [
            "Held-out feasibility isolates acceptance quality by using measured cycle latency and measured prompt-matched vanilla TPS.",
            "Expected emitted count is 1 + sum survival probabilities; EOS and output-budget truncation can reduce it.",
            "Online EMA updates only after each held-out cycle and resets for each prompt.",
        ],
    }
    (root / "acceptance_prediction_errors.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
