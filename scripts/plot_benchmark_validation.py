#!/usr/bin/env python3
"""Create paired-speedup, by-task, control, and exactness figures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "vanilla_via_speculative_core": "Vanilla via core",
    "pld_forced_miss": "PLD forced miss",
    "pld": "PLD",
    "pld_gate": "PLD gate",
    "routed": "Routed",
}


def bootstrap_ci(values: list[float], seed: int = 2026) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(array, size=(4000, len(array)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))


def prompt_ratios(rows: list[dict], method: str, metric: str = "end_to_end_tps") -> dict:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["benchmark"], row["id"], row["repeat"], row["order_seed"])
        grouped.setdefault(key, {})[row["method"]] = row
    by_prompt: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, methods in grouped.items():
        if method not in methods or "vanilla" not in methods:
            continue
        baseline = methods["vanilla"][metric]
        if baseline and np.isfinite(baseline):
            by_prompt[key[:2]].append(methods[method][metric] / baseline)
    return {key: float(np.mean(values)) for key, values in by_prompt.items()}


def save_figures(run_dir: Path) -> None:
    raw_rows = [json.loads(line) for line in (run_dir / "raw_measurements.jsonl").read_text().splitlines()
                if line.strip()]
    summary = json.loads((run_dir / "summary.json").read_text())
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})

    methods = list(METHOD_LABELS)
    labels = [METHOD_LABELS[m] for m in methods]
    ratios = [list(prompt_ratios(raw_rows, method).values()) for method in methods]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(ratios, tick_labels=labels, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#cfe3f1", "edgecolor": "#2b5c7a"},
               medianprops={"color": "#bb4d00", "linewidth": 1.7})
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_ylabel("Paired end-to-end speedup vs vanilla")
    ax.set_title("Prompt-level speedup distribution (N=150)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figures / "paired_speedup.pdf", bbox_inches="tight")
    plt.close(fig)

    tasks = list(summary["by_benchmark"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(tasks))
    width = 0.15
    for index, method in enumerate(methods):
        means, lows, highs = [], [], []
        for task in tasks:
            task_rows = [row for row in raw_rows if row["benchmark"] == task]
            values = list(prompt_ratios(task_rows, method).values())
            mean = float(np.mean(values)) if values else float("nan")
            lo, hi = bootstrap_ci(values, 2026 + index)
            means.append(mean)
            lows.append(mean - lo)
            highs.append(hi - mean)
        ax.bar(x + (index - (len(methods) - 1) / 2) * width, means, width,
               yerr=[lows, highs], capsize=2, label=METHOD_LABELS[method])
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xticks(x, tasks)
    ax.set_ylabel("Paired end-to-end speedup vs vanilla")
    ax.set_title("Speedup by task (prompt bootstrap 95% CI)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "speedup_by_task.pdf", bbox_inches="tight")
    plt.close(fig)

    controls = ["vanilla_via_speculative_core", "pld_forced_miss"]
    control_values = [list(prompt_ratios(raw_rows, method).values()) for method in controls]
    means = [float(np.mean(values)) for values in control_values]
    intervals = [bootstrap_ci(values) for values in control_values]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    positions = np.arange(len(controls))
    ax.errorbar(means, positions,
                xerr=[[mean - ci[0] for mean, ci in zip(means, intervals)],
                      [ci[1] - mean for mean, ci in zip(means, intervals)]],
                fmt="o", color="#2b5c7a", capsize=4)
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(positions, [METHOD_LABELS[m] for m in controls])
    ax.set_xlabel("Paired end-to-end speed ratio vs vanilla")
    ax.set_title("Negative-control overhead (prompt bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(figures / "control_overhead.pdf", bbox_inches="tight")
    plt.close(fig)

    exact = [summary["overall"][method]["exact_match_pct"] for method in methods]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(np.arange(len(methods)), exact, color="#8bb7ce")
    ax.set_xticks(np.arange(len(methods)), labels, rotation=15)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Sequence exact match (%)")
    ax.set_title("Exactness by method against direct vanilla")
    for i, value in enumerate(exact):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "exactness_by_method.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    save_figures(Path(args.run_dir))
    print(f"Saved figures under {Path(args.run_dir) / 'figures'}")


if __name__ == "__main__":
    main()
