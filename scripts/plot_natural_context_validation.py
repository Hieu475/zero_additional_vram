#!/usr/bin/env python3
"""Plot paired speedup and sequence identity from natural-context summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    buckets = ["128-512", "512-1024", "1024-2048", "2048-4096"]
    methods = sorted({method for bucket in summary["by_context_bucket"].values()
                      for method in bucket})
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = list(range(len(buckets)))
    for index, method in enumerate(methods):
        color = colors[index % len(colors)]
        means, lower, upper, exactness = [], [], [], []
        for bucket in buckets:
            row = summary["by_context_bucket"].get(bucket, {}).get(method)
            if row is None:
                means.append(float("nan")); lower.append(0); upper.append(0)
                exactness.append(float("nan")); continue
            lo, hi = row["bootstrap_prompt_mean_95ci"]
            means.append(row["mean_speedup"])
            lower.append(row["mean_speedup"] - lo)
            upper.append(hi - row["mean_speedup"])
            exactness.append(row["sequence_exact_match_rate"])
        axes[0].errorbar(x, means, yerr=[lower, upper], marker="o", capsize=3,
                         linewidth=1.6, color=color, label=method)
        axes[1].plot(x, exactness, marker="o", linewidth=1.6, color=color, label=method)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Decode throughput speedup vs vanilla")
    axes[0].set_ylim(0.45, 1.35)
    axes[0].set_title("Paired speedup (prompt bootstrap 95% CI)")
    axes[1].set_ylabel("Sequence exact-match rate")
    axes[1].set_ylim(0.5, 1.02)
    axes[1].set_title("Greedy sequence identity")
    for ax in axes:
        ax.set_xticks(x, buckets, rotation=15)
        ax.set_xlabel("Input context tokens")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"Natural-document context pilot (N={summary['n_prompts']} prompts)", y=1.03)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
