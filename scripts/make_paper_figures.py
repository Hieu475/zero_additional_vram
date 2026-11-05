#!/usr/bin/env python3
"""Figures for the direction-C paper draft, built only from recorded result JSON (writes PDF+PNG+CSV into paper/figures_c/).

Fig 1: share of prompts whose 128-token continuation differs from sequential greedy, naive PLD vs audit-derived row-wise list.
Fig 2: share of 4-bit projection output elements that differ between a block of M rows and row-by-row execution.
Palette: blue/orange/aqua (validated with the dataviz validator; aqua is < 3:1 on the light surface so every series is
direct-labelled and a CSV table of the plotted values is written next to each figure).
"""
from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles")
OUT = Path("paper/figures_c")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "text.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def tally(names):
    k = n = 0
    for name in names:
        d = json.loads((P / name).read_text())
        k += d["diverged"]; n += d["prompts"]
    return k, n


def one(pattern):
    hits = sorted(glob.glob(str(P / pattern)))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0]).name


CONFIGS = [  # label, key, kind
    ("Qwen2.5-0.5B (bnb)", "qwen05b"), ("Qwen2.5-1.5B (bnb)", "qwen15b"), ("Qwen2.5-3B (bnb)", "qwen3b"),
    ("OLMo-2-1B (bnb)", "olmo2_1b"), ("Phi-3.5-mini (bnb)", "phi35"), ("Qwen2.5-3B (torchao)", "torchao_qwen3b"),
]


def files_for(dataset, key, planned):
    """Result files (naive or audit-list) for a config on a dataset; GQA-fixed variants where required."""
    tq = key == "torchao_qwen3b"
    if dataset == "GSM8K (50)":
        if tq:
            return [one("c8_torchao_qwen3b_k2_lm_head+rmsnorm+sdpa_gsm8k50_v1.json" if planned else "c8_torchao_qwen3b_k2_none_gsm8k50_v1.json")]
        if not planned:
            return [f"c6_{key}_k2_gsm8k20_v1.json", f"c6x_{key}_k2_gsm8k20to99_v1.json"]
        return [one(f"fix_{key}_k2_rmsnorm+sdpa+lm_head_gsm8k50_v1.json")]
    if dataset == "HumanEval (50)":
        if tq:
            return None
        return [one(f"heldout_humaneval_{key}_k2_none_v1.json")] if not planned else [one(f"heldout_humaneval_{key}_k2_*[!e]_v1.json")]
    if dataset == "CNN/DM (50)":
        if tq:
            return [one("heldout_cnndm_torchao_qwen3b_k2_lm_head+rmsnorm+sdpa_gqafix_v2.json" if planned else "heldout_cnndm_torchao_qwen3b_k2_none_gqafix_v2.json")]
        if not planned:
            return [one(f"heldout_cnndm_{key}_k2_none_v1.json")]
        g = glob.glob(str(P / f"heldout_cnndm_{key}_k2_*_gqafix_v2.json"))
        return [Path(g[0]).name] if g else [one(f"heldout_cnndm_{key}_k2_lm_head+rmsnorm+sdpa_v1.json")]
    if dataset == "CNN/DM, independent (60-120)":
        if not planned:
            names = [one(f"ext150_cnndm_{key}_k2_none_n60_v1.json")]
            if key == "qwen3b":
                names.append(one("ext150b_cnndm_qwen3b_k2_none_v1.json"))
            return names
        names = [Path(sorted(glob.glob(str(P / f"ext150_cnndm_{key}_k2_*_n60_v1.json")), key=lambda s: "none" in s)[0]).name]
        if key in ("qwen3b", "torchao_qwen3b"):
            names.append(one("ext150b_cnndm_qwen3b_k2_lm_head+rmsnorm+sdpa_v1.json" if key == "qwen3b" else "ext150b_cnndm_torchao_qwen3b_k2_lm_head+rmsnorm+sdpa_v1.json"))
        return names
    raise KeyError(dataset)


def fig1():
    datasets = ["GSM8K (50)", "HumanEval (50)", "CNN/DM (50)", "CNN/DM, independent (60-120)"]
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.9), sharey=True)
    for ax, ds in zip(axes, datasets):
        for i, (label, key) in enumerate(CONFIGS):
            y = len(CONFIGS) - 1 - i
            for planned, color, dy in ((False, ORANGE, 0.17), (True, BLUE, -0.17)):
                names = files_for(ds, key, planned)
                if names is None:
                    continue
                k, n = tally(names)
                lo, hi = wilson(k, n)
                ax.plot([100 * lo, 100 * hi], [y + dy] * 2, color=color, linewidth=2, solid_capstyle="round", zorder=2)
                ax.plot([100 * k / n], [y + dy], "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
                ax.text(100 * hi + 2, y + dy, f"{k}/{n}", va="center", fontsize=7.5, color=MUTED)
                rows.append([ds, label, "audit list row-wise" if planned else "naive PLD", k, n, round(100 * k / n, 1), round(100 * lo, 1), round(100 * hi, 1)])
        ax.set_title(ds, fontsize=9.5, loc="left", color=INK)
        ax.set_xlim(-2, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if ds == datasets[0]:
            ax.set_yticks(range(len(CONFIGS)))
            ax.set_yticklabels([c[0] for c in CONFIGS][::-1])
        ax.set_ylim(-0.6, len(CONFIGS) - 0.4)
    fig.supxlabel("Prompts whose 128-token greedy continuation differs from sequential decoding (%, 95% Wilson interval)", fontsize=9, y=0.01)
    fig.legend(handles=[plt.Line2D([], [], marker="o", color=ORANGE, linewidth=2, markersize=7, label="Naive block verification (K=2)"),
                        plt.Line2D([], [], marker="o", color=BLUE, linewidth=2, markersize=7, label="Audit-derived operator list evaluated row by row")],
               loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig, rows, ["dataset", "config", "method", "diverged", "prompts", "pct", "ci_low_pct", "ci_high_pct"]


def proj_rate(name):
    d = json.loads((P / name).read_text())
    lin = [v for k, v in d["per_op_class"].items() if k.startswith("linear")]
    return 100 * sum(v["mismatches"] for v in lin) / sum(v["elements"] for v in lin)


def fig2():
    series = [
        ("bitsandbytes 0.50.0", BLUE, {2: "c2_m2_gsm8k20_v1.json", 3: "op_shape_sensitivity_gsm8k20_v1.json", 4: "c2_m4_gsm8k20_v1.json",
                                       5: "c2_m5_gsm8k20_v1.json", 8: "c2_m8_gsm8k20_v1.json", 16: "c2_m16_gsm8k20_v1.json"}),
        ("bitsandbytes 0.49.2", ORANGE, {3: "c10_bnb0492_qwen3b_m3_gsm8k20_v1.json", 4: "c10_bnb0492_qwen3b_m4_gsm8k20_v1.json", 8: "c10_bnb0492_qwen3b_m8_gsm8k20_v1.json"}),
        ("torchao INT4", AQUA, {3: "c8_torchao_qwen3b_m3_gsm8k20_v1.json", 4: "c8_torchao_qwen3b_m4_gsm8k20_v1.json"}),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    rows = []
    offsets = {"bitsandbytes 0.50.0": 0.0, "bitsandbytes 0.49.2": 0.0, "torchao INT4": 0.0}
    for label, color, pts in series:
        xs = sorted(pts)
        ys = [proj_rate(pts[m]) for m in xs]
        ax.plot(xs, ys, color=color, linewidth=2, zorder=2)
        ax.plot(xs, ys, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, clip_on=False)
        rows += [[label, m, round(y, 3)] for m, y in zip(xs, ys)]
        dy = {"bitsandbytes 0.50.0": 2.6, "bitsandbytes 0.49.2": 2.6, "torchao INT4": 2.8}[label]
        ax.text(xs[-1] + (0.4 if label != "torchao INT4" else 0.3), ys[-1] + dy if label != "torchao INT4" else ys[-1] + dy, label,
                color=INK, fontsize=8.5, va="center", ha="left" if label != "bitsandbytes 0.50.0" else "left")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 4, 5, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(1.9, 34)
    ax.set_ylim(-1.5, 40)
    ax.set_xlabel("Rows per block, M (Qwen2.5-3B)")
    ax.set_ylabel("Projection output elements differing\nfrom row-by-row execution (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=c, linewidth=2, markersize=7, label=l) for l, c, _ in series],
              loc="center right", frameon=False, fontsize=8, bbox_to_anchor=(1.0, 0.56))
    fig.tight_layout()
    return fig, rows, ["backend", "M", "projection_mismatch_pct"]


def save(fig, rows, header, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    with open(OUT / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    plt.close(fig)
    print("wrote", OUT / f"{stem}.pdf", len(rows), "rows")


if __name__ == "__main__":
    save(*fig1(), "fig1_trajectory_divergence")
    save(*fig2(), "fig2_projection_vs_block_size")
