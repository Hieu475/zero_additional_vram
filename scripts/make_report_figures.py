#!/usr/bin/env python3
"""Build the technical-report figures from recorded evidence JSON (writes PDFs into bao_cao/images)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = Path("experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles")
OUT = Path("bao_cao/images")
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_methods():
    boot = json.loads((N / "heldout2_gsm8k_400_449_bootstrap_v1.json").read_text())
    lazy, comp = boot["lazy_run"], boot["vs_compiled"]
    rows = [("B2 (LM head đầy đủ)", lazy["b2_speedup"], f"{lazy['tokens_match_bnb']}/{lazy['prompts']}", AQUA),
            ("B2 lười", lazy["lazy_speedup"], f"{lazy['tokens_match_bnb']}/{lazy['prompts']}", BLUE),
            ("BNB biên dịch\n(không exact)", comp["compiled_over_eager"], f"{comp['compiled_equals_eager']}/{comp['prompts']}", ORANGE)]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    style(ax)
    for i, (name, (mid, lo, hi), exact, color) in enumerate(rows):
        ax.errorbar(i, mid, yerr=[[mid - lo], [hi - mid]], fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4, markersize=8,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.text(i + 0.12, mid, f"{mid:.3f}x", va="center", color=INK, fontsize=9)
        ax.text(i, 0.985, f"khớp BNB: {exact}", ha="center", va="top", color=MUTED, fontsize=8, transform=ax.get_xaxis_transform())
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(2.45, 1.003, "BNB eager = 1", ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.98, 1.36)
    ax.set_ylabel("Tốc độ so với BNB eager")
    fig.tight_layout()
    fig.savefig(OUT / "tr_methods_heldout2.pdf")
    plt.close(fig)


def figure_context():
    lengths, b2, lazy = [], [], []
    for tag in ("0512", "1024", "2048", "3072"):
        s = json.loads((N / f"long_context_{tag}_lazy_v2.json").read_text())["summary"]
        lengths.append(int(tag)); b2.append(s["geomean_b2_speedup"]); lazy.append(s["geomean_lazy_speedup"])
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    style(ax)
    ax.plot(lengths, lazy, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    ax.plot(lengths, b2, color=AQUA, linewidth=2, marker="s", markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(lengths[-1] + 60, lazy[-1], "B2 lười", color=INK, va="center")
    ax.text(lengths[-1] + 60, b2[-1] - 0.02, "B2", color=INK, va="center")
    ax.set_xticks(lengths)
    ax.set_xlim(380, 3550)
    ax.set_ylim(0.96, 1.2)
    ax.set_xlabel("Độ dài prompt (token)")
    ax.set_ylabel("Tốc độ so với BNB eager")
    fig.tight_layout()
    fig.savefig(OUT / "tr_context_length.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_methods()
    figure_context()
    print("wrote", sorted(p.name for p in OUT.glob("tr_*.pdf")))
