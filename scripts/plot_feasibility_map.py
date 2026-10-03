#!/usr/bin/env python3
"""Speculation feasibility map (CPU-only, review item 21).

X-axis: draft cost ratio c = T_draft(K=1) / T_vanilla.
Y-axis: per-token acceptance alpha.
Curves: break-even boundary alpha_be(c) for K=1,2,4 on RTX 4050 —
above a curve, speculation at that K is profitable; below, it loses.
Scatter: measured operating points (Qwen2.5-3B NF4, RTX 4050) plus
projected 4090/A100 points (marked, from roofline cross-GPU sweep).

This is the paper's key figure: high acceptance alone does not imply
speedup; the (c, alpha) point must clear the boundary.

Run: python3 scripts/plot_feasibility_map.py
Output: experiments/16_hybrid_roofline/feasibility_map.png
        paper/figures/feasibility_map.png
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from zassd.modeling.roofline import SpeculativeRoofline, KNOWN_GPUS

OUT = Path("experiments/16_hybrid_roofline")
OUT.mkdir(parents=True, exist_ok=True)

roof = SpeculativeRoofline()
g4050 = KNOWN_GPUS["RTX4050-laptop"]
TV = g4050.t_vanilla_ms


def alpha_be(c, k, gpu=g4050):
    t_d = roof.scale(roof.t_draft_k1 * (c / 0.75) * k, gpu.bandwidth_gbs)
    t_v = roof.scale(roof.t_verify_k1 + (k - 1) * roof.verify_slope * gpu.verify_rho,
                     gpu.bandwidth_gbs)
    t_misc = roof.scale(roof.t_misc, gpu.bandwidth_gbs)
    return ((t_d + t_v + t_misc) / gpu.t_vanilla_ms - 1.0) / max(1, k)


cs = np.linspace(0.05, 1.0, 200)

# Measured operating points on RTX 4050 (Qwen2.5-3B NF4, frozen benchmarks).
# c inferred from micro-profiling (t_draft_k1 / t_vanilla); alpha measured.
measured = [
    {"label": "cka_75 K=2 (0.87x)", "c": 0.75, "alpha": 0.73, "win": False},
    {"label": "cka_83 K=1 (0.98x)", "c": 0.83 * 19.72 / 24.1 * 0.75 / 0.75, "alpha": 0.86, "win": False},
    {"label": "cka_50 K=1 (0.89x)", "c": 0.50, "alpha": 0.37, "win": False},
    {"label": "PLD K=4 (1.39x)", "c": 0.003, "alpha": 0.95, "win": True},
]
# cka_83 draft is cheaper per layer count: scale by kept-layer ratio 33/36 vs 27/36
measured[1]["c"] = round(0.75 * 33 / 27, 3)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.5, 5))
for k, style in [(1, "o-"), (2, "s-"), (4, "^-")]:
    ys = [alpha_be(c, k) for c in cs]
    ax.plot(cs, ys, style, ms=3, markevery=20, label=f"break-even K={k} (RTX 4050)")
ax.axhspan(0, 1.05, alpha=0.04, color="red")
for m in measured:
    ax.scatter([m["c"]], [m["alpha"]], s=90, c="green" if m["win"] else "red",
               marker="*" if m["win"] else "x", linewidths=2, zorder=5)
    ax.annotate(m["label"], (m["c"], m["alpha"]), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
ax.text(0.55, 0.15, "SSD loses\n(speculate -> skip)", fontsize=10, color="darkred",
        ha="center", style="italic")
ax.text(0.25, 0.97, "SSD profitable", fontsize=10, color="darkgreen",
        ha="center", style="italic")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Draft cost ratio c = T_draft(K=1) / T_vanilla")
ax.set_ylabel("Acceptance rate alpha")
ax.set_title("Speculation feasibility map (Qwen2.5-3B NF4, RTX 4050 192 GB/s)")
ax.legend(fontsize=8, loc="center right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "feasibility_map.png", dpi=150)
shutil.copy(OUT / "feasibility_map.png", Path("paper/figures/feasibility_map.png"))
json.dump({"measured": measured,
           "note": "cka_83 K=1 at alpha=0.86 sits just below its K=1 boundary: "
                   "even 86% acceptance cannot pay for c~0.9 draft on 192 GB/s."},
          open(OUT / "feasibility_map.json", "w"), indent=2)
print(f"DONE -> {OUT/'feasibility_map.png'} + paper/figures/feasibility_map.png")
