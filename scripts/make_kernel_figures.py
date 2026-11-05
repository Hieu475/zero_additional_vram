"""Figures for the M-invariant kernel direction (report + paper). Reads results/kernels/*.json."""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results/kernels"); OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "bao_cao/images"); OUT.mkdir(exist_ok=True, parents=True)
BLUE, ORANGE, GRAY = "#2a6fb0", "#d9822b", "#8a8a8a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})

cells = [("Qwen2.5-3B GSM8K", "qwen3b_gsm8k200"), ("Qwen2.5-3B HumanEval", "qwen3b_humaneval100"), ("Qwen2.5-3B CNN/DM", "qwen3b_cnndm100"),
         ("Qwen2.5-1.5B GSM8K", "qwen15b_gsm8k100"), ("Qwen2.5-0.5B GSM8K", "qwen05b_gsm8k100"), ("OLMo-2-1B GSM8K", "olmo2_1b_gsm8k100"), ("Phi-3.5-mini GSM8K", "phi35_gsm8k100"),
         ("Llama-3.2-1B GSM8K", "wp8:llama1b_gsm8k"), ("Llama-3.2-3B GSM8K", "wp8:llama3b_gsm8k")]
per = {}
tot = {}
for name, key in cells:
    d = json.load(open(R / (f"wp8_eos_{key[4:]}.json" if key.startswith("wp8:") else f"wp5_scale_{key}.json")))
    sp = np.array([r["s_graph_m1"] / r["s_a3"] for r in d["results"]])
    per[name] = sp; tot[name] = d["speedup_a3_over_graph_m1"]

# Fig A: per-prompt speedup distributions
fig, ax = plt.subplots(figsize=(7.5, 3.8))
data = [per[n] for n, _ in cells]
bp = ax.boxplot(data, orientation='horizontal', widths=.55, showfliers=True, patch_artist=True, medianprops=dict(color="k"),
                flierprops=dict(marker="o", markersize=3, markerfacecolor=GRAY, markeredgecolor="none"))
for b in bp["boxes"]: b.set(facecolor="#cfe0f1", edgecolor=BLUE)
ax.axvline(1.0, color=ORANGE, lw=1.2); ax.set_yticks(range(1, len(cells) + 1)); ax.set_yticklabels([n for n, _ in cells]); ax.invert_yaxis()
for i, (n, _) in enumerate(cells, 1): ax.plot(tot[n], i, "D", color=ORANGE, ms=5)
ax.set_xlabel("Speedup per prompt vs exact M=1 baseline (same kernels)"); ax.set_title("Exact PLD loop: per-prompt speedup (diamond = aggregate)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "kernel_speedup_cells.pdf"); plt.close(fig)

# Fig B: decode step time vs M (numbers from plan section 14, measured in CUDA graph)
Ms = ["1", "3-4", "5-6"]; ours = [15.0, 17.7, 23.6]; bnb = [16.3, 21.2, 31.0]
x = np.arange(len(Ms)); w = .36
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.bar(x - w / 2, bnb, w, color=GRAY, label="bitsandbytes (stock)"); ax.bar(x + w / 2, ours, w, color=BLUE, label="M-invariant CUDA NF4")
for xi, v in zip(x - w / 2, bnb): ax.text(xi, v + .3, f"{v:.1f}", ha="center", fontsize=8)
for xi, v in zip(x + w / 2, ours): ax.text(xi, v + .3, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(Ms); ax.set_xlabel("Rows per block M"); ax.set_ylabel("Decode step (ms), Qwen2.5-3B"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "kernel_step_time.pdf"); plt.close(fig)

# Fig C: trajectory divergence of naive PLD vs invariant (CSV from audit)
import csv
rows = list(csv.DictReader(open("paper/figures_c/fig1_trajectory_divergence.csv")))
naive = [float(r["pct"]) for r in rows if r["method"] == "naive PLD" and "independent" not in r["dataset"] and "torchao" not in r["config"]]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.hist(naive, bins=np.arange(20, 90, 8), color=ORANGE, edgecolor="w")
ax.set_xlabel("Sequences leaving the sequential greedy trajectory (%)"); ax.set_ylabel("Cells (model x dataset)")
ax.set_title(f"Naive PLD, 15 cells: {min(naive):.0f}-{max(naive):.0f}%; M-invariant loop: 0 in 1000 runs", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "kernel_naive_divergence.pdf"); plt.close(fig)
print("figures ->", OUT)
