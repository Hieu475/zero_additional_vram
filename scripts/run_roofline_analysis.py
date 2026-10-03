"""Direction 2 — Roofline analysis (CPU-only, no GPU needed).

1. Calibrate: predicted vs measured speedup on RTX 4050 (report MAPE).
2. Cross-GPU sweep: predicted speedup on 5 GPUs x {c, alpha, K}.
3. Break-even table: minimum alpha to win for each (GPU, K, c).
4. Save JSON + figure for the paper.

Run: python3 scripts/run_roofline_analysis.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from zassd.modeling.roofline import SpeculativeRoofline, KNOWN_GPUS

OUT = Path("experiments/16_hybrid_roofline")
OUT.mkdir(parents=True, exist_ok=True)

roof = SpeculativeRoofline()

# --- 1. Calibration vs measured numbers (Qwen2.5-3B, RTX 4050, report Phase 14/15) ---
measured_ls = [
    # (config, K, measured alpha, measured speedup, description)
    ("cka_75", 2, 0.732, 36.4 / 41.7, "CKA Fixed K=2"),
    ("cka_75", 1, 0.86, 0.98, "ZASSD mid K=1 (40.2/41.0)"),
]
c_map = {"cka_75": 0.75, "cka_83": 0.83}
calib = []
for cfg, k, alpha, meas_sp, desc in measured_ls:
    p = roof.predict(KNOWN_GPUS["RTX4050-laptop"], c=c_map[cfg], alpha=alpha, k=k)
    err = abs(p.speedup - meas_sp) / meas_sp
    calib.append({"desc": desc, "predicted": p.speedup, "measured": round(meas_sp, 3),
                  "rel_err": round(err, 3), "cycle_ms": p.t_cycle_ms})
    print(f"[CALIB] {desc}: pred={p.speedup:.3f} vs meas={meas_sp:.3f} err={err*100:.1f}%")
# PLD calibration with hit-rate h (fit h to match the measured 1.39x)
for h in [0.10, 0.12, 0.15, 0.22]:
    pp = roof.predict_pld(KNOWN_GPUS["RTX4050-laptop"], k=4, alpha=0.95, hit_rate=h)
    print(f"[CALIB] PLD h={h}: pred={pp.speedup:.3f} vs meas=1.390")
pp = roof.predict_pld(KNOWN_GPUS["RTX4050-laptop"], k=4, alpha=0.95, hit_rate=0.12)
calib.append({"desc": "Prompt Lookup K=4 (h=0.12 fit)", "predicted": pp.speedup,
              "measured": 1.39, "rel_err": round(abs(pp.speedup - 1.39) / 1.39, 3),
              "cycle_ms": pp.t_cycle_ms})
mape = sum(c["rel_err"] for c in calib) / len(calib)
print(f"[CALIB] MAPE = {mape*100:.1f}% (2 layer-skip points + 1 PLD hit-rate-fit point)")

# --- 2. Cross-GPU sweep ---
sweep_rows = []
for gname, g in KNOWN_GPUS.items():
    for (cfg, c, alpha, k) in [
        ("cka_75", 0.75, 0.73, 2), ("cka_83", 0.83, 0.86, 1),
        ("cka_50", 0.50, 0.37, 1), ("pld", 0.01, 0.90, 3),
    ]:
        p = roof.predict(g, c=c, alpha=alpha, k=k)
        sweep_rows.append({"gpu": gname, "strategy": cfg, "k": k, "alpha": alpha,
                           "speedup": p.speedup, "alpha_be": p.alpha_breakeven,
                           "cycle_ms": p.t_cycle_ms, "wins": p.wins})
        print(f"[SWEEP] {gname:14s} {cfg:7s} K={k} -> {p.speedup:.3f}x (be_alpha={p.alpha_breakeven:.2f})")

# --- 3. Break-even alpha table (c=0.75, K=1..4) ---
be_table = []
for gname, g in KNOWN_GPUS.items():
    row = {"gpu": gname}
    for k in [1, 2, 4]:
        p = roof.predict(g, c=0.75, alpha=0.0, k=k)
        row[f"K={k}"] = p.alpha_breakeven
    be_table.append(row)
print("\n[BREAK-EVEN] minimum alpha for speedup>1 (c=0.75):")
for r_ in be_table:
    print(" ", r_)

with open(OUT / "roofline_results.json", "w") as f:
    json.dump({"calibration": calib, "mape": round(mape, 4),
               "sweep": sweep_rows, "breakeven": be_table}, f, indent=2)

# --- 4. Figure ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bws = [KNOWN_GPUS[g].bandwidth_gbs for g in KNOWN_GPUS]
    names = list(KNOWN_GPUS.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (cfg, c, alpha, k, style) in [
        ("cka_75/K=2", 0.75, 0.73, 2, "o-"),
        ("cka_83/K=1", 0.83, 0.86, 1, "s-"),
        ("PLD/K=3", 0.01, 0.90, 3, "^-"),
    ]:
        ys = [roof.predict(KNOWN_GPUS[g], c=c, alpha=alpha, k=k).speedup for g in names]
        ax.plot(bws, ys, style, label=cfg)
    ax.axhline(1.0, color="red", ls="--", label="break-even")
    ax.set_xscale("log")
    ax.set_xlabel("Memory bandwidth (GB/s)")
    ax.set_ylabel("Predicted speedup vs vanilla")
    ax.set_title("Speculative Roofline: speedup vs bandwidth (Qwen2.5-3B NF4)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "roofline_speedup_vs_bandwidth.png", dpi=150)
    print(f"[FIG] saved {OUT/'roofline_speedup_vs_bandwidth.png'}")
except Exception as e:
    print(f"[FIG] skip ({e})")

print(f"\nDONE. Results -> {OUT/'roofline_results.json'}")
