#!/usr/bin/env python3
"""Aggregate results/drafter_runs/*.json: speedup over the exact M=1 baseline per (model, set, method), mean/sd over interleaved reps, exactness flags."""
import json, glob, os, statistics as st
O = "results/drafter_runs"; SETS = ["gsm8k_wp5_100", "humaneval_wp5_100", "cnndm_wp5_100"]
def load(files, kind):
    xs, ex = [], True
    for f in files:
        d = json.load(open(f))
        if kind == "pld": xs.append(d["speedup_a3_over_graph_m1"]); ex &= bool(d["all_exact"])
        else: xs.append(d["speedup_vs_g1_async"]); ex &= bool(d["all_exact_vs_g1"] and d["all_exact_vs_eager"])
    return xs, ex
def files(model, method, b):
    if model == "qwen3b":
        pat = {"pld": f"pld_{b}_k3_rep*.json", "drafter": f"{b}_d2k3_rep*.json", "hybrid": f"{b}_hyb_n2*.json"}[method]
    else:
        pat = f"q15_{method}_{b}_rep*.json"
    return sorted(glob.glob(f"{O}/{pat}"))
rows = []
for model in ("qwen3b", "qwen1.5b"):
    print(f"\n{model}: speedup vs exact M=1 baseline (mean ± sd over n reps; all_exact)")
    print(f"{'set':12s} {'PLD K=3':26s} {'learned drafter K=3':26s} {'hybrid':26s}")
    for b in SETS:
        cells = []
        for method in ("pld", "drafter", "hybrid"):
            fs = files(model, method, b)
            if not fs: cells.append("-"); continue
            xs, ex = load(fs, method)
            sd = f"±{st.stdev(xs):.3f}" if len(xs) > 1 else "     "
            cells.append(f"{st.mean(xs):.3f}{sd} n={len(xs)} {'exact' if ex else 'MISMATCH'}")
            rows.append({"model": model, "set": b, "method": method, "n": len(xs), "mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else None, "reps": xs, "exact": ex})
        print(f"{b.split('_')[0]:12s} {cells[0]:26s} {cells[1]:26s} {cells[2]:26s}")
json.dump(rows, open(f"{O}/summary.json", "w"), indent=1)
