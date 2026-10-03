#!/usr/bin/env python3
"""Router-vs-oracle regret + robustness metrics (CPU-only).

Oracle: per-PROMPT best of {Vanilla, PLD, LayerSkip} measured TPS, where
LayerSkip = ZASSD Empirical-6 (k=1), the pure layer-skip strategy.
  regret(prompt) = TPS_oracle(prompt) - TPS_router(prompt)
Regret can be NEGATIVE: the router mixes strategies within a prompt, so it
may beat every single-strategy oracle on that prompt. The per-prompt oracle
is therefore a reference, not an upper bound (a per-cycle oracle would need
per-cycle traces, which we log as future work).

Robustness per strategy: min task-mean speedup, std across tasks,
fraction of prompts slower than vanilla, fraction of tasks with mean < 1.

Run: python3 scripts/analyze_oracle_regret.py
Output: experiments/16_hybrid_roofline/oracle_regret.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

try:
    from scipy import stats as ss
    HAS = True
except ImportError:
    HAS = False

OUT = Path("experiments/16_hybrid_roofline")
ROUTED = "ZASSD Routed (k=2+PLD4)"
ORACLE_SET = ["Vanilla", "Prompt-Lookup (PLD)", "ZASSD Empirical-6 (k=1)"]

SOURCES = {
    "gsm8k_N50": "experiments/16_hybrid_roofline/standard_suite_N50.json",
    "gsm8k_N200": "experiments/16_hybrid_roofline/standard_suite_gsm8k_N200.json",
}


def load_bench(path, bench=None):
    d = json.load(open(path))
    if "benchmarks" in d:  # N50-style multi-bench file
        out = {}
        for bname, b in d["benchmarks"].items():
            out[bname] = b["raw_samples"]
        return out
    return {bench or "gsm8k": d["raw_samples"]}  # merged single-bench file


def main():
    rep = {"oracle_set": ORACLE_SET, "benches": {}, "robustness": {}}
    for tag, path in SOURCES.items():
        for bname, raw in load_bench(path).items():
            if tag == "gsm8k_N200" and bname != "gsm8k":
                continue
            key = bname if tag == "gsm8k_N50" else "gsm8k_N200"
            ids = [r["id"] for r in raw["Vanilla"]]
            for s in list(raw):
                assert [r["id"] for r in raw[s]] == ids, f"ID misalign {key}/{s}"
            tps = {s: np.array([r["tps"] for r in raw[s]]) for s in raw}
    # per-bench oracle analysis on N50 (3 benches) + N200 gsm8k
            n = len(ids)
            oracle_tps = np.max([tps[s] for s in ORACLE_SET], axis=0)
            oracle_choice = np.argmax([tps[s] for s in ORACLE_SET], axis=0)
            r_tps = tps[ROUTED]
            regret = oracle_tps - r_tps
            share = {s: round(float(np.mean(oracle_choice == i)), 3)
                     for i, s in enumerate(ORACLE_SET)}
            eff = float(np.mean(r_tps) / np.mean(oracle_tps))  # mean-ratio efficiency
            rep["benches"][key] = {
                "n": n,
                "mean_tps": {s: round(float(np.mean(tps[s])), 2) for s in raw},
                "oracle_mean_tps": round(float(np.mean(oracle_tps)), 2),
                "routed_mean_tps": round(float(np.mean(r_tps)), 2),
                "oracle_share": share,
                "regret_mean": round(float(np.mean(regret)), 3),
                "regret_std": round(float(np.std(regret, ddof=1)), 3),
                "regret_frac_negative": round(float(np.mean(regret < 0)), 3),
                "router_efficiency_vs_oracle": round(eff, 4),
            }
            if HAS:
                t = ss.ttest_1samp(regret, 0.0)
                rep["benches"][key]["regret_ttest_p"] = round(float(t.pvalue), 5)
    # robustness across the 3 N50 tasks + gsm8k_N200 slice
    n50 = json.load(open(SOURCES["gsm8k_N50"]))["benchmarks"]
    for s in ["Prompt-Lookup (PLD)", "ZASSD Empirical-6 (k=1)",
              "ZASSD Hybrid (k=2)", ROUTED]:
        task_means, slow_frac, all_sp = [], [], []
        for bname, b in n50.items():
            sp = np.array([r["speedup"] for r in b["raw_samples"][s]])
            task_means.append(float(np.mean(sp)))
            slow_frac.append(float(np.mean(sp < 1.0)))
            all_sp.append(sp)
        all_sp = np.concatenate(all_sp)
        rep["robustness"][s] = {
            "min_task_mean": round(min(task_means), 4),
            "std_task_means": round(float(np.std(task_means)), 4),
            "frac_prompts_slower_than_vanilla": round(float(np.mean(all_sp < 1.0)), 4),
            "frac_tasks_mean_below_1": sum(m < 1.0 for m in task_means),
        }
    json.dump(rep, open(OUT / "oracle_regret.json", "w"), indent=2)
    for k, v in rep["benches"].items():
        print(f"[{k:12s}] n={v['n']} oracle={v['oracle_mean_tps']:.1f} routed={v['routed_mean_tps']:.1f} "
              f"eff={v['router_efficiency_vs_oracle']:.3f} regret={v['regret_mean']:+.2f}±{v['regret_std']:.2f} "
              f"neg={v['regret_frac_negative']:.2f} share={v['oracle_share']}")
    print("\nRobustness (N50, 150 prompts):")
    for s, v in rep["robustness"].items():
        print(f"  {s:26s} min_task={v['min_task_mean']:.3f} slow_frac={v['frac_prompts_slower_than_vanilla']:.3f} "
              f"tasks<1: {v['frac_tasks_mean_below_1']}/3")
    print(f"DONE -> {OUT/'oracle_regret.json'}")


if __name__ == "__main__":
    main()
