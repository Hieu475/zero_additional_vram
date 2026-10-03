#!/usr/bin/env python3
"""Statistical rigor for the N50 standard benchmark (CPU-only).

Reads experiments/16_hybrid_roofline/standard_suite_N50.json per-sample
speedups and reports, per strategy x benchmark:
  mean, std, 95% CI (t-distribution), plus paired tests of
  ZASSD Routed vs (Vanilla=1.0, Hybrid, PLD): paired t-test, Wilcoxon
  signed-rank, and Cohen's d effect size.

Run: python3 scripts/analyze_benchmark_stats.py
Output: experiments/16_hybrid_roofline/benchmark_stats.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

try:
    from scipy import stats as sstats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SRC = Path("experiments/16_hybrid_roofline/standard_suite_N50.json")
OUT = Path("experiments/16_hybrid_roofline/benchmark_stats.json")

ROUTED = "ZASSD Routed (k=2+PLD4)"
STRATEGIES = ["Vanilla", "Prompt-Lookup (PLD)", "ZASSD Empirical-6 (k=1)",
              "ZASSD Hybrid (k=2)", ROUTED]


def ci95(x: np.ndarray):
    n = len(x)
    m, s = float(np.mean(x)), float(np.std(x, ddof=1)) if n > 1 else 0.0
    if HAS_SCIPY and n > 1:
        h = sstats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    else:
        h = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
    return m, s, h


def paired_tests(a: np.ndarray, b: np.ndarray):
    """Two-sided paired tests of a vs b. Returns dict."""
    d = a - b
    out = {"mean_diff": round(float(np.mean(d)), 4)}
    if HAS_SCIPY and len(d) >= 8:
        t = sstats.ttest_rel(a, b)
        out["paired_t_p"] = round(float(t.pvalue), 5)
        try:
            w = sstats.wilcoxon(d)
            out["wilcoxon_p"] = round(float(w.pvalue), 5)
        except Exception as e:
            out["wilcoxon_p"] = f"n/a ({e})"
        sd = float(np.std(d, ddof=1))
        out["cohens_dz"] = round(float(np.mean(d) / sd), 3) if sd > 0 else 0.0
    else:
        out["note"] = "scipy missing or n<8; descriptive only"
    return out


def main():
    d = json.load(open(SRC))
    benches = d["benchmarks"]
    report = {"per_benchmark": {}, "overall": {}}

    for bname, b in benches.items():
        raw = b["raw_samples"]
        # Paired unit = same prompt under both methods: assert ID alignment
        # so paired tests never compare mismatched prompts.
        ids = [r["id"] for r in raw[STRATEGIES[0]]]
        for s in STRATEGIES[1:]:
            assert [r["id"] for r in raw[s]] == ids, \
                f"ID misalignment in {bname} for {s}; pairing would be invalid"
        rep = {"paired_unit": "same prompt id", "n": len(ids)}
        for s in STRATEGIES:
            sp = np.array([r["speedup"] for r in raw[s]], dtype=float)
            m, sd, h = ci95(sp)
            rep[s] = {"n": len(sp), "mean": round(m, 4),
                      "std": round(sd, 4), "ci95_half": round(h, 4)}
        r = np.array([x["speedup"] for x in raw[ROUTED]])
        for other in ["ZASSD Hybrid (k=2)", "Prompt-Lookup (PLD)"]:
            o = np.array([x["speedup"] for x in raw[other]])
            rep[f"routed_vs_{other}"] = paired_tests(r, o)
        # routed vs vanilla (constant 1.0): one-sample test on (r - 1)
        rep["routed_vs_vanilla"] = paired_tests(
            r, np.ones_like(r)) if HAS_SCIPY else {"mean_diff": round(float(np.mean(r - 1)), 4)}
        report["per_benchmark"][bname] = rep

    # overall = mean of the 3 bench means per sample-id is not aligned across
    # benches, so pool all per-sample speedups across benches per strategy
    for s in STRATEGIES:
        pooled = np.concatenate([
            np.array([x["speedup"] for x in benches[b]["raw_samples"][s]])
            for b in benches])
        m, sd, h = ci95(pooled)
        report["overall"][s] = {"n": len(pooled), "mean": round(m, 4),
                                "std": round(sd, 4), "ci95_half": round(h, 4)}
    rp = np.concatenate([np.array([x["speedup"] for x in benches[b]["raw_samples"][ROUTED]]) for b in benches])
    for other in ["ZASSD Hybrid (k=2)", "Prompt-Lookup (PLD)"]:
        op = np.concatenate([np.array([x["speedup"] for x in benches[b]["raw_samples"][other]]) for b in benches])
        report["overall"][f"routed_vs_{other}"] = paired_tests(rp, op)
    report["overall"]["routed_vs_vanilla"] = paired_tests(rp, np.ones_like(rp))

    json.dump(report, open(OUT, "w"), indent=2)

    print(f"{'strategy':28s} {'GSM8K':>16s} {'HumanEval':>16s} {'CNN/DM':>16s} {'overall':>16s}")
    for s in STRATEGIES:
        row = []
        for bname in benches:
            e = report["per_benchmark"][bname][s]
            row.append(f"{e['mean']:.3f}+/-{e['ci95_half']:.3f}")
        e = report["overall"][s]
        row.append(f"{e['mean']:.3f}+/-{e['ci95_half']:.3f}")
        print(f"{s:28s} {row[0]:>16s} {row[1]:>16s} {row[2]:>16s} {row[3]:>16s}")
    print("\nPaired tests (overall, n=150):")
    for k in ["routed_vs_ZASSD Hybrid (k=2)", "routed_vs_Prompt-Lookup (PLD)", "routed_vs_vanilla"]:
        print(f"  {k}: {report['overall'][k]}")
    print(f"\nDONE -> {OUT}")


if __name__ == "__main__":
    main()
