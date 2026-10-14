#!/usr/bin/env python3
"""Analyze paired WP4 records with prompt-cluster bootstrap intervals."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED = ("pld_k2", "pld_k4", "pld_k8", "cka90_k2")


def ci_mean(values: list[float], seed: int = 20261016, rounds: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[int(.025 * len(means))], means[min(len(means) - 1, int(.975 * len(means)))]]


def load_pairs(path: Path) -> dict[str, dict[str, dict]]:
    pairs: dict[str, dict[str, dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pairs.setdefault(row["prompt_id"], {})[row["method"]] = row
    return pairs


def metric(pair: dict[str, dict], method: str, key: str):
    return pair[method]["measurement"][key]


def choose_best_fixed(pairs: dict[str, dict[str, dict]]) -> tuple[str, dict]:
    dev = [p for p in pairs.values() if p["direct_vanilla"]["selection_role"] == "development"]
    if not dev:
        raise ValueError("No development prompts available for selecting best fixed method")
    means = {m: sum(metric(p, m, "decode_tps") for p in dev) / len(dev) for m in FIXED}
    return max(means, key=means.get), means


def summarize_method(pairs: dict[str, dict[str, dict]], method: str,
                     label: str, seed: int) -> dict:
    def in_split(pair: dict) -> bool:
        if label == "all":
            return True
        if label == "diagnostic_evaluation":
            return (pair["direct_vanilla"]["cohort"] == "diagnostic"
                    and pair["direct_vanilla"]["selection_role"] != "development")
        return pair["direct_vanilla"]["cohort"] == label

    selected = [(pid, p) for pid, p in pairs.items() if in_split(p) and method in p]
    speedups, tps, e2e, exact, slow, vram, verify, acceptance = [], [], [], [], [], [], [], []
    pld4_ratios, speedup_diffs, regret_tps, regret_fraction = [], [], [], []
    for _, p in selected:
        data = p[method]["measurement"]
        vanilla = p["direct_vanilla"]["measurement"]
        speedups.append(data["paired_speedup_vs_vanilla"])
        tps.append(data["decode_tps"])
        e2e.append(data["total_time_s"])
        exact.append(float(data["sequence_exact_match"]))
        slow.append(float(data["slowdown_vs_vanilla"]))
        vram.append(data["peak_vram_mb"])
        verify.append(data["verification_time_s"])
        acceptance.append(data.get("acceptance_rate"))
        best_fixed_tps = max(metric(p, fixed, "decode_tps") for fixed in FIXED)
        regret_tps.append(max(0.0, best_fixed_tps - data["decode_tps"]))
        regret_fraction.append(max(0.0, (best_fixed_tps - data["decode_tps"]) / max(best_fixed_tps, 1e-9)))
        if method != "pld_k4":
            pld4_ratios.append(data["decode_tps"] / max(p["pld_k4"]["measurement"]["decode_tps"], 1e-9))
            speedup_diffs.append(data["paired_speedup_vs_vanilla"] - p["pld_k4"]["measurement"]["paired_speedup_vs_vanilla"])
        else:
            pld4_ratios.append(1.0)
            speedup_diffs.append(0.0)
    acc_values = [x for x in acceptance if x is not None]
    return {"method": method, "n": len(selected),
            "mean_decode_tps": sum(tps) / max(1, len(tps)),
            "mean_end_to_end_latency_s": sum(e2e) / max(1, len(e2e)),
            "mean_verification_latency_s": sum(verify) / max(1, len(verify)),
            "mean_paired_speedup_vs_vanilla": sum(speedups) / max(1, len(speedups)),
            "paired_speedup_95ci": ci_mean(speedups, seed),
            "mean_speedup_difference_vs_pld_k4": sum(speedup_diffs) / max(1, len(speedup_diffs)),
            "speedup_difference_vs_pld_k4_95ci": ci_mean(speedup_diffs, seed + 1),
            "paired_tps_ratio_vs_pld_k4": sum(pld4_ratios) / max(1, len(pld4_ratios)),
            "tps_ratio_vs_pld_k4_95ci": ci_mean(pld4_ratios, seed + 2),
            "mean_regret_tps_vs_offline_best_fixed": sum(regret_tps) / max(1, len(regret_tps)),
            "mean_regret_fraction_vs_offline_best_fixed": sum(regret_fraction) / max(1, len(regret_fraction)),
            "regret_fraction_95ci": ci_mean(regret_fraction, seed + 3),
            "sequence_exact_match_count": sum(exact),
            "sequence_exact_match_rate": sum(exact) / max(1, len(exact)),
            "slowdown_count": sum(slow), "slowdown_frequency": sum(slow) / max(1, len(slow)),
            "mean_peak_vram_mb": sum(vram) / max(1, len(vram)),
            "mean_acceptance_rate": sum(acc_values) / len(acc_values) if acc_values else None,
            "first_divergence_positions": [p[method]["measurement"]["first_divergence_position"]
                                            for _, p in selected
                                            if p[method]["measurement"]["first_divergence_position"] is not None]}


def analyze(run_dir: Path) -> dict:
    raw = run_dir / "raw_measurements.jsonl"
    pairs = load_pairs(raw)
    best, dev_means = choose_best_fixed(pairs)
    wp3_path = ROOT / "experiments/numerical_causality/precision_validation_final/sdpa_static.jsonl"
    wp3_refs = {row["prompt_id"]: row["canonical_token_ids"]
                for row in (json.loads(line) for line in wp3_path.read_text().splitlines() if line.strip())}
    ref_checks = {pid: pair["direct_vanilla"]["measurement"]["generated_token_ids"] == wp3_refs[pid]
                  for pid, pair in pairs.items()}
    heldout_ids = [pid for pid, p in pairs.items()
                   if p["direct_vanilla"]["cohort"] == "heldout"]
    methods = [*ALL_METHODS]
    result = {"n_prompts": len(pairs), "development_prompt_ids": sorted(
                  pid for pid, p in pairs.items() if p["direct_vanilla"]["selection_role"] == "development"),
              "heldout_prompt_ids": sorted(heldout_ids), "best_fixed_method": best,
              "development_mean_decode_tps": dev_means,
              "direct_vanilla_agreement_with_wp3_reference": {
                  "n_equal": sum(ref_checks.values()), "n_prompts": len(ref_checks),
                  "mismatched_prompt_ids": sorted(pid for pid, equal in ref_checks.items() if not equal)},
              "splits": {}}
    for split in ("all", "heldout", "diagnostic", "diagnostic_evaluation"):
        if split == "all":
            group = dict(pairs)
        elif split == "diagnostic_evaluation":
            group = {pid: p for pid, p in pairs.items()
                     if p["direct_vanilla"]["cohort"] == "diagnostic"
                     and p["direct_vanilla"]["selection_role"] != "development"}
        else:
            group = {pid: p for pid, p in pairs.items()
                     if p["direct_vanilla"]["cohort"] == split}
        rows = {}
        for method in methods:
            if method == "best_fixed_selected_on_development":
                rows[method] = summarize_method(group, best, split, 611)
                rows[method]["selected_fixed_method"] = best
            elif method == "offline_best_fixed_per_prompt":
                # Deliberately clairvoyant per-prompt comparator, not a deployable policy.
                oracle = {}
                for pid, pair in group.items():
                    choice = max(FIXED, key=lambda m: metric(pair, m, "decode_tps"))
                    oracle[pid] = pair[choice]
                replaced = {pid: {**pair, "offline_best_fixed_per_prompt": oracle[pid]}
                            for pid, pair in group.items()}
                rows[method] = summarize_method(replaced, method, split, 612)
                rows[method]["oracle_methods_selected"] = dict(Counter(
                    max(FIXED, key=lambda m: metric(pair, m, "decode_tps")) for pair in group.values()))
            else:
                rows[method] = summarize_method(group, method, split, 610 + len(rows))
        actions = Counter()
        for pair in group.values():
            counts = pair["hybrid_router"]["measurement"].get("action_counts", {})
            actions.update(counts)
        result["splits"][split] = {"methods": rows, "router_action_distribution": dict(actions)}
    return result


ALL_METHODS = ("direct_vanilla", "vanilla_speculative_core_k0", *FIXED,
               "hybrid_router", "best_fixed_selected_on_development",
               "offline_best_fixed_per_prompt")


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/numerical_causality/unified_router_evaluation"
    result = analyze(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = ["# Unified paired router evaluation", "",
              f"Prompts: {result['n_prompts']}; best fixed selected on development: `{result['best_fixed_method']}`.",
              f"Development fixed-method mean decode TPS: `{result['development_mean_decode_tps']}`.",
              f"Direct vanilla matched WP3 canonical token IDs on {result['direct_vanilla_agreement_with_wp3_reference']['n_equal']}/{result['n_prompts']} prompts.",
              "", "All speedups use per-prompt decode TPS. CIs bootstrap prompts; exactness compares raw output token IDs with direct vanilla.", ""]
    for split, data in result["splits"].items():
        report += [f"## {split}", "", "| Method | N | Decode TPS | Speedup vs vanilla (95% CI) | TPS ratio vs PLD K4 (95% CI) | Exact match | Slowdown | Regret vs offline fixed (TPS) | VRAM MB | Acceptance |", "|---|---:|---:|---|---|---:|---:|---:|---:|---:|"]
        for method, row in data["methods"].items():
            ci = row["paired_speedup_95ci"]
            ci_txt = "—" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            rci = row["tps_ratio_vs_pld_k4_95ci"]
            rci_txt = "—" if rci is None else f"[{rci[0]:.3f}, {rci[1]:.3f}]"
            acc = "—" if row["mean_acceptance_rate"] is None else f"{row['mean_acceptance_rate']:.3f}"
            report.append(f"| {method} | {row['n']} | {row['mean_decode_tps']:.2f} | {row['mean_paired_speedup_vs_vanilla']:.3f} {ci_txt} | {row['paired_tps_ratio_vs_pld_k4']:.3f} {rci_txt} | {row['sequence_exact_match_count']}/{row['n']} | {row['slowdown_count']}/{row['n']} | {row['mean_regret_tps_vs_offline_best_fixed']:.2f} | {row['mean_peak_vram_mb']:.0f} | {acc} |")
        report += ["", f"Router action distribution: `{data['router_action_distribution']}`.", ""]
    report += ["## Limits", "", "The 8 development prompts are selected from the known mismatch diagnostic cohort, so best-fixed selection may not generalize. Held-out results use the 14 unselected WP3 prompts. Offline best-per-prompt is an upper-bound diagnostic using future per-prompt timing and is not deployable. Each method is run once per prompt; prompt bootstrap intervals do not represent between-session or thermal variation. No cycle-level oracle is constructed.", ""]
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"best_fixed_method": result["best_fixed_method"], "splits": {
        k: {m: {"exact": v["sequence_exact_match_rate"], "speedup": v["mean_paired_speedup_vs_vanilla"]}
            for m, v in value["methods"].items()} for k, value in result["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
