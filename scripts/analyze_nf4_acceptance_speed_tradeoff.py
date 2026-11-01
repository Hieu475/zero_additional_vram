"""Summarize 256-token B1 speedup against proposal acceptance by development prompt."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_row_kernel_v1"


def geomean(values):
    return math.exp(mean(math.log(value) for value in values))


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def correlation(xs, ys):
    xbar, ybar = mean(xs), mean(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - xbar) ** 2 for x in xs) * sum((y - ybar) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def main():
    cnn_path = EXP / "closed_loop_256_summary.json"
    he_path = EXP / "humaneval_256_four_prompt_summary.json"
    cnn = json.loads(cnn_path.read_text())
    humaneval = json.loads(he_path.read_text())
    groups = {}

    for prompt_id, prompt in cnn["prompts"].items():
        records = [session["artifact"] for session in prompt["sessions"]]
        groups[prompt_id] = ("CNN/DM", records)
    for prompt_id, prompt in humaneval["prompts"].items():
        groups[prompt_id] = ("HumanEval", prompt["artifacts"])

    rows = []
    for prompt_id, (task, artifacts) in groups.items():
        accepted = drafted = 0
        ratios_b1, ratios_bnb = [], []
        exact = True
        bnb_matches = bnb_comparisons = sessions = 0
        for name in artifacts:
            record = json.loads((EXP / name).read_text())
            sessions += 1
            exact &= bool(record["token_match"] and record["all_shared_states_equal"] and record["same_prefix_domain"])
            first_block = record["timing"]["paired_rounds"][0]["methods"]["optimized_b1_block"]
            accepted += int(first_block["accepted_draft_tokens"])
            drafted += int(first_block["draft_tokens"])
            for timing in record["timing"]["paired_rounds"]:
                methods = timing["methods"]
                block = methods["optimized_b1_block"]
                ratios_b1.append(methods["optimized_b1_singleton"]["total_s"] / block["total_s"])
                ratios_bnb.append(methods["original_bnb_greedy"]["total_s"] / block["total_s"])
                bnb_comparisons += 1
                bnb_matches += int(timing["block_matches_original_bnb"])
        rows.append({
            "task": task,
            "prompt_id": prompt_id,
            "process_sessions": sessions,
            "timing_pairs": len(ratios_b1),
            "accepted_draft_tokens": accepted,
            "draft_tokens": drafted,
            "acceptance_rate": accepted / drafted if drafted else None,
            "geomean_speedup_vs_optimized_b1_singleton": geomean(ratios_b1),
            "geomean_speedup_vs_original_bnb": geomean(ratios_bnb),
            "b1_token_kv_logical_lengths_exact": exact,
            "bnb_token_matches": bnb_matches,
            "bnb_token_comparisons": bnb_comparisons,
            "artifacts": artifacts,
        })
    rows.sort(key=lambda item: (item["task"], item["prompt_id"]))

    acceptance = [row["acceptance_rate"] for row in rows]
    b1 = [row["geomean_speedup_vs_optimized_b1_singleton"] for row in rows]
    bnb = [row["geomean_speedup_vs_original_bnb"] for row in rows]
    loo = []
    for omit in range(len(rows)):
        kept = [i for i in range(len(rows)) if i != omit]
        loo.append({
            "omitted_prompt": rows[omit]["prompt_id"],
            "pearson_acceptance_vs_b1_speedup": correlation([acceptance[i] for i in kept], [b1[i] for i in kept]),
            "pearson_acceptance_vs_bnb_speedup": correlation([acceptance[i] for i in kept], [bnb[i] for i in kept]),
        })
    by_task = {}
    for task in sorted({row["task"] for row in rows}):
        members = [row for row in rows if row["task"] == task]
        by_task[task] = {
            "prompts": len(members),
            "geomean_speedup_vs_optimized_b1_singleton_equal_prompt_weight": geomean([r["geomean_speedup_vs_optimized_b1_singleton"] for r in members]),
            "geomean_speedup_vs_original_bnb_equal_prompt_weight": geomean([r["geomean_speedup_vs_original_bnb"] for r in members]),
            "mean_acceptance_rate_equal_prompt_weight": mean(r["acceptance_rate"] for r in members),
        }

    # Resample prompt means within each task, preserving the observed 3:4 task mix.
    # This captures prompt heterogeneity, not within-process timing uncertainty.
    rng = random.Random(20261029)
    task_rows = {task: [row for row in rows if row["task"] == task]
                 for task in sorted({row["task"] for row in rows})}
    bootstrap_values = {"b1": [], "bnb": []}
    task_bootstrap = {task: {"b1": [], "bnb": []} for task in task_rows}
    bootstrap_replicates = 20000
    for _ in range(bootstrap_replicates):
        sampled = []
        for task, members in task_rows.items():
            draw = [rng.choice(members) for _ in range(len(members))]
            sampled.extend(draw)
            task_bootstrap[task]["b1"].append(geomean([r["geomean_speedup_vs_optimized_b1_singleton"] for r in draw]))
            task_bootstrap[task]["bnb"].append(geomean([r["geomean_speedup_vs_original_bnb"] for r in draw]))
        bootstrap_values["b1"].append(geomean([r["geomean_speedup_vs_optimized_b1_singleton"] for r in sampled]))
        bootstrap_values["bnb"].append(geomean([r["geomean_speedup_vs_original_bnb"] for r in sampled]))

    def interval(values):
        ordered = sorted(values)
        return {"lower_2_5_percent": ordered[int(0.025 * (len(ordered) - 1))],
                "upper_97_5_percent": ordered[int(0.975 * (len(ordered) - 1))]}

    clustered_bootstrap = {
        "method": "stratified prompt-cluster bootstrap of per-prompt geomean speedups; within-task prompt counts are preserved",
        "resamples": bootstrap_replicates,
        "seed": 20261029,
        "unit": "prompt mean (timing repetitions within each prompt are not resampled)",
        "equal_prompt_weighted_speedup_vs_b1_singleton": interval(bootstrap_values["b1"]),
        "equal_prompt_weighted_speedup_vs_original_bnb": interval(bootstrap_values["bnb"]),
        "by_task": {
            task: {
                "speedup_vs_b1_singleton": interval(task_bootstrap[task]["b1"]),
                "speedup_vs_original_bnb": interval(task_bootstrap[task]["bnb"]),
                "prompt_count": len(task_rows[task]),
            }
            for task in task_rows
        },
        "interpretation_limit": "Describes prompt heterogeneity in this selected cohort only; seven non-random prompt means across two tasks do not support population-level coverage.",
    }

    output = {
        "status": "descriptive analysis of existing 256-token development runs; not an inferential or general performance claim",
        "sources": [cnn_path.name, he_path.name],
        "prompt_count": len(rows),
        "process_count": sum(row["process_sessions"] for row in rows),
        "timing_pair_count": sum(row["timing_pairs"] for row in rows),
        "all_prompts_exact_to_optimized_b1": all(row["b1_token_kv_logical_lengths_exact"] for row in rows),
        "prompts": rows,
        "equal_prompt_weighted_geomean_speedup_vs_optimized_b1_singleton": geomean(b1),
        "equal_prompt_weighted_geomean_speedup_vs_original_bnb": geomean(bnb),
        "acceptance_speedup_association_across_prompt_means": {
            "pearson_acceptance_vs_b1_speedup": correlation(acceptance, b1),
            "spearman_acceptance_vs_b1_speedup": correlation(average_ranks(acceptance), average_ranks(b1)),
            "pearson_acceptance_vs_bnb_speedup": correlation(acceptance, bnb),
            "spearman_acceptance_vs_bnb_speedup": correlation(average_ranks(acceptance), average_ranks(bnb)),
            "leave_one_prompt_out_pearson": loo,
            "interpretation_limit": "n=7 prompt means; task and prompt effects are confounded, HumanEval/112 has exceptionally high acceptance, and these correlations are descriptive only.",
        },
        "task_level_equal_prompt_weighted": by_task,
        "stratified_prompt_cluster_bootstrap_intervals": clustered_bootstrap,
        "bnb_token_matches": sum(row["bnb_token_matches"] for row in rows),
        "bnb_token_comparisons": sum(row["bnb_token_comparisons"] for row in rows),
        "limitations": [
            "Five prompts across two development tasks on one RTX 4050; prompts, not timing repetitions, are the analysis units.",
            "The HumanEval/112 high-acceptance prompt strongly influences the association and aggregate speedups.",
            "The stratified prompt-cluster bootstrap intervals describe this selected cohort only; they are not generalizable population confidence intervals.",
            "All exactness is relative to the experimental optimized B1 singleton; BNB token identity is separate.",
            "Held-out GSM8K indices 300-349 were not used.",
        ],
    }
    target = EXP / "acceptance_speed_tradeoff_256_analysis.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(target.relative_to(ROOT)),
        "prompt_count": len(rows),
        "process_count": output["process_count"],
        "timing_pairs": output["timing_pair_count"],
        "equal_prompt_geomean_vs_b1": output["equal_prompt_weighted_geomean_speedup_vs_optimized_b1_singleton"],
        "equal_prompt_geomean_vs_bnb": output["equal_prompt_weighted_geomean_speedup_vs_original_bnb"],
        "correlation": output["acceptance_speedup_association_across_prompt_means"],
        "prompts": [{k: row[k] for k in ("task", "prompt_id", "acceptance_rate", "geomean_speedup_vs_optimized_b1_singleton", "geomean_speedup_vs_original_bnb")} for row in rows],
    }, indent=2))


if __name__ == "__main__":
    main()
