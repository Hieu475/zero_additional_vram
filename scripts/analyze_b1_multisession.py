#!/usr/bin/env python3
"""Aggregate paired B1 fixed-row pilots across independent process sessions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Pilot output directories for all K/session combinations")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = []
    for name in args.runs:
        path = Path(name)
        manifest = json.loads((path / "manifest.json").read_text())
        rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines() if line]
        if not rows:
            raise ValueError(f"no rows in {path}")
        runs.append((path, manifest, rows))

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    reference = runs[0][1]
    groups: dict[tuple[int, int, str, str, int], dict] = {}
    config_keys = ("model", "revision", "max_new_tokens", "attention_strategy", "capacity")
    for path, manifest, rows in runs:
        for key in config_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"configuration mismatch for {key}: {path}")
        session = int(manifest["session_index"])
        start_index = int(manifest.get("start_index", 0))
        k = int(manifest["k"])
        physical_rows = int(manifest["physical_rows"])
        run_key = (k, physical_rows, manifest["attention_strategy"], str(session), start_index)
        if run_key in groups:
            raise ValueError(f"duplicate K/session run: {run_key}")
        prompt_rows = {}
        for row in rows:
            prompt_key = (row["task"], row["id"])
            if prompt_key in prompt_rows:
                raise ValueError(f"duplicate prompt in {path}: {prompt_key}")
            if not row["block_matches_b1"]:
                raise ValueError(f"B1 token mismatch in {path}: {prompt_key}")
            if not row.get("state_audit", {}).get("all_shared_states_equal", False):
                raise ValueError(f"B1 frontier state mismatch or missing audit in {path}: {prompt_key}")
            prompt_rows[prompt_key] = {
                "task": row["task"], "id": row["id"],
                "input_token_ids_sha256": row["input_token_ids_sha256"],
                "method_order": row["method_order"],
                "block_matches_b1": row["block_matches_b1"],
                "all_shared_states_equal": row["state_audit"]["all_shared_states_equal"],
                "block_matches_original_bnb": row["block_matches_original_bnb"],
                "speedup_vs_original_bnb": row["block_speedup_vs_original"],
                "speedup_vs_b1_singleton": row["block_speedup_vs_b1"],
                "shared_frontier_count": row["state_audit"]["shared_prefixes"],
                "block_total_s": row["results"]["b1_block"]["total_s"],
                "original_total_s": row["results"]["original_greedy"]["total_s"],
                "b1_singleton_total_s": row["results"]["b1_reference"]["total_s"],
            }
        groups[run_key] = {
            "directory": str(path), "manifest": manifest, "prompts": prompt_rows,
        }

    prompt_sets_by_round: dict[tuple[int, int], set] = {}
    prompt_sets_by_cohort: dict[int, set] = {}
    for (k, physical_rows, strategy, session, start_index), run in groups.items():
        prompt_keys_for_run = set(run["prompts"])
        round_key = (int(session), start_index)
        if round_key in prompt_sets_by_round and prompt_sets_by_round[round_key] != prompt_keys_for_run:
            raise ValueError(f"K runs have different prompts in repeat round/cohort {round_key}")
        prompt_sets_by_round[round_key] = prompt_keys_for_run
        if start_index in prompt_sets_by_cohort and prompt_sets_by_cohort[start_index] != prompt_keys_for_run:
            raise ValueError(f"repeat rounds have different prompts for cohort start={start_index}")
        prompt_sets_by_cohort[start_index] = prompt_keys_for_run
    prompt_keys = sorted(set().union(*prompt_sets_by_cohort.values()))
    hashes = {}
    for run in groups.values():
        for key, row in run["prompts"].items():
            if key in hashes and hashes[key] != row["input_token_ids_sha256"]:
                raise ValueError(f"prompt tokenization differs across runs for {key}")
            hashes[key] = row["input_token_ids_sha256"]

    by_k = {}
    for (k, physical_rows, strategy, session, start_index), run in sorted(groups.items()):
        if k not in by_k:
            by_k[k] = {"physical_rows": physical_rows, "attention_strategy": strategy,
                       "sessions": [], "pairs": []}
        elif (by_k[k]["physical_rows"], by_k[k]["attention_strategy"]) != (physical_rows, strategy):
            raise ValueError(f"inconsistent schedule for K={k}")
        by_k[k]["sessions"].append({"session_index": int(session),
                                     "cohort_start_index": start_index,
                                     "directory": run["directory"]})
        for key, row in run["prompts"].items():
            by_k[k]["pairs"].append({"session_index": int(session),
                                      "cohort_start_index": start_index, **row})

    for k, summary in by_k.items():
        ratios = [row["speedup_vs_original_bnb"] for row in summary["pairs"]]
        b1_ratios = [row["speedup_vs_b1_singleton"] for row in summary["pairs"]]
        summary["summary"] = {
            "n_paired_prompt_rounds": len(ratios),
            "n_repeat_rounds": len({row["session_index"] for row in summary["pairs"]}),
            "n_prompt_cohorts": len(prompt_sets_by_cohort),
            "n_unique_prompts": len(prompt_keys),
            "exact_vs_b1": sum(row["block_matches_b1"] for row in summary["pairs"]),
            "exact_vs_original_bnb": sum(row["block_matches_original_bnb"] for row in summary["pairs"]),
            "mean_speedup_vs_original_bnb": sum(ratios) / len(ratios),
            "geometric_mean_speedup_vs_original_bnb": math.exp(sum(math.log(x) for x in ratios) / len(ratios)),
            "median_speedup_vs_original_bnb": median(ratios),
            "min_speedup_vs_original_bnb": min(ratios),
            "max_speedup_vs_original_bnb": max(ratios),
            "mean_speedup_vs_b1_singleton": sum(b1_ratios) / len(b1_ratios),
            "all_shared_frontier_states_equal": all(
                row["all_shared_states_equal"] for row in summary["pairs"]
            ),
        }
        by_prompt = {}
        for key in prompt_keys:
            values = [row["speedup_vs_original_bnb"] for row in summary["pairs"]
                      if (row["task"], row["id"]) == key]
            by_prompt[f"{key[0]}/{key[1]}"] = {
                "n_rounds": len(values), "ratios": values, "mean_ratio": sum(values) / len(values),
            }
        summary["by_prompt"] = by_prompt

    comparison = {}
    ordered_k = sorted(by_k)
    for i, left_k in enumerate(ordered_k):
        for right_k in ordered_k[i + 1:]:
            left = {(row["session_index"], row["cohort_start_index"], row["task"], row["id"]): row["block_total_s"]
                    for row in by_k[left_k]["pairs"]}
            right = {(row["session_index"], row["cohort_start_index"], row["task"], row["id"]): row["block_total_s"]
                     for row in by_k[right_k]["pairs"]}
            shared = sorted(set(left) & set(right))
            speedups = [right[key] / left[key] for key in shared]
            comparison[f"K{left_k}_over_K{right_k}"] = {
                "paired_session_prompt_count": len(shared),
                "ratio_definition": "K_right latency / K_left latency; >1 favors K_left",
                "mean_ratio": sum(speedups) / len(speedups),
                "ratios": speedups,
            }

    report = {
        "status": "diagnostic multi-session B1 pilot; not a universal proof or held-out evaluation",
        "model": reference["model"], "revision": reference["revision"],
        "max_new_tokens": reference["max_new_tokens"],
        "attention_strategy": reference["attention_strategy"],
        "prompt_count": len(prompt_keys), "prompt_hashes": {
            f"{task}/{prompt_id}": hashes[(task, prompt_id)] for task, prompt_id in prompt_keys
        },
        "by_k": by_k, "paired_k_comparisons": comparison,
        "limitations": [
            "Previously used development prompts; no held-out claim.",
            "Process sessions share the same GPU and software environment.",
            "Latency ratios are descriptive; the sample is too small for strong confidence claims.",
            "Exactness here means measured B1 token output and equality of KV hashes at shared cycle frontiers.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in by_k.items()}, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
