#!/usr/bin/env python3
"""Analyze completed output-128 n=20 context-matrix cells.

Incomplete cells are ignored. Results are split into all structurally complete
cells and cells whose records all have clean working/source-tree provenance.
Within-condition bootstraps resample sessions and prompt/source documents,
stratified by LongBench task. Cross-condition contrasts pair source documents
and resample sessions independently within each backend or context.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/backend_validation/context_matrix_n20"
CONTEXTS = (128, 384, 1024, 2048)
BACKENDS = {"bnb": "hf_bnb_nf4", "hqq": "hqq_gemlite_w4a16"}
METHODS = ("vanilla", "pld_k4", "cka90_k2")
SPECULATIVE = METHODS[1:]
EXPECTED_PROMPTS = 20
EXPECTED_ROWS = EXPECTED_PROMPTS * len(METHODS)
BOOTSTRAPS = 5000


def prompt_stratum(prompt_id: str) -> str:
    source_id = re.sub(r"_ctx\d+$", "", prompt_id)
    return source_id.split("_", 1)[0]


def resample_prompts(prompts: list[str], rng: random.Random) -> list[str]:
    """Resample documents within LongBench task to preserve cohort mix."""
    strata: dict[str, list[str]] = defaultdict(list)
    for prompt in prompts:
        strata[prompt_stratum(prompt)].append(prompt)
    return [selected for stratum in sorted(strata)
            for selected in rng.choices(strata[stratum], k=len(strata[stratum]))]


def spearman_rho(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        ordered = sorted(range(len(values)), key=values.__getitem__)
        output = [0.0] * len(values)
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
                j += 1
            rank = (i + 1 + j) / 2
            for k in range(i, j):
                output[ordered[k]] = rank
            i = j
        return output

    xr, yr = ranks(x_values), ranks(y_values)
    xmean, ymean = statistics.mean(xr), statistics.mean(yr)
    covariance = sum((a - xmean) * (b - ymean) for a, b in zip(xr, yr))
    denom = (sum((a - xmean) ** 2 for a in xr) * sum((b - ymean) ** 2 for b in yr)) ** 0.5
    return covariance / denom if denom else None


def acceptance_speedup_association(acceptance: dict[tuple[str, str], float],
                                   speedups: dict[tuple[str, str], float],
                                   seed: int) -> dict[str, Any]:
    prompts = sorted({p for _, p in acceptance} & {p for _, p in speedups})
    per_prompt = {}
    for prompt in prompts:
        a = [v for (_, p), v in acceptance.items() if p == prompt]
        s = [v for (_, p), v in speedups.items() if p == prompt]
        if a and s:
            per_prompt[prompt] = (statistics.mean(a), statistics.mean(s))
    if len(per_prompt) < 3:
        return {"n_prompts": len(per_prompt), "spearman_rho": None,
                "ci95_stratified_prompt_bootstrap": None}
    ids = list(per_prompt)
    rho = spearman_rho([per_prompt[p][0] for p in ids], [per_prompt[p][1] for p in ids])
    rng = random.Random(seed)
    boot = []
    for _ in range(BOOTSTRAPS):
        sampled = resample_prompts(ids, rng)
        value = spearman_rho([per_prompt[p][0] for p in sampled],
                             [per_prompt[p][1] for p in sampled])
        if value is not None:
            boot.append(value)
    return {
        "n_prompts": len(per_prompt),
        "spearman_rho": rho,
        "ci95_stratified_prompt_bootstrap": [quantile(boot, 0.025), quantile(boot, 0.975)] if boot else None,
        "interpretation": "descriptive association; prompt bootstrap is conditional on the included session measurements",
    }


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_mean_ci(observations: dict[tuple[str, str], float], seed: int) -> list[float] | None:
    """Bootstrap sessions and prompts within a single experimental condition."""
    sessions = sorted({s for s, _ in observations})
    prompts = sorted({p for _, p in observations})
    if len(sessions) < 2 or len(prompts) < 2:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled_sessions = rng.choices(sessions, k=len(sessions))
        sampled_prompts = resample_prompts(prompts, rng)
        values = [observations[(s, p)] for s in sampled_sessions for p in sampled_prompts
                  if (s, p) in observations]
        if values:
            means.append(statistics.mean(values))
    return [quantile(means, 0.025), quantile(means, 0.975)] if means else None


def nested_condition_delta(base: dict[tuple[str, str], float],
                           target: dict[tuple[str, str], float],
                           seed: int) -> dict[str, Any]:
    """Compare conditions paired by prompt, resampling sessions independently.

    Session labels are local replicate labels, not evidence that measurements
    in different backends/contexts occurred in one synchronized run block.
    """
    base_sessions = sorted({s for s, _ in base})
    target_sessions = sorted({s for s, _ in target})
    prompts = sorted({p for _, p in base} & {p for _, p in target})
    per_prompt = {}
    for prompt in prompts:
        base_values = [v for (s, p), v in base.items() if p == prompt]
        target_values = [v for (s, p), v in target.items() if p == prompt]
        if base_values and target_values:
            per_prompt[prompt] = statistics.mean(target_values) - statistics.mean(base_values)
    if not per_prompt:
        return {"mean": None, "ci95_two_way_bootstrap": None, "n_prompts": 0,
                "base_session_count": len(base_sessions), "target_session_count": len(target_sessions)}
    estimate = statistics.mean(per_prompt.values())
    ci = None
    if len(base_sessions) >= 2 and len(target_sessions) >= 2 and len(prompts) >= 2:
        rng = random.Random(seed)
        boot = []
        for _ in range(BOOTSTRAPS):
            sampled_base = rng.choices(base_sessions, k=len(base_sessions))
            sampled_target = rng.choices(target_sessions, k=len(target_sessions))
            sampled_prompts = resample_prompts(prompts, rng)
            deltas = []
            for prompt in sampled_prompts:
                base_values = [base[(s, prompt)] for s in sampled_base if (s, prompt) in base]
                target_values = [target[(s, prompt)] for s in sampled_target if (s, prompt) in target]
                if base_values and target_values:
                    deltas.append(statistics.mean(target_values) - statistics.mean(base_values))
            if deltas:
                boot.append(statistics.mean(deltas))
        if boot:
            ci = [quantile(boot, 0.025), quantile(boot, 0.975)]
    return {
        "mean": estimate,
        "ci95_two_way_bootstrap": ci,
        "n_prompts": len(per_prompt),
        "base_session_count": len(base_sessions),
        "target_session_count": len(target_sessions),
    }


def read_cell(path: Path, backend: str, context: int, cohort: list[str],
              expected_hashes: dict[str, str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    invalid = False
    for row in rows:
        pid, method = row.get("prompt_id"), row.get("method")
        if pid not in cohort or method not in METHODS or method in grouped[pid]:
            invalid = True
            continue
        grouped[pid][method] = row
    complete = (
        len(rows) == EXPECTED_ROWS
        and set(grouped) == set(cohort)
        and all(set(grouped[pid]) == set(METHODS) for pid in cohort)
        and all(row.get("status") == "ok" and row.get("warmup_status", {}).get("passed") is True
                for methods in grouped.values() for row in methods.values())
        and all(row.get("backend") == backend
                and row.get("prompt_tokens") == context
                and row.get("max_new_tokens") == 128
                and row.get("kv_capacity") == context + 132
                and row.get("prompt_token_ids_sha256") == expected_hashes.get(row.get("prompt_id"))
                for methods in grouped.values() for row in methods.values())
        and not invalid
    )
    if complete:
        for methods in grouped.values():
            vanilla_ids = methods["vanilla"].get("token_ids")
            if not isinstance(vanilla_ids, list):
                complete = False
                break
            for method, row in methods.items():
                generated = row.get("token_ids")
                if not isinstance(generated, list):
                    complete = False
                    break
                exact = generated == vanilla_ids
                first_difference = next(
                    (index for index, (left, right) in enumerate(zip(generated, vanilla_ids))
                     if left != right), None)
                if first_difference is None and len(generated) != len(vanilla_ids):
                    first_difference = min(len(generated), len(vanilla_ids))
                if (row.get("within_backend_exact_vs_vanilla") is not exact
                        or row.get("first_divergence_vs_vanilla") != first_difference):
                    complete = False
                    break
            if not complete:
                break
    if not complete:
        return None
    dirty_values = [row.get("runtime", {}).get("working_tree_dirty")
                    for methods in grouped.values() for row in methods.values()]
    source_dirty_values = [row.get("runtime", {}).get("source_tree_dirty")
                           for methods in grouped.values() for row in methods.values()]
    commits = sorted({row.get("runtime", {}).get("execution_commit")
                      for methods in grouped.values() for row in methods.values()})
    source_clean = all(value is False for value in source_dirty_values)
    tree_clean = all(value is False for value in dirty_values)
    valid_hashes = all(bool(row.get("prompt_token_ids_sha256"))
                       for methods in grouped.values() for row in methods.values())
    return {
        "rows": rows,
        "grouped": grouped,
        "complete": True,
        "working_tree_clean": tree_clean,
        "source_tree_clean": source_clean,
        "clean_provenance": tree_clean and source_clean and valid_hashes and len(commits) == 1,
        "execution_commits": commits,
        "working_tree_dirty_values": sorted({str(v) for v in dirty_values}),
        "source_tree_dirty_values": sorted({str(v) for v in source_dirty_values}),
    }


def summarize_observations(observations: dict[tuple[str, str], float], seed: int,
                           threshold: float = 1.0) -> dict[str, Any]:
    values = list(observations.values())
    return {
        "n_prompt_session_pairs": len(values),
        "n_sessions": len({s for s, _ in observations}),
        "n_prompts": len({p for _, p in observations}),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p10": quantile(values, 0.10),
        "p90": quantile(values, 0.90),
        "below_threshold_fraction": sum(v < threshold for v in values) / len(values) if values else None,
        "ci95_two_way_bootstrap": bootstrap_mean_ci(observations, seed),
    }


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": quantile(values, 0.90),
    }


def main() -> None:
    cohorts = {
        context: [line.strip() for line in (BASE / f"cohort_ctx{context}.txt").read_text().splitlines()
                  if line.strip()]
        for context in CONTEXTS
    }
    prepared = {
        row["prompt_id"]: hashlib.sha256(
            json.dumps(row["input_ids"], separators=(",", ":")).encode()
        ).hexdigest()
        for row in (json.loads(line) for line in (BASE / "prepared_inputs.jsonl").read_text().splitlines()
                    if line.strip())
    }
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    inventory = []
    for tag, backend in BACKENDS.items():
        for context in CONTEXTS:
            for session in (1, 2, 3):
                name = f"n20_ctx{context}_out128_{tag}_session{session}.jsonl"
                cell = read_cell(BASE / name, backend, context, cohorts[context], prepared)
                inventory.append({
                    "file": name,
                    "complete": cell is not None,
                    "clean_provenance": bool(cell and cell["clean_provenance"]),
                    "execution_commits": cell["execution_commits"] if cell else [],
                })
                if cell:
                    cells[(tag, context, session)] = cell

    result: dict[str, Any] = {
        "scope": "output-128 context matrix; only structurally complete 20-prompt cells",
        "prompt_cohort": "20 prompts per context; four prompts from each of narrativeqa, qasper, gov_report, multi_news, qmsum",
        "expected_cells": len(BACKENDS) * len(CONTEXTS) * 3,
        "complete_cells": len(cells),
        "clean_provenance_cells": sum(cell["clean_provenance"] for cell in cells.values()),
        "bootstrap": {
            "method": "within-condition summaries resample sessions and prompt documents stratified by LongBench task; backend/context contrasts pair source documents and resample sessions independently within each condition",
            "resamples": BOOTSTRAPS,
            "limitations": "Intervals are omitted if either condition has fewer than two independent sessions. Intervals based on only two or three sessions remain fragile. Prompt/source IDs are the paired unit across conditions.",
        },
        "inventory": inventory,
        "per_backend_context": {},
        "paired_backend_speedup_delta": {},
        "paired_context_contrasts": {},
        "acceptance_speedup_associations": {},
        "related_profile_evidence": {
            "report": "experiments/backend_validation/kernel_profile/report.md",
            "scope": "instrumented one-prompt BNB NF4 profile; context 384, output budget 32, KV capacity 436",
            "method_level_dequant_kernel_time_share": {
                "vanilla": 0.0803,
                "pld_k4": 0.0669,
                "cka90_k2": 0.0420,
            },
            "method_level_nf4_gemm_kernel_time_share": {
                "vanilla": 0.3777,
                "pld_k4": 0.4931,
                "cka90_k2": 0.4747,
            },
            "interpretation": "supports rejecting dequantization-only bottleneck claims; instrumented single-prompt profiles are not causal or production timing evidence",
        },
        "metric_definition": "decode TPS is (generated token count - 1) / decode time; internal speedup = method TPS / same-session, same-prompt vanilla TPS",
    }

    for tag, backend in BACKENDS.items():
        for context in CONTEXTS:
            sessions = {s: cells[(tag, context, s)] for s in (1, 2, 3)
                        if (tag, context, s) in cells}
            if not sessions:
                continue
            for clean_only in (False, True):
                selected = {s: c for s, c in sessions.items()
                            if not clean_only or c["clean_provenance"]}
                if not selected:
                    continue
                section = "clean_provenance_only" if clean_only else "all_complete_including_exploratory"
                key = f"{backend}/ctx{context}/{section}"
                result["per_backend_context"][key] = {
                    "sessions": sorted(selected),
                    "methods": {},
                }
                for method in SPECULATIVE:
                    ratios: dict[tuple[str, str], float] = {}
                    e2e_speedups: dict[tuple[str, str], float] = {}
                    exact_observations: dict[tuple[str, str], float] = {}
                    exact_values = []
                    tps_values = []
                    token_mismatches = 0
                    token_comparison_slots = 0
                    allocated_values = []
                    reserved_values = []
                    acceptance_values = []
                    acceptance_observations: dict[tuple[str, str], float] = {}
                    divergence_positions = []
                    output_lengths = []
                    for session, cell in selected.items():
                        for prompt in cohorts[context]:
                            pair = cell["grouped"][prompt]
                            vanilla = pair["vanilla"]
                            measure = pair[method]
                            observation_key = (str(session), prompt)
                            v_tps = float(vanilla["decode_tokens_per_second"])
                            m_tps = float(measure["decode_tokens_per_second"])
                            if v_tps > 0:
                                ratios[observation_key] = m_tps / v_tps
                            v_e2e = float(vanilla["total_time_s"])
                            m_e2e = float(measure["total_time_s"])
                            if m_e2e > 0:
                                e2e_speedups[observation_key] = v_e2e / m_e2e
                            tps_values.append(m_tps)
                            exact = measure.get("within_backend_exact_vs_vanilla")
                            if exact is not None:
                                exact_values.append(bool(exact))
                                exact_observations[observation_key] = float(bool(exact))
                            reference_ids = pair["vanilla"]["token_ids"]
                            generated_ids = measure["token_ids"]
                            compared = max(len(reference_ids), len(generated_ids))
                            mismatches = sum(left != right for left, right in zip(reference_ids, generated_ids))
                            mismatches += abs(len(reference_ids) - len(generated_ids))
                            token_mismatches += mismatches
                            token_comparison_slots += compared
                            divergence = measure.get("first_divergence_vs_vanilla")
                            if divergence is not None:
                                divergence_positions.append(int(divergence))
                            output_lengths.append(len(measure.get("token_ids", [])))
                            if measure.get("peak_allocated_mb") is not None:
                                allocated_values.append(float(measure["peak_allocated_mb"]))
                            if measure.get("peak_reserved_mb") is not None:
                                reserved_values.append(float(measure["peak_reserved_mb"]))
                            if measure.get("acceptance_rate") is not None:
                                acceptance = float(measure["acceptance_rate"])
                                acceptance_values.append(acceptance)
                                acceptance_observations[observation_key] = acceptance
                    result["per_backend_context"][key]["methods"][method] = {
                        "internal_speedup": summarize_observations(ratios, 7219 + context),
                        "end_to_end_speedup": summarize_observations(e2e_speedups, 17129 + context),
                        "mean_decode_tps": statistics.mean(tps_values) if tps_values else None,
                        "exact_count": sum(exact_values),
                        "exact_n": len(exact_values),
                        "exact_rate": sum(exact_values) / len(exact_values) if exact_values else None,
                        "exact_rate_ci95_two_way_bootstrap": bootstrap_mean_ci(exact_observations, 22123 + context),
                        "token_mismatches": token_mismatches,
                        "token_comparison_slots": token_comparison_slots,
                        "token_disagreement_rate": token_mismatches / token_comparison_slots
                        if token_comparison_slots else None,
                        "first_divergence_generated_token": describe([float(x) for x in divergence_positions]),
                        "output_length": describe([float(x) for x in output_lengths]),
                        "peak_allocated_mb": describe(allocated_values),
                        "peak_reserved_mb": describe(reserved_values),
                        "acceptance_rate": describe(acceptance_values),
                        "acceptance_vs_internal_speedup": acceptance_speedup_association(
                            acceptance_observations, ratios, 24107 + context),
                    }
                    result["acceptance_speedup_associations"][f"{key}/{method}"] = result["per_backend_context"][key]["methods"][method]["acceptance_vs_internal_speedup"]

    # Backend differences are paired by prompt only. Session replicates are
    # nested within backend and resampled independently; equal session labels
    # do not imply that the runs were physically paired.
    for context in CONTEXTS:
        for provenance, clean_only in (("all_complete_including_exploratory", False),
                                       ("clean_provenance_only", True)):
            selected = {
                tag: {s: cells[(tag, context, s)] for s in (1, 2, 3)
                      if (tag, context, s) in cells
                      and (not clean_only or cells[(tag, context, s)]["clean_provenance"])}
                for tag in BACKENDS
            }
            if not all(selected[tag] for tag in BACKENDS):
                continue
            for method in SPECULATIVE:
                backend_observations: dict[str, dict[tuple[str, str], float]] = {"bnb": {}, "hqq": {}}
                for tag in BACKENDS:
                    for session, cell in selected[tag].items():
                        for prompt in cohorts[context]:
                            pair = cell["grouped"][prompt]
                            vanilla_tps = float(pair["vanilla"]["decode_tokens_per_second"])
                            backend_observations[tag][(str(session), prompt)] = (
                                float(pair[method]["decode_tokens_per_second"]) / vanilla_tps
                            )
                contrast = nested_condition_delta(
                    backend_observations["bnb"], backend_observations["hqq"], 9913 + context)
                result["paired_backend_speedup_delta"][f"{context}/{method}/{provenance}"] = {
                    "estimand": "HQQ internal speedup minus BNB internal speedup, paired by prompt; session replicates resampled independently within backend",
                    "base_backend_sessions": sorted(selected["bnb"]),
                    "target_backend_sessions": sorted(selected["hqq"]),
                    "provenance_scope": provenance,
                    **contrast,
                }

    # Context effects are paired by source document. Replicate labels are
    # context-local, so sessions are resampled independently across contexts.
    base_context = 128
    for tag, backend in BACKENDS.items():
        for target_context in CONTEXTS:
            if target_context == base_context:
                continue
            for provenance, clean_only in (("all_complete_including_exploratory", False),
                                           ("clean_provenance_only", True)):
                base_sessions = [s for s in (1, 2, 3)
                                 if (tag, base_context, s) in cells
                                 and (not clean_only or cells[(tag, base_context, s)]["clean_provenance"])]
                target_sessions = [s for s in (1, 2, 3)
                                   if (tag, target_context, s) in cells
                                   and (not clean_only or cells[(tag, target_context, s)]["clean_provenance"])]
                if not base_sessions or not target_sessions:
                    continue
                for method in SPECULATIVE:
                    base_speedups: dict[tuple[str, str], float] = {}
                    target_speedups: dict[tuple[str, str], float] = {}
                    base_e2e: dict[tuple[str, str], float] = {}
                    target_e2e: dict[tuple[str, str], float] = {}
                    base_exact: dict[tuple[str, str], float] = {}
                    target_exact: dict[tuple[str, str], float] = {}

                    def prompt_metrics(cell: dict[str, Any], prompt: str) -> tuple[float, float, float]:
                        pair = cell["grouped"][prompt]
                        vanilla, speculative = pair["vanilla"], pair[method]
                        tps_ratio = float(speculative["decode_tokens_per_second"]) / float(vanilla["decode_tokens_per_second"])
                        e2e_ratio = float(vanilla["total_time_s"]) / float(speculative["total_time_s"])
                        exact = speculative.get("within_backend_exact_vs_vanilla")
                        return tps_ratio, e2e_ratio, float(bool(exact)) if exact is not None else float("nan")

                    for session in base_sessions:
                        base_cell = cells[(tag, base_context, session)]
                        for base_prompt in cohorts[base_context]:
                            source_id = re.sub(r"_ctx\d+$", "", base_prompt)
                            values = prompt_metrics(base_cell, base_prompt)
                            obs_key = (str(session), source_id)
                            base_speedups[obs_key], base_e2e[obs_key] = values[:2]
                            if values[2] == values[2]:
                                base_exact[obs_key] = values[2]
                    for session in target_sessions:
                        target_cell = cells[(tag, target_context, session)]
                        for target_prompt in cohorts[target_context]:
                            source_id = re.sub(r"_ctx\d+$", "", target_prompt)
                            values = prompt_metrics(target_cell, target_prompt)
                            obs_key = (str(session), source_id)
                            target_speedups[obs_key], target_e2e[obs_key] = values[:2]
                            if values[2] == values[2]:
                                target_exact[obs_key] = values[2]
                    key = f"{backend}/{target_context}-minus-{base_context}/{method}/{provenance}"
                    exact_contrast = nested_condition_delta(base_exact, target_exact, 21311 + target_context)
                    result["paired_context_contrasts"][key] = {
                        "estimand": "within-source change in method speedup relative to vanilla; session replicates resampled independently within context",
                        "base_context": base_context,
                        "target_context": target_context,
                        "base_sessions": base_sessions,
                        "target_sessions": target_sessions,
                        "provenance_scope": provenance,
                        "decode_speedup_delta": nested_condition_delta(base_speedups, target_speedups, 18311 + target_context),
                        "end_to_end_speedup_delta": nested_condition_delta(base_e2e, target_e2e, 19391 + target_context),
                        "exact_rate_delta_percentage_points": {
                            **exact_contrast,
                            "mean": 100 * exact_contrast["mean"] if exact_contrast["mean"] is not None else None,
                            "ci95_two_way_bootstrap": [100 * v for v in exact_contrast["ci95_two_way_bootstrap"]]
                            if exact_contrast["ci95_two_way_bootstrap"] else None,
                        },
                    }

    out_json = BASE / "paired_summary.json"
    out_md = BASE / "paired_summary.md"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Output-128 context matrix: paired preliminary summary", "",
        "Only structurally complete cells are included (20 prompts × 3 methods, successful warmups, no failed method rows, expected KV capacity). Each context has four prompts from each of five LongBench tasks: NarrativeQA, QASPER, GovReport, MultiNews, and QMSum. Incomplete files are ignored. Clean-provenance results are shown separately from exploratory runs.", "",
        f"- Complete cells: **{result['complete_cells']}/{result['expected_cells']}**.",
        f"- Cells with fully clean working-tree and source-tree provenance: **{result['clean_provenance_cells']}**.",
        "- TPS definition: `(generated tokens - 1) / decode time`; speedup is paired against vanilla for the same backend, prompt, and session.",
        "- Within-cell 95% intervals resample sessions and prompt documents stratified by LongBench task. Backend/context contrasts pair source documents and resample sessions independently within each condition; session numbers are not assumed to identify synchronized runs.",
        "- Cross-backend comparison is HQQ internal speedup minus BNB internal speedup. Quantized weights differ, so this is an inference-stack comparison, not a kernel-only effect.", "",
        "## Clean-provenance cells", "",
        "| Backend | Context | Method | Session IDs | Decode speedup | 95% interval | Decode slowdown | E2E speedup | Exact vs own vanilla | Token disagreement | First mismatch (median zero-based index) | Mean decode TPS | Peak allocated (MB) |",
        "|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, cell in result["per_backend_context"].items():
        if not key.endswith("clean_provenance_only"):
            continue
        backend, ctx, _ = key.split("/")
        for method, metrics in cell["methods"].items():
            speed = metrics["internal_speedup"]
            ci = speed["ci95_two_way_bootstrap"]
            ci_text = "n/a (<2 sessions)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            e2e = metrics["end_to_end_speedup"]
            mismatch = metrics["first_divergence_generated_token"]["median"]
            mismatch_text = "none" if mismatch is None else f"{mismatch:.0f}"
            lines.append(
                f"| {backend} | {ctx.removeprefix('ctx')} | {method} | {','.join(map(str, cell['sessions']))} | "
                f"{speed['mean']:.3f}× | {ci_text} | {speed['below_threshold_fraction']:.1%} | "
                f"{e2e['mean']:.3f}× | {metrics['exact_count']}/{metrics['exact_n']} | "
                f"{metrics['token_disagreement_rate']:.1%} | {mismatch_text} | {metrics['mean_decode_tps']:.2f} | "
                f"{metrics['peak_allocated_mb']['mean']:.0f} |"
            )
    lines += [
        "", "## Complete cells including exploratory provenance", "",
        "These results include complete cells with historical dirty or missing provenance fields. Use them as sensitivity analysis, not as the clean-only result.", "",
        "| Backend | Context | Method | Session count | Decode speedup | 95% interval | Decode slowdown | E2E speedup | Exact vs own vanilla | Token disagreement | First mismatch (median zero-based index) | Mean decode TPS | Peak allocated (MB) |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, cell in result["per_backend_context"].items():
        if not key.endswith("all_complete_including_exploratory"):
            continue
        backend, ctx, _ = key.split("/")
        for method, metrics in cell["methods"].items():
            speed = metrics["internal_speedup"]
            ci = speed["ci95_two_way_bootstrap"]
            ci_text = "n/a (<2 sessions)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            e2e = metrics["end_to_end_speedup"]
            mismatch = metrics["first_divergence_generated_token"]["median"]
            mismatch_text = "none" if mismatch is None else f"{mismatch:.0f}"
            lines.append(
                f"| {backend} | {ctx.removeprefix('ctx')} | {method} | {len(cell['sessions'])} | "
                f"{speed['mean']:.3f}× | {ci_text} | {speed['below_threshold_fraction']:.1%} | "
                f"{e2e['mean']:.3f}× | {metrics['exact_count']}/{metrics['exact_n']} | "
                f"{metrics['token_disagreement_rate']:.1%} | {mismatch_text} | {metrics['mean_decode_tps']:.2f} | "
                f"{metrics['peak_allocated_mb']['mean']:.0f} |"
            )
    lines += ["", "## Paired backend deltas", "",
              "Positive values mean HQQ gains more internal speedup than BNB. Rows pair by prompt; replicate sessions are resampled independently within backend. They do not compare cross-backend token identity.", "",
              "| Context | Method | Provenance | BNB session IDs | HQQ session IDs | Mean HQQ−BNB speedup delta | 95% interval |",
              "|---:|---|---|---|---|---:|---|"]
    if result["paired_backend_speedup_delta"]:
        for key, metrics in result["paired_backend_speedup_delta"].items():
            ci = metrics["ci95_two_way_bootstrap"]
            ci_text = "n/a (<2 sessions)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            context, method, provenance = key.split("/")
            provenance_label = "clean" if provenance == "clean_provenance_only" else "exploratory"
            lines.append(f"| {context} | {method} | {provenance_label} | {','.join(map(str, metrics['base_backend_sessions']))} | {','.join(map(str, metrics['target_backend_sessions']))} | {metrics['mean']:+.3f} | {ci_text} |")
    else:
        lines.append("| — | — | none yet | — | — |")
    lines += ["", "## Paired context-length contrasts", "",
              "Each contrast pairs the same source documents between ctx128 and the target context. Replicate sessions are resampled independently within each context. Positive decode or E2E deltas mean the method's speedup ratio improved at the longer context; negative exact-rate deltas mean exactness declined.", "",
              "| Backend | Context contrast | Method | Provenance | ctx128 sessions | Target sessions | Decode speedup delta | 95% interval | E2E speedup delta | Exact-rate delta (pp) |",
              "|---|---:|---|---|---|---|---:|---|---:|---:|"]
    if result["paired_context_contrasts"]:
        for key, item in result["paired_context_contrasts"].items():
            backend, contrast, method, provenance = key.split("/")
            decode = item["decode_speedup_delta"]
            ci = decode["ci95_two_way_bootstrap"]
            ci_text = "n/a (<2 sessions)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            e2e = item["end_to_end_speedup_delta"]["mean"]
            exact = item["exact_rate_delta_percentage_points"]["mean"]
            provenance_label = "clean" if provenance == "clean_provenance_only" else "exploratory"
            exact_text = "n/a" if exact is None else f"{exact:+.1f}"
            lines.append(
                f"| {backend} | {contrast} | {method} | {provenance_label} | "
                f"{','.join(map(str, item['base_sessions']))} | {','.join(map(str, item['target_sessions']))} | "
                f"{decode['mean']:+.3f} | {ci_text} | "
                f"{e2e:+.3f} | {exact_text} |"
            )
    else:
        lines.append("| — | — | — | — | no matched cells yet | — | — | — | — |")
    lines += ["", "## Acceptance versus speedup", "",
              "Spearman association uses each prompt's mean acceptance rate and internal speedup over the available sessions. Its interval resamples prompts within task and is conditional on those sessions; it is descriptive, not a causal effect.", "",
              "| Backend | Context | Method | Provenance | Prompts | Spearman rho | 95% prompt interval |",
              "|---|---:|---|---|---:|---:|---|"]
    for key, association in result["acceptance_speedup_associations"].items():
        backend, ctx, provenance, method = key.split("/")
        ci = association["ci95_stratified_prompt_bootstrap"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        rho_text = "n/a" if association["spearman_rho"] is None else f"{association['spearman_rho']:+.3f}"
        provenance_label = "clean" if provenance == "clean_provenance_only" else "exploratory"
        lines.append(f"| {backend} | {ctx.removeprefix('ctx')} | {method} | {provenance_label} | {association['n_prompts']} | {rho_text} | {ci_text} |")
    lines += ["", "## Cross-check with existing kernel profile", "",
              "The separate instrumented BNB NF4 profile at context 384/output 32 reports dequantization kernel shares of 8.0% for vanilla, 6.7% for PLD K4, and 4.2% for CKA90/K2, while NF4 GEMM shares are 37.8%, 49.3%, and 47.5%. CKA90/K2 has the lowest dequant share yet remains slower in the paired context matrix. This supports avoiding a dequantization-only explanation; it does not establish what causes the slowdown because the profile is a single instrumented prompt with a different budget/capacity. See [`kernel_profile/report.md`](../kernel_profile/report.md).", ""]
    lines += ["", "## Evidence visible in this rolling snapshot", "",
              "The clean-provenance point estimates show CKA90/K2 below 1× for both decode and end-to-end speedup in every completed cell. PLD K4 is not uniformly profitable: BNB ctx1024 is below 1×, and many prompt-level runs still slow down even when the cell mean exceeds 1×. Neither method reaches 100% exact sequence match in these clean cells, so the current evidence does not support a lossless claim.", ""]
    bnb_ctx = result["paired_context_contrasts"].get(
        "hf_bnb_nf4/1024-minus-128/pld_k4/all_complete_including_exploratory")
    hqq_ctx = result["paired_context_contrasts"].get(
        "hqq_gemlite_w4a16/1024-minus-128/pld_k4/all_complete_including_exploratory")
    if bnb_ctx and hqq_ctx:
        bnb_ci = bnb_ctx["decode_speedup_delta"]["ci95_two_way_bootstrap"]
        hqq_ci = hqq_ctx["decode_speedup_delta"]["ci95_two_way_bootstrap"]
        bnb_ci_text = "not estimable" if bnb_ci is None else f"[{bnb_ci[0]:+.3f}, {bnb_ci[1]:+.3f}]"
        hqq_ci_text = "not estimable" if hqq_ci is None else f"[{hqq_ci[0]:+.3f}, {hqq_ci[1]:+.3f}]"
        lines.append(
            "The exploratory matched-source comparison from ctx128 to ctx1024 estimates a PLD speedup change of "
            f"{bnb_ctx['decode_speedup_delta']['mean']:+.3f} for BNB (95% interval {bnb_ci_text}) and "
            f"{hqq_ctx['decode_speedup_delta']['mean']:+.3f} for HQQ (95% interval {hqq_ci_text}). "
            "This pattern is consistent with backend-dependent context sensitivity, but the BNB/HQQ runs include historical provenance limitations and should be treated as a follow-up hypothesis."
        )
    lines += ["", "## Interpretation limits", "",
              "This report is a rolling summary: it changes as queued benchmark cells finish. A clean single-session cell supports a reproducible descriptive result but not a session-level uncertainty interval. Bootstrap intervals based on only two or three session clusters remain fragile; collecting more independent sessions would strengthen them. Exploratory cells are retained in `paired_summary.json` and are not mixed into the clean-only table.",
              ""]
    out_md.write_text("\n".join(lines))
    print(json.dumps({"complete_cells": result["complete_cells"],
                      "clean_provenance_cells": result["clean_provenance_cells"],
                      "json": str(out_json.relative_to(ROOT)),
                      "report": str(out_md.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
