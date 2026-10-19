#!/usr/bin/env python3
"""Analyze completed output-128 n=20 context-matrix cells.

Incomplete cells are ignored. Results are split into all structurally complete
cells and cells whose records all have clean working/source-tree provenance.
The bootstrap resamples sessions and prompt/source documents as paired units.
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
    """Two-way paired bootstrap over complete sessions and prompt IDs."""
    sessions = sorted({s for s, _ in observations})
    prompts = sorted({p for _, p in observations})
    if len(sessions) < 2 or len(prompts) < 2:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled_sessions = rng.choices(sessions, k=len(sessions))
        sampled_prompts = rng.choices(prompts, k=len(prompts))
        values = [observations[(s, p)] for s in sampled_sessions for p in sampled_prompts
                  if (s, p) in observations]
        if values:
            means.append(statistics.mean(values))
    return [quantile(means, 0.025), quantile(means, 0.975)] if means else None


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


def summarize_observations(observations: dict[tuple[str, str], float], seed: int) -> dict[str, Any]:
    values = list(observations.values())
    return {
        "n_prompt_session_pairs": len(values),
        "n_sessions": len({s for s, _ in observations}),
        "n_prompts": len({p for _, p in observations}),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p10": quantile(values, 0.10),
        "p90": quantile(values, 0.90),
        "slowdown_fraction": sum(v < 1.0 for v in values) / len(values) if values else None,
        "ci95_two_way_bootstrap": bootstrap_mean_ci(observations, seed),
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
        "expected_cells": len(BACKENDS) * len(CONTEXTS) * 3,
        "complete_cells": len(cells),
        "clean_provenance_cells": sum(cell["clean_provenance"] for cell in cells.values()),
        "bootstrap": {
            "method": "two-way paired resampling over complete sessions and prompt IDs",
            "resamples": BOOTSTRAPS,
            "limitations": "With fewer than two clean sessions, clean-only intervals are omitted. Prompt IDs are resampled as the paired document unit.",
        },
        "inventory": inventory,
        "per_backend_context": {},
        "paired_backend_speedup_delta": {},
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
                    exact_values = []
                    tps_values = []
                    for session, cell in selected.items():
                        for prompt in cohorts[context]:
                            pair = cell["grouped"][prompt]
                            vanilla = pair["vanilla"]
                            measure = pair[method]
                            v_tps = float(vanilla["decode_tokens_per_second"])
                            m_tps = float(measure["decode_tokens_per_second"])
                            if v_tps > 0:
                                ratios[(str(session), prompt)] = m_tps / v_tps
                            tps_values.append(m_tps)
                            exact = measure.get("within_backend_exact_vs_vanilla")
                            if exact is not None:
                                exact_values.append(bool(exact))
                    result["per_backend_context"][key]["methods"][method] = {
                        "internal_speedup": summarize_observations(ratios, 7219 + context),
                        "mean_decode_tps": statistics.mean(tps_values) if tps_values else None,
                        "exact_count": sum(exact_values),
                        "exact_n": len(exact_values),
                        "exact_rate": sum(exact_values) / len(exact_values) if exact_values else None,
                    }

    # Backend difference is paired on the same clean session and prompt.
    for context in CONTEXTS:
        common_sessions = [s for s in (1, 2, 3)
                           if all((tag, context, s) in cells and cells[(tag, context, s)]["clean_provenance"]
                                  for tag in BACKENDS)]
        if not common_sessions:
            continue
        for method in SPECULATIVE:
            deltas: dict[tuple[str, str], float] = {}
            for session in common_sessions:
                bnb, hqq = cells[("bnb", context, session)], cells[("hqq", context, session)]
                for prompt in cohorts[context]:
                    def speedup(cell: dict[str, Any]) -> float:
                        pair = cell["grouped"][prompt]
                        return float(pair[method]["decode_tokens_per_second"]) / float(pair["vanilla"]["decode_tokens_per_second"])
                    deltas[(str(session), prompt)] = speedup(hqq) - speedup(bnb)
            result["paired_backend_speedup_delta"][f"ctx{context}/{method}"] = {
                "estimand": "HQQ internal speedup minus BNB internal speedup, paired by clean session and prompt",
                "sessions": common_sessions,
                **summarize_observations(deltas, 9913 + context),
            }

    out_json = BASE / "paired_summary.json"
    out_md = BASE / "paired_summary.md"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Output-128 context matrix: paired preliminary summary", "",
        "Only structurally complete cells are included (20 prompts × 3 methods, successful warmups, no failed method rows, expected KV capacity). Incomplete files are ignored. Clean-provenance results are shown separately from exploratory runs.", "",
        f"- Complete cells: **{result['complete_cells']}/{result['expected_cells']}**.",
        f"- Cells with fully clean working-tree and source-tree provenance: **{result['clean_provenance_cells']}**.",
        "- TPS definition: `(generated tokens - 1) / decode time`; speedup is paired against vanilla for the same backend, prompt, and session.",
        "- 95% intervals resample complete sessions and prompt IDs as paired units; they are omitted with fewer than two sessions.",
        "- Cross-backend comparison is HQQ internal speedup minus BNB internal speedup. Quantized weights differ, so this is an inference-stack comparison, not a kernel-only effect.", "",
        "## Clean-provenance cells", "",
        "| Backend | Context | Method | Session IDs | Mean speedup | 95% interval | Slowdown | Exact vs own vanilla | Mean decode TPS |",
        "|---|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for key, cell in result["per_backend_context"].items():
        if not key.endswith("clean_provenance_only"):
            continue
        backend, ctx, _ = key.split("/")
        for method, metrics in cell["methods"].items():
            speed = metrics["internal_speedup"]
            ci = speed["ci95_two_way_bootstrap"]
            ci_text = "n/a (<2 sessions)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(
                f"| {backend} | {ctx.removeprefix('ctx')} | {method} | {','.join(map(str, cell['sessions']))} | "
                f"{speed['mean']:.3f}× | {ci_text} | {speed['slowdown_fraction']:.1%} | "
                f"{metrics['exact_count']}/{metrics['exact_n']} | {metrics['mean_decode_tps']:.2f} |"
            )
    lines += [
        "", "## Complete cells including exploratory provenance", "",
        "These results include complete cells with historical dirty or missing provenance fields. Use them as sensitivity analysis, not as the clean-only result.", "",
        "| Backend | Context | Method | Sessions | Mean speedup | 95% interval | Slowdown | Exact vs own vanilla | Mean decode TPS |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for key, cell in result["per_backend_context"].items():
        if not key.endswith("all_complete_including_exploratory"):
            continue
        backend, ctx, _ = key.split("/")
        for method, metrics in cell["methods"].items():
            speed = metrics["internal_speedup"]
            ci = speed["ci95_two_way_bootstrap"]
            ci_text = "n/a (<2 sessions)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(
                f"| {backend} | {ctx.removeprefix('ctx')} | {method} | {len(cell['sessions'])} | "
                f"{speed['mean']:.3f}× | {ci_text} | {speed['slowdown_fraction']:.1%} | "
                f"{metrics['exact_count']}/{metrics['exact_n']} | {metrics['mean_decode_tps']:.2f} |"
            )
    lines += ["", "## Paired clean backend deltas", "",
              "Positive values mean HQQ gains more internal speedup than BNB; this does not compare cross-backend token identity.", "",
              "| Context | Method | Common clean sessions | Mean HQQ−BNB speedup delta | 95% interval |",
              "|---:|---|---|---:|---|"]
    if result["paired_backend_speedup_delta"]:
        for key, metrics in result["paired_backend_speedup_delta"].items():
            ci = metrics["ci95_two_way_bootstrap"]
            ci_text = "n/a (<2 sessions)" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            context, method = key.split("/")
            lines.append(f"| {context.removeprefix('ctx')} | {method} | {','.join(map(str, metrics['sessions']))} | {metrics['mean']:.3f} | {ci_text} |")
    else:
        lines.append("| — | — | none yet | — | — |")
    lines += ["", "## Interpretation limits", "",
              "This report is a rolling summary: it changes as queued benchmark cells finish. A clean single-session cell supports a reproducible descriptive result but not a session-level uncertainty interval. Exploratory cells are retained in `paired_summary.json` and are not mixed into the clean-only table.",
              ""]
    out_md.write_text("\n".join(lines))
    print(json.dumps({"complete_cells": result["complete_cells"],
                      "clean_provenance_cells": result["clean_provenance_cells"],
                      "json": str(out_json.relative_to(ROOT)),
                      "report": str(out_md.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
