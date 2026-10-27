#!/usr/bin/env python3
"""Build a finite, cost-aware per-cycle hindsight oracle from B1 pilot traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_b1_reference_contract import canonical_hash, reference_payload


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def read_run(k: int, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines() if line.strip()]
    return {"k": k, "path": path, "manifest": manifest}, rows


def shortest_path(length: int, edges: list[dict[str, Any]], allowed_k: int | None = None) -> tuple[float | None, list[dict[str, Any]]]:
    outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if allowed_k is not None and edge["k"] not in {allowed_k, None}:
            continue
        outgoing[edge["start"]].append(edge)
    cost: list[float] = [float("inf")] * (length + 1)
    back: list[tuple[int, dict[str, Any]] | None] = [None] * (length + 1)
    cost[0] = 0.0
    for position in range(length):
        if cost[position] == float("inf"):
            continue
        for edge in outgoing.get(position, []):
            end = edge["end"]
            if end <= position or end > length:
                continue
            proposed = cost[position] + edge["cost_s"]
            if proposed < cost[end]:
                cost[end] = proposed
                back[end] = (position, edge)
    if cost[length] == float("inf"):
        return None, []
    path = []
    position = length
    while position:
        previous = back[position]
        if previous is None:
            return None, []
        position, edge = previous
        path.append(edge)
    path.reverse()
    return cost[length], path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("K", "DIR"), required=True)
    parser.add_argument("--prompt-id", action="append", default=None,
                        help="Optional prompt ID filter; may be repeated")
    parser.add_argument("--reference-contract", type=Path,
                        help="Validate runs against a contract, including seed runs created before pinning it")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    runs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    contract_ids = set()
    contract = json.loads(args.reference_contract.read_text()) if args.reference_contract else None
    expected_contract = ({key: value for key, value in contract.items()
                          if key not in {"reference_contract_id", "provenance"}} if contract else None)
    contract_validations = []
    for k_text, directory in args.run:
        k = int(k_text)
        run, rows = read_run(k, Path(directory))
        valid = None
        reference_id = run["manifest"].get("reference_contract_id")
        if contract is not None:
            valid = (reference_payload(run["manifest"]) == expected_contract and
                     canonical_hash(expected_contract) == contract.get("reference_contract_id"))
            reference_id = contract.get("reference_contract_id") if valid else None
        run["reference_contract_id"] = reference_id
        run["reference_contract_valid"] = valid
        run["rows"] = rows
        runs.append(run)
        contract_ids.add(reference_id)
        contract_validations.append(valid)
        for row in rows:
            if args.prompt_id and row["id"] not in args.prompt_id:
                continue
            grouped[(row["task"], row["id"])].append({"run": run, "row": row})

    prompt_reports = []
    for (task, prompt_id), samples in sorted(grouped.items()):
        reference_sequences = [s["row"]["results"]["b1_reference"]["token_ids"] for s in samples]
        sequence_equal = all(seq == reference_sequences[0] for seq in reference_sequences[1:])
        input_hashes = {s["row"].get("input_token_ids_sha256") for s in samples}
        input_identity_equal = len(input_hashes) == 1 and None not in input_hashes
        reference_state_traces = [
            s["row"].get("state_audit", {}).get("reference_state_trace") for s in samples
        ]
        reference_states_equal = (
            bool(reference_state_traces) and reference_state_traces[0] is not None and
            all(trace == reference_state_traces[0] for trace in reference_state_traces[1:])
        )
        all_audits_pass = all(
            s["row"].get("state_audit", {}).get("all_shared_states_equal", False)
            for s in samples
        )
        length = len(reference_sequences[0])
        cycle_groups: dict[tuple[int, int, int, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
        fixed_costs: dict[int, list[float]] = defaultdict(list)
        ref_costs, prefill_costs, overheads = [], [], []
        component_sums = []
        coverage = set()
        for sample in samples:
            run, row = sample["run"], sample["row"]
            method = row["results"]["b1_block"]
            k = run["k"]
            coverage.add(k)
            if row["results"]["b1_reference"]["token_ids"] != reference_sequences[0]:
                continue
            total = float(method["total_s"])
            prefill = float(method["prefill_s"])
            cycles = method.get("cycles", [])
            cycle_sum = sum(float(c.get("cycle_s", 0.0)) for c in cycles)
            audit = row.get("state_audit", {})
            common_state_prefixes = set(audit.get("reference_prefix_counts", [])) & set(
                audit.get("block_prefix_counts", [])
            )
            fixed_costs[k].append(total)
            prefill_costs.append(prefill)
            overheads.append(total - prefill - cycle_sum)
            ref_costs.append(float(row["results"]["b1_reference"]["total_s"]))
            component_sums.extend(
                abs(float(c.get("cycle_s", 0.0)) - sum(float(c.get(key, 0.0)) for key in
                    ("draft_s", "verify_s", "commit_s"))) for c in cycles
            )
            for cycle in cycles:
                if not all(key in cycle for key in ("draft_s", "verify_s", "commit_s", "cycle_s")):
                    continue
                start = int(cycle["output_start"])
                emitted = tuple(cycle["emitted_tokens"])
                output_end = start + len(emitted)
                if tuple(reference_sequences[0][start:output_end]) != emitted:
                    continue
                terminal = method.get("terminal_reason") in {"eos", "budget"} and output_end == length
                state_end = output_end - 1 if terminal else output_end
                # The cycle edge is usable only when its committed cache state
                # was audited equal to the singleton state at that prefix.
                if state_end not in common_state_prefixes:
                    continue
                path_tokens = emitted[:-1] if terminal else emitted
                end = start + len(path_tokens)
                if end != state_end:
                    continue
                cycle_groups[(k, start, end, emitted)].append(cycle)

        # Average repeated timing observations per action. Only actions whose
        # emitted tokens agree with the common singleton trajectory are edges.
        edges = []
        for (k, start, end, emitted), cycles in cycle_groups.items():
            output_end = start + len(emitted)
            if output_end > length or tuple(reference_sequences[0][start:output_end]) != emitted:
                continue
            edges.append({
                "k": k, "start": start, "end": end,
                "emitted_tokens": list(emitted),
                "state_end": end,
                "cost_s": median([float(c["cycle_s"]) for c in cycles]),
                "draft_s": median([float(c["draft_s"]) for c in cycles]),
                "verify_s": median([float(c["verify_s"]) for c in cycles]),
                "commit_s": median([float(c["commit_s"]) for c in cycles]),
                "timing_samples": len(cycles),
            })

        # The final token may already be pending after the last measured cycle
        # and needs no extra target forward. Add that zero-cost completion edge.
        reachable_end = max((edge["end"] for edge in edges), default=0)
        if reachable_end == length - 1 and length:
            edges.append({"k": None, "start": length - 1, "end": length,
                          "emitted_tokens": [reference_sequences[0][-1]], "cost_s": 0.0,
                          "action": "emit already-pending terminal/budget token", "timing_samples": 0})
        one_contract = (len(contract_ids) == 1 and None not in contract_ids and
                        all(valid is not False for valid in contract_validations))
        cycle_cost, path = shortest_path(length, edges) if (
            one_contract and input_identity_equal and sequence_equal and
            reference_states_equal and all_audits_pass
        ) else (None, [])
        residual = median(overheads)
        oracle_total = (median(prefill_costs) + cycle_cost + residual) if cycle_cost is not None else None
        fixed_summary = {
            str(k): {"median_total_s": median(costs), "samples": len(costs)}
            for k, costs in sorted(fixed_costs.items())
        }
        fixed_cycle_model = {}
        for k in sorted(fixed_costs):
            fixed_path_cost, fixed_path = shortest_path(length, edges, allowed_k=k)
            fixed_cycle_model[str(k)] = {
                "predicted_total_s": (median(prefill_costs) + fixed_path_cost + residual)
                    if fixed_path_cost is not None else None,
                "cycle_cost_s": fixed_path_cost,
                "path": fixed_path,
            }
        best_fixed_model = min(
            (entry["predicted_total_s"] for entry in fixed_cycle_model.values()
             if entry["predicted_total_s"] is not None), default=None
        )
        prompt_reports.append({
            "task": task, "id": prompt_id, "sessions": len(samples),
            "k_coverage": sorted(coverage), "common_reference_tokens": sequence_equal,
            "input_identity_equal": input_identity_equal,
            "reference_states_equal": reference_states_equal,
            "all_state_audits_exact": all_audits_pass,
            "reference_token_count": length,
            "cycle_action_count": len(edges),
            "median_b1_singleton_total_s": median(ref_costs),
            "fixed_k": fixed_summary,
            "fixed_k_cycle_model": fixed_cycle_model,
            "cycle_oracle": {
                "predicted_total_s": oracle_total,
                "cycle_cost_s": cycle_cost,
                "median_prefill_s": median(prefill_costs),
                "median_noncycle_overhead_s": residual,
                "median_component_sum_error_s": median(component_sums),
                "speedup_vs_singleton": (median(ref_costs) / oracle_total) if oracle_total else None,
                "gain_vs_best_fixed_k": (best_fixed_model / oracle_total)
                    if oracle_total and best_fixed_model is not None else None,
                "path": path,
            },
        })

    payload = {
        "status": "finite per-cycle hindsight cost oracle; optimistic, not a deployable policy or proof",
        "reference_contract_ids": sorted(str(item) for item in contract_ids),
        "same_reference_contract": (len(contract_ids) == 1 and None not in contract_ids and
                                    all(valid is not False for valid in contract_validations)),
        "k_values": sorted({run["k"] for run in runs}),
        "oracle_assumptions": [
            "All candidate cycle edges were measured from the same B1 token and state trajectory.",
            "Cycle action cost uses the median observed draft+verify+commit wall time across sessions.",
            "The oracle has zero K-selection/switching/checkpoint overhead and uses future knowledge.",
            "Only K values and fixed-row schedule represented in the supplied runs are considered.",
            "A zero-cost final edge is allowed only for a token already pending at EOS/budget completion.",
            "Cycle timing stops before trace-list append and state-observer hashing; per-request residual is the median measured block total minus prefill and cycle sums.",
            "That residual may change with cycle count; applying its median to a different path is an approximation.",
        ],
        "prompt_count": len(prompt_reports),
        "prompts": prompt_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({
        "prompt_count": len(prompt_reports),
        "complete_paths": sum(p["cycle_oracle"]["predicted_total_s"] is not None for p in prompt_reports),
        "median_speedup_vs_singleton": median([
            p["cycle_oracle"]["speedup_vs_singleton"] for p in prompt_reports
            if p["cycle_oracle"]["speedup_vs_singleton"] is not None
        ]),
    }, indent=2))


if __name__ == "__main__":
    main()
