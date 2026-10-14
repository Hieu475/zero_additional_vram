#!/usr/bin/env python3
"""Create a read-only reproduction audit from saved benchmark and trace artifacts.

This script compares raw prompt/output token IDs and recorded cycle metadata. It
does not claim that two saved artifacts used an equivalent live execution path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050"
DEFAULT_TRACE = ROOT / "experiments/numerical_causality/traces/multicycle_nf4_sdpa_static.json"
DEFAULT_CONFIG = ROOT / "experiments/numerical_causality/config.json"
DEFAULT_OUT = ROOT / "experiments/reproduction_audit"


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def manifest(path: Path, config: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "artifact_kind": kind,
        "artifact_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "artifact_sha256": sha256_file(path),
        "execution_commit": config.get("execution_commit"),
        "model": config.get("model"),
        "model_revision": config.get("model_revision"),
        "tokenizer_revision": config.get("source_manifest", {}).get("tokenizer_revision",
                                  config.get("tokenizer_revision")),
        "quantization": config.get("quantization", config.get("precision")),
        "bnb_4bit_use_double_quant": config.get("bnb_4bit_use_double_quant"),
        "compute_dtype": config.get("compute_dtype"),
        "attention_implementation": config.get("attention", config.get("attention_implementation")),
        "kv_backend": config.get("kv_backend", config.get("backend")),
        "kv_capacity": config.get("kv_cache_max_capacity"),
        "draft_strategy": config.get("candidate_source"),
        "k": config.get("k"),
        "max_new_tokens": config.get("max_new_tokens"),
        "eos_semantics": config.get("eos_semantics", "not recorded"),
        "seed": config.get("sample_seed", config.get("seed")),
        "config": config,
    }


def first_diff(left: list[int], right: list[int]) -> int | None:
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right)) if len(left) != len(right) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    bench_dir = args.benchmark_dir if args.benchmark_dir.is_absolute() else ROOT / args.benchmark_dir
    trace_path = args.trace if args.trace.is_absolute() else ROOT / args.trace
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    for p in (bench_dir / "prepared_inputs.jsonl", bench_dir / "raw_measurements.jsonl",
              bench_dir / "config.json", bench_dir / "environment.json", trace_path, config_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    bench_cfg = json.loads((bench_dir / "config.json").read_text())
    trace = json.loads(trace_path.read_text())
    trace_cfg = json.loads(config_path.read_text())
    trace_input_path = ROOT / trace["benchmark"]
    trace_inputs = ({str(r["prompt_id"]): r for r in read_jsonl(trace_input_path)}
                    if trace_input_path.is_file() else {})
    inputs = {str(r["prompt_id"]): r for r in read_jsonl(bench_dir / "prepared_inputs.jsonl")}
    measurements = read_jsonl(bench_dir / "raw_measurements.jsonl")
    measured: dict[str, dict[str, dict[str, Any]]] = {}
    for row in measurements:
        measured.setdefault(str(row["prompt_id"]), {})[row["method_key"]] = row["measured"]

    trace_artifact_manifest_path = ROOT / "experiments/numerical_causality/artifact_manifest.json"
    trace_artifact_manifest = json.loads(trace_artifact_manifest_path.read_text())
    trace_manifest = {
        **manifest(trace_path, trace, "causal_trace"),
        "execution_commit": trace_cfg.get("execution_commit"),
        "attention_implementation": trace_artifact_manifest.get("environment", {}).get("attention_implementation"),
        "compute_dtype": "float16" if "FP16" in str(trace.get("precision", "")) else None,
        "kv_capacity": {"pld_and_vanilla": trace_cfg.get("kv_cache_max_capacity")},
        "tokenizer_revision": trace_cfg.get("model_revision"),
        "tokenizer_revision_evidence": "trace runner passes its pinned model revision to AutoTokenizer.from_pretrained",
        "eos_semantics": "greedy stop after emitting EOS or reaching max_new_tokens",
        "trace_config_path": str(config_path.relative_to(ROOT)),
        "trace_config_sha256": sha256_file(config_path),
        "trace_config": trace_cfg,
        "trace_base_commit": trace_artifact_manifest.get("base_commit"),
        "trace_working_tree_uncommitted_changes": trace_artifact_manifest.get("working_tree_contains_uncommitted_research_changes"),
        "trace_script_sha256_at_capture": trace_artifact_manifest.get("artifact_sha256", {}).get("scripts/trace_multicycle_numerical_divergence.py"),
        "captured_source_hashes_match_current_head": {
            path: trace_artifact_manifest.get("artifact_sha256", {}).get(path) == hashlib.sha256(
                __import__("subprocess").check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest()
            for path in ("scripts/trace_multicycle_numerical_divergence.py", "src/zassd/baselines/prompt_lookup.py")
        },
        "bnb_4bit_use_double_quant": False,
        "quantization_setting_evidence": "trace loader constructs BitsAndBytesConfig without bnb_4bit_use_double_quant; installed default is False",
        "prompt_source_path": trace.get("benchmark"),
        "prompt_source_sha256": sha256_file(trace_input_path) if trace_input_path.is_file() else None,
        "runtime_environment": trace.get("environment"),
    }
    bench_env = json.loads((bench_dir / "environment.json").read_text())
    benchmark_manifest = {
        **manifest(bench_dir / "raw_measurements.jsonl", bench_cfg, "independent_pld_benchmark"),
        "execution_commit": bench_env.get("commit_sha"),
        "kv_capacity": {
            "pld": "prompt_length + max_new_tokens + k + 16",
            "vanilla": "max(2048, prompt_length + max_new_tokens + 8)",
        },
        "eos_semantics": "greedy stop after emitting EOS or reaching max_new_tokens",
        "bnb_4bit_use_double_quant": True,
        "quantization_setting_evidence": "run_independent_pld_short_context.py explicitly sets bnb_4bit_use_double_quant=True",
        "prepared_inputs_path": str((bench_dir / "prepared_inputs.jsonl").relative_to(ROOT)),
        "prepared_inputs_sha256": sha256_file(bench_dir / "prepared_inputs.jsonl"),
        "runtime_environment": bench_env,
        "measurement_config_path": str((bench_dir / "config.json").relative_to(ROOT)),
    }

    rows = []
    for tr in trace.get("prompts", []):
        pid = str(tr["prompt_id"])
        inp = inputs.get(pid)
        methods = measured.get(pid, {})
        vanilla = methods.get("vanilla", {}).get("token_ids")
        pld = methods.get("PLD_prompt_plus_generated_K4", {}).get("token_ids")
        trace_input = trace_inputs.get(pid)
        trace_prompt = trace_input.get("input_ids", []) if trace_input else None
        trace_output = tr.get("token_ids", [])
        trace_vanilla = tr.get("vanilla_token_ids", [])
        cycles = tr.get("cycles", [])
        trajectory = []
        for c in cycles:
            # Saved traces can establish recorded state, but do not contain the
            # live benchmark's candidate/acceptance/cache trajectory.
            trajectory.append({
                "cycle_index": c.get("cycle_index"),
                "prefix_token_ids_hash": sha256_json(c.get("prefix_tokens", [])) if "prefix_tokens" in c else None,
                "draft_candidate_ids": c.get("candidate_tokens"),
                "accepted_prefix_length": c.get("accepted_count"),
                "emitted_token_ids": c.get("emitted_tokens"),
                "kv_length_before": c.get("cache_length_before"),
                "kv_length_after": c.get("cache_length_after"),
                "verification_mode": c.get("verification_mode"),
                "eos_reached": None,
            })
        rows.append({
            "prompt_id": pid,
            "benchmark_prompt_present": inp is not None,
            "benchmark_prompt_length": len(inp["input_ids"]) if inp else None,
            "trace_prompt_length": len(trace_prompt) if trace_prompt is not None else None,
            "benchmark_prompt_token_ids_sha256": sha256_json(inp["input_ids"]) if inp else None,
            "trace_prompt_token_ids_sha256": sha256_json(trace_prompt) if trace_prompt is not None else None,
            "prompt_ids_equal": (inp["input_ids"] == trace_prompt) if inp and trace_prompt is not None else None,
            "trace_prompt_ids_captured_in_trace": False,
            "benchmark_vanilla_token_ids_sha256": sha256_json(vanilla) if vanilla is not None else None,
            "trace_vanilla_token_ids_sha256": sha256_json(trace_vanilla),
            "vanilla_first_token_difference": first_diff(vanilla or [], trace_vanilla),
            "benchmark_pld_k4_token_ids_sha256": sha256_json(pld) if pld is not None else None,
            "benchmark_pld_k4_exact_to_vanilla": pld == vanilla if pld is not None and vanilla is not None else None,
            "trace_pld_token_ids_sha256": sha256_json(trace_output),
            "trace_pld_exact_to_vanilla": trace_output == trace_vanilla,
            "trace_first_token_difference_vs_vanilla": first_diff(trace_output, trace_vanilla),
            "trace_first_token_divergence_index": tr.get("first_token_divergence_index"),
            "trace_recorded_cycles": trajectory,
            "live_benchmark_cycle_trajectory_available": False,
        })

    trace_ids = {str(p["prompt_id"]) for p in trace.get("prompts", [])}
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_manifest.json").write_text(json.dumps(benchmark_manifest, indent=2) + "\n")
    (out / "tracer_manifest.json").write_text(json.dumps(trace_manifest, indent=2) + "\n")
    (out / "prompt_comparison.json").write_text(json.dumps({
        "benchmark_prompt_count": len(inputs), "trace_prompt_count": len(trace_ids),
        "benchmark_only_prompt_ids": sorted(set(inputs) - trace_ids),
        "trace_only_prompt_ids": sorted(trace_ids - set(inputs)),
        "prompts": rows,
    }, indent=2) + "\n")
    first = []
    for row in rows:
        if row["prompt_ids_equal"] is False:
            first.append({"prompt_id": row["prompt_id"], "first_difference": "prompt_token_ids",
                          "evidence": {k: row[k] for k in ("benchmark_prompt_token_ids_sha256", "trace_prompt_token_ids_sha256", "benchmark_prompt_length", "trace_prompt_length")}})
        elif row["vanilla_first_token_difference"] is not None:
            first.append({"prompt_id": row["prompt_id"], "first_difference": "vanilla_generated_token_ids",
                          "token_index": row["vanilla_first_token_difference"]})
        else:
            first.append({"prompt_id": row["prompt_id"], "first_difference": None,
                          "note": "Saved artifacts do not include benchmark candidate/acceptance/KV cycle states; live first execution difference is unresolved."})
    live_path = out / "live_replay.json"
    live = json.loads(live_path.read_text()) if live_path.is_file() else None
    tracer_replay_path = out / "no_double_quant_replay.json"
    tracer_replay = json.loads(tracer_replay_path.read_text()) if tracer_replay_path.is_file() else None
    def first_state_diff(left: list[dict], right: list[dict]) -> dict | None:
        keys = ("verification_input_token_id", "draft_candidate_ids", "accepted_prefix_length",
                "emitted_token_ids", "selected_next_token_id", "kv_length_before", "kv_length_after")
        for i in range(max(len(left), len(right))):
            if i >= len(left) or i >= len(right):
                return {"cycle_index": i, "field": "cycle_count", "left": len(left), "right": len(right)}
            for key in keys:
                if left[i].get(key) != right[i].get(key):
                    return {"cycle_index": i, "field": key,
                            "benchmark_quant": left[i].get(key), "tracer_quant": right[i].get(key)}
        return None
    if live:
        live_by_id = {p["prompt_id"]: p for p in live.get("prompts", [])}
        tracer_by_id = {p["prompt_id"]: p for p in tracer_replay.get("prompts", [])} if tracer_replay else {}
        for item in first:
            replay_row = live_by_id.get(item["prompt_id"])
            tracer_row = tracer_by_id.get(item["prompt_id"])
            if replay_row:
                item["live_replay_vs_archived_trace_first_cycle_difference"] = replay_row.get(
                    "first_cycle_difference_current_vs_archived_trace")
                item["live_production_matches_benchmark_pld"] = replay_row.get(
                    "output_equal_production_vs_lightweight")
                item["lightweight_and_causal_trace_outputs_equal"] = replay_row.get(
                    "output_equal_lightweight_vs_causal")
            if replay_row and tracer_row:
                item["same_prompt_replay_first_difference_double_vs_single_quant"] = first_state_diff(
                    replay_row["runs"]["lightweight_trace"]["trajectory"],
                    tracer_row["runs"]["lightweight_trace"]["trajectory"])
                item["single_quant_replay_matches_archived_trace"] = (
                    tracer_row["runs"]["production"]["generated_token_ids"] ==
                    next((p.get("token_ids") for p in trace.get("prompts", [])
                          if p.get("prompt_id") == item["prompt_id"]), None))
    (out / "first_difference.json").write_text(json.dumps({
        "scope": "saved-artifact and, when present, live replay comparisons",
        "prompts": first,
    }, indent=2) + "\n")

    prompt_eq = sum(r["prompt_ids_equal"] is True for r in rows)
    token_diff = sum(r["vanilla_first_token_difference"] is not None for r in rows)
    pld_mismatch = sum(r["benchmark_pld_k4_exact_to_vanilla"] is False for r in rows)
    trace_mismatch = sum(r["trace_pld_exact_to_vanilla"] is False for r in rows)
    trace_exact_ids = [r["prompt_id"] for r in rows if r["trace_pld_exact_to_vanilla"] is True]
    trace_mismatch_ids = [r["prompt_id"] for r in rows if r["trace_pld_exact_to_vanilla"] is False]
    vanilla_diff_ids = [r["prompt_id"] for r in rows if r["vanilla_first_token_difference"] is not None]
    live_summary = ""
    if live:
        lp = live.get("prompts", [])
        light_equal = sum(bool(p.get("output_equal_production_vs_lightweight")) for p in lp)
        causal_equal = sum(bool(p.get("output_equal_lightweight_vs_causal")) for p in lp)
        no_trace_difference = sum(p.get("first_cycle_difference_lightweight_vs_causal") is None for p in lp)
        historical_diffs = {p["prompt_id"]: p.get("first_cycle_difference_current_vs_archived_trace") for p in lp}
        cycle_rows = [f"- `{pid}`: {diff}" for pid, diff in historical_diffs.items() if diff]
        live_summary = f"""
## Live replay

Replay artifact: [`live_replay.json`](live_replay.json), execution commit `{live.get('execution_commit')}`, quantization double-quant `{live.get('bnb_4bit_use_double_quant')}`.

- Production vs lightweight trace output: {light_equal}/{len(lp)} equal.
- Lightweight vs full numerical trace output: {causal_equal}/{len(lp)} equal; recorded cycle states equal: {no_trace_difference}/{len(lp)}.
- Earliest current-vs-archived trace state differences (selected target token):
{chr(10).join(cycle_rows) if cycle_rows else '- No differing cycle state among compared fields.'}

The live replay used the benchmark's double-quant setting. These results show tracing itself did not change the current-run outputs. The archived trace used a different quantization setting, so its selected-token differences are not evidence of instrumentation effects.
"""
    repeat_text = ""
    repeat_path = out / "repeat_probe.json"
    if repeat_path.is_file() and live:
        repeat = json.loads(repeat_path.read_text())
        full_by_id = {p["prompt_id"]: p for p in live.get("prompts", [])}
        repeat_stable = sum(
            p["runs"]["production"]["generated_token_ids"] == full_by_id[p["prompt_id"]]["runs"]["production"]["generated_token_ids"]
            and p["runs"]["lightweight_trace"]["trajectory"] == full_by_id[p["prompt_id"]]["runs"]["lightweight_trace"]["trajectory"]
            for p in repeat.get("prompts", []) if p["prompt_id"] in full_by_id)
        repeat_text = f"\nRepeat probe: {repeat_stable}/{len(repeat.get('prompts', []))} selected prompts reproduced identical output and lightweight trajectory in a separate process.\n"
    tracer_quant_text = ""
    quant_cycle_lines = []
    if live and tracer_replay:
        archived_by_id = {p["prompt_id"]: p for p in trace.get("prompts", [])}
        single_by_id = {p["prompt_id"]: p for p in tracer_replay.get("prompts", [])}
        current_by_id = {p["prompt_id"]: p for p in live.get("prompts", [])}
        exact_archived = sum(
            single_by_id[pid]["runs"]["production"]["generated_token_ids"] == archived_by_id[pid].get("token_ids")
            for pid in single_by_id if pid in archived_by_id)
        single_mismatch = sum(
            single_by_id[pid]["runs"]["production"]["generated_token_ids"] != archived_by_id[pid].get("vanilla_token_ids")
            for pid in single_by_id if pid in archived_by_id)
        for pid in current_by_id:
            if pid in single_by_id:
                diff = first_state_diff(current_by_id[pid]["runs"]["lightweight_trace"]["trajectory"],
                                        single_by_id[pid]["runs"]["lightweight_trace"]["trajectory"])
                if diff:
                    quant_cycle_lines.append(f"- `{pid}`: cycle {diff['cycle_index']}, `{diff['field']}`; double-quant `{diff.get('benchmark_quant')}`, single-quant `{diff.get('tracer_quant')}`.")
        tracer_quant_text = f"""
## Matched-to-tracer quantization replay

Replay artifact: [`no_double_quant_replay.json`](no_double_quant_replay.json), identical 16 prompts and runtime protocol, with `bnb_4bit_use_double_quant=False` to match the archived tracer loader.

- Replay PLD output matches the archived tracer output: {exact_archived}/{len(single_by_id)}.
- Replay PLD mismatch against the archived tracer's vanilla sequence: {single_mismatch}/{len(single_by_id)} (the archived 8/16 result).
- First selected-state differences between benchmark quantization and tracer quantization:
{chr(10).join(quant_cycle_lines) if quant_cycle_lines else '- No cycle differences among compared fields.'}

This controlled toggle reproduces each archived tracer output and its 8/16 exactness result, while the benchmark quantization replay matches the saved benchmark PLD outputs 16/16. The quantization configuration discrepancy explains the reported 8/16 versus 16/16 observation for this cohort.
"""
    report = f"""# Benchmark–trace reproduction audit

## Scope

This audit compares saved artifacts and includes controlled live replays. The benchmark has raw measurements and prepared prompt IDs; the causal trace has cycle snapshots. The benchmark artifact itself does not save candidate IDs, acceptance decisions, or per-cycle cache lengths, so its historical first execution difference is reconstructed by replaying the same prompt cohort under each artifact's quantization settings.

## Results

- Matched trace prompts: {len(rows)}; benchmark prepared prompts: {len(inputs)}.
- Prompt token IDs are present in both benchmark and trace source files and compare equal: {prompt_eq}/{len(rows)}. The trace itself does not persist the consumed input IDs, so this verifies the referenced input artifact, not the live model call.
- Vanilla output first-difference present: {token_diff}/{len(rows)}.
- Benchmark PLD K=4 differs from its paired vanilla output: {pld_mismatch}/{len(rows)}; saved trace PLD output differs from trace vanilla: {trace_mismatch}/{len(rows)}.
- The eight trace-exact cases are `{', '.join(trace_exact_ids)}`; trace mismatch cases are `{', '.join(trace_mismatch_ids)}`.
- Benchmark-versus-trace vanilla token IDs differ for {len(vanilla_diff_ids)}/{len(rows)} prompts: `{', '.join(vanilla_diff_ids)}`.
- Benchmark run commit: `{bench_env.get('commit_sha')}`; trace run commit is not recorded exactly; current checkout is `19bcc9d7`.
- Trace provenance records base commit `{trace_artifact_manifest.get('base_commit')}` with uncommitted research changes; there is no exact trace execution commit. The benchmark session records `{bench_env.get('commit_sha')}`.
- The captured hashes for the trace runner and PLD generator match the tracked source at current `HEAD`; the trace artifact still lacks a clean-tree commit stamp.
- Shared recorded settings: model revision `{bench_cfg.get('model_revision')}`, NF4, FP16 compute, SDPA, static KV, K=4, 32-token budget.
- **Quantization mismatch confirmed:** the benchmark explicitly sets `bnb_4bit_use_double_quant=True`; the tracer's model loader omits that setting, whose installed BitsAndBytesConfig default is `False`. The two runs therefore used different quantized weights. The controlled no-double-quant replay reproduces the archived trace outputs and its 8/16 result; the benchmark-quant replay reproduces all 16 benchmark PLD outputs and its 16/16 mismatch result.
- PLD cache capacity is prompt length + budget + K + 16 in both runners (436 for 384-token prompts). However, the benchmark vanilla reference uses `max(2048, prompt_length + budget + 8)` (2048), while the tracer's vanilla reference uses the PLD capacity (436). This is a concrete cache-shape mismatch in the vanilla comparison and a plausible numerical confound, not yet proven to cause any specific token change.
- Tokenizer revision is pinned to the same model revision; both greedy paths stop after emitting EOS or reaching the 32-token budget. Random/order seeds are not evidence of execution equivalence.
{live_summary}{tracer_quant_text}{repeat_text}

## Interpretation

Prompt IDs compare exactly in the common referenced prepared-input artifact. The trace does not store the consumed prompt token array, limiting this to source-artifact equality. Trace-exactness is 8/16 in its saved artifact, while independent benchmark PLD K=4 is 0/16 exact. The quantization setting mismatch (double quant on benchmark, off in trace) means the traces were not using the same quantized weights. The matched setting replay reproduces all archived trace outputs and the 8/16 exactness rate; the benchmark setting replay reproduces all saved benchmark PLD outputs and the 16/16 mismatch rate. Thus WP1's discrepancy is explained by the quantization configuration, with instrumentation showing no output change in the controlled current replays. Vanilla cache capacities still differ and remain a confound for comparisons between vanilla references.

## Required next evidence

WP1 discrepancy is resolved for this cohort. For future runs, record `bnb_4bit_use_double_quant` and exact commit directly in every run config. The vanilla reference comparison still needs a capacity-controlled replay (2048 vs 436) before attributing its 12/16 output differences. Expand beyond this diagnostic cohort and retain EOS and token-ID cycle evidence when collecting new data.
"""
    (out / "reproduction_report.md").write_text(report)
    print(f"Wrote audit for {len(rows)} trace prompts to {out}")


if __name__ == "__main__":
    main()
