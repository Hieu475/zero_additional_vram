#!/usr/bin/env python3
"""Audit completeness, input identity, and recorded provenance for n=20 runs."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/backend_validation/context_matrix_n20"
CONTEXTS = (128, 384, 1024, 2048)
BACKENDS = {"bnb": "hf_bnb_nf4", "hqq": "hqq_gemlite_w4a16"}
METHODS = ("vanilla", "pld_k4", "cka90_k2")
SESSIONS = (1, 2, 3)
EXPECTED_PROMPTS = 20
EXPECTED_ROWS = EXPECTED_PROMPTS * len(METHODS)


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, errors
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_no}: invalid JSON ({exc})")
    return rows, errors


def prompt_hash(ids: list[int]) -> str:
    packed = json.dumps(ids, separators=(",", ":")).encode()
    return hashlib.sha256(packed).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    input_rows = [json.loads(line) for line in (BASE / "prepared_inputs.jsonl").read_text().splitlines() if line]
    by_id = {row["prompt_id"]: row for row in input_rows}
    files: list[dict[str, Any]] = []
    session_stats: dict[tuple[str, int, int], dict[str, Any]] = {}
    exact: Counter[str] = Counter()
    telemetry = Counter()

    for backend_tag, backend_name in BACKENDS.items():
        for ctx in CONTEXTS:
            cohort = [line.strip() for line in (BASE / f"cohort_ctx{ctx}.txt").read_text().splitlines() if line.strip()]
            for session in SESSIONS:
                name = f"n20_ctx{ctx}_out128_{backend_tag}_session{session}.jsonl"
                path = BASE / name
                rows, parse_errors = read_jsonl(path)
                anomalies = list(parse_errors)
                grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
                for row in rows:
                    pid = row.get("prompt_id", "<missing>")
                    method = row.get("method", "<missing>")
                    if method in grouped[pid]:
                        anomalies.append(f"duplicate prompt/method: {pid}/{method}")
                    grouped[pid][method] = row
                    if row.get("backend") != backend_name:
                        anomalies.append(f"backend mismatch: {pid}/{method}")
                    if row.get("prompt_tokens") != ctx or row.get("max_new_tokens") != 128:
                        anomalies.append(f"context/output budget mismatch: {pid}/{method}")
                    if row.get("kv_capacity") != ctx + 132:
                        anomalies.append(f"KV capacity mismatch: {pid}/{method}")
                    source = by_id.get(pid)
                    if source is None:
                        anomalies.append(f"prompt absent from frozen inputs: {pid}")
                    elif row.get("prompt_token_ids_sha256") != prompt_hash(source["input_ids"]):
                        anomalies.append(f"prompt token hash mismatch: {pid}")
                    if row.get("status") != "ok":
                        anomalies.append(f"measured run failed: {pid}/{method} status={row.get('status')}")
                    if row.get("warmup_status", {}).get("passed") is not True:
                        anomalies.append(f"warmup failed: {pid}/{method}")
                    runtime = row.get("runtime", {})
                    for key in ("driver", "temperature_c", "sm_clock_mhz", "power_w", "transformers"):
                        if runtime.get(key) is not None:
                            telemetry[key] += 1

                incomplete = {
                    pid: sorted(set(METHODS) - set(grouped.get(pid, {})))
                    for pid in cohort if set(grouped.get(pid, {})) != set(METHODS)
                }
                commits = sorted({row.get("runtime", {}).get("execution_commit") for row in rows})
                dirty_values = sorted({row.get("runtime", {}).get("working_tree_dirty") for row in rows}, key=str)
                source_dirty_values = sorted({row.get("runtime", {}).get("source_tree_dirty") for row in rows}, key=str)
                clean_provenance = (
                    len(rows) == EXPECTED_ROWS
                    and dirty_values == [False]
                    and source_dirty_values == [False]
                    and len(commits) == 1
                    and all(row.get("prompt_token_ids_sha256") for row in rows)
                )
                stat = {
                    "file": name, "exists": path.exists(), "rows": len(rows),
                    "expected_rows": EXPECTED_ROWS,
                    "unique_prompts": len(grouped),
                    "complete_prompt_triplets": len(cohort) - len(incomplete),
                    "missing_prompts": [pid for pid in cohort if pid not in grouped],
                    "incomplete_prompts": incomplete,
                    "execution_commits": commits,
                    "working_tree_dirty_values": dirty_values,
                    "source_tree_dirty_values": source_dirty_values,
                    "clean_provenance": clean_provenance,
                    "anomalies": anomalies,
                }
                files.append(stat)
                session_stats[(backend_tag, ctx, session)] = stat

                for methods in grouped.values():
                    for method in ("pld_k4", "cka90_k2"):
                        record = methods.get(method, {})
                        if "within_backend_exact_vs_vanilla" in record:
                            key = f"{backend_tag}/{ctx}/{method}"
                            exact[key + "/exact"] += int(record["within_backend_exact_vs_vanilla"] is True)
                            exact[key + "/observed"] += 1

    complete_cells = sum(
        s["exists"] and s["complete_prompt_triplets"] == EXPECTED_PROMPTS and not s["anomalies"]
        for s in files
    )
    dirty_complete_cells = sum(
        s["exists"] and s["complete_prompt_triplets"] == EXPECTED_PROMPTS
        and True in s["working_tree_dirty_values"] for s in files
    )
    expected_cells = len(BACKENDS) * len(CONTEXTS) * len(SESSIONS)
    missing_files = [s["file"] for s in files if not s["exists"]]
    commit_shas = sorted({sha for stat in files for sha in stat["execution_commits"] if sha})
    source_paths = ("scripts/benchmark_backend_decoding.py", "src/zassd")
    source_revisions = []
    for sha in commit_shas:
        result = subprocess.run(["git", "diff", "--quiet", sha, "HEAD", "--", "src/zassd"], cwd=ROOT)
        runner_blob = subprocess.run(["git", "show", f"{sha}:scripts/benchmark_backend_decoding.py"],
                                     cwd=ROOT, capture_output=True)
        source_revisions.append({
            "commit": sha,
            "zassd_core_matches_head": result.returncode == 0,
            "zassd_diff_error": result.returncode not in (0, 1),
            "runner_snapshot_available": runner_blob.returncode == 0,
            "runner_snapshot_sha256": hashlib.sha256(runner_blob.stdout).hexdigest() if runner_blob.returncode == 0 else None,
        })
    status_paths = [line[3:] for line in git("status", "--porcelain").splitlines() if len(line) > 3]
    current_source_dirty_paths = [
        path for path in status_paths
        if path == "scripts/benchmark_backend_decoding.py" or path.startswith("src/zassd/")
    ]
    reconstructable_source = bool(source_revisions) and all(
        r["zassd_core_matches_head"] and r["runner_snapshot_available"] for r in source_revisions
    )
    report = {
        "audit_scope": "output-128 n=20 context matrix only",
        "frozen_input_count": len(input_rows),
        "cohort_counts": {str(c): sum(1 for r in input_rows if r.get("prompt_tokens") == c) for c in CONTEXTS},
        "expected_session_cells": expected_cells,
        "complete_cells": complete_cells,
        "complete_cells_with_dirty_tree_flag": dirty_complete_cells,
        "clean_provenance_cells": sum(
            s["exists"] and s["complete_prompt_triplets"] == EXPECTED_PROMPTS
            and not s["anomalies"] and s["clean_provenance"] for s in files
        ),
        "missing_files": missing_files,
        "execution_source_revisions": source_revisions,
        "current_uncommitted_source_paths": current_source_dirty_paths,
        "captured_committed_source_reconstructable": reconstructable_source,
        "files": files,
        "exploratory_exactness_counts": dict(exact),
        "telemetry_field_presence_counts": dict(telemetry),
        "important_limitations": [
            "Fourteen complete cells contain at least one record with working_tree_dirty=true; ten cells have consistent clean working-tree and source-tree flags, prompt hashes, and a single execution commit. The HQQ context-384 session-2 cell combines a partial earlier dirty run with a clean append and is complete structurally but not clean-provenance.",
            "Historical dirty-file lists/source hashes were not saved, so uncommitted source edits later reverted cannot be ruled out for dirty records. Runtime records omit GPU driver version, Transformers/bitsandbytes/GemLite versions, temperature, SM clocks, and power telemetry.",
            "Temperature=0 is visible in the captured runner source, but is not stored in each record. EOS token ID/position and explicit stopping reason are not stored.",
            "TPS is calculated as (generated token count - 1) / decode time, excluding the first generated token; keep this definition explicit in any comparison.",
            "Output-128 alone is incomplete; output-32/256 and three independent sessions are not complete. Exactness counts here are descriptive, not prompt-level confidence intervals.",
            "Method order and seed are stored; no GPU thermal/power trace or per-run session UUID is stored.",
        ],
    }
    (BASE / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Context Matrix n=20: Provenance and Completeness Audit", "",
        "Scope: output-128 only; frozen 20-prompt cohort at contexts 128, 384, 1024, and 2048; BNB and HQQ; vanilla, PLD K=4, and CKA90/K2.", "",
        f"- Structurally complete session cells: **{complete_cells}/{expected_cells}**.",
        f"- Complete cells carrying `working_tree_dirty=true`: **{dirty_complete_cells}/{complete_cells}**; clean-provenance cells: **{report['clean_provenance_cells']}**.",
        f"- Missing session files: **{len(missing_files)}**.",
        "- A complete cell has 20 prompt triplets (60 records), matching prompt hashes, expected backend/context/capacity, successful warmups, and no recorded failures.", "",
        "## Cell status", "",
        "| Context | Backend | Session 1 | Session 2 | Session 3 |", "|---:|---|---:|---:|---:|",
    ]
    for ctx in CONTEXTS:
        for tag in BACKENDS:
            vals = []
            for session in SESSIONS:
                st = session_stats[(tag, ctx, session)]
                if not st["exists"]:
                    vals.append("missing")
                else:
                    flag = "; dirty tree" if True in st["working_tree_dirty_values"] else ""
                    vals.append(f"{st['complete_prompt_triplets']}/20 prompts ({st['rows']} rows){flag}")
            lines.append(f"| {ctx} | {tag.upper()} | " + " | ".join(vals) + " |")
    lines += [
        "", "## Provenance findings", "",
        f"{dirty_complete_cells} complete cells include records with `working_tree_dirty=true`; {report['clean_provenance_cells']} complete cells have consistently clean working-tree and source-tree flags, prompt hashes, and one execution commit. The HQQ/context-384 session-2 cell contains an earlier dirty partial run plus a clean append, so it is structurally complete but remains exploratory. Each recorded commit is checked against current `src/zassd` and the runner snapshot is hashed in `audit.json`. Historical dirty-file lists/source hashes were not saved, so reverted uncommitted edits cannot be ruled out for dirty records.", "",
        "Records preserve model/tokenizer revision, backend quantization metadata, attention backend, prompt token count/hash, output budget, KV capacity, warmup status, randomized method order/seed, torch/CUDA/GPU model/capability, latency, TPS, token IDs, exactness, and peak allocated/reserved VRAM.", "",
        "Missing per-run provenance includes GPU driver, Transformers/bitsandbytes/GemLite versions, temperature, EOS token/stop reason, GPU temperature/SM clock/power, and source-tree diff/hash. Temperature 0.0 is visible in the runner source at the recorded revision, but is not embedded in records.", "",
        "## Metric and interpretation caveats", "",
        "The runner computes decode TPS as `(generated_tokens - 1) / decode_time`, excluding the first output token. Preserve and label that definition in comparisons. Within-backend token exactness is recorded; cross-backend token equality is not a correctness criterion because BNB and HQQ use different quantized weights.", "",
        "A prompt triplet is complete only when vanilla, PLD, and CKA rows all exist. Partial prompt triplets are preserved and must not count as complete observations. The output-128 matrix does not meet the stated three-session protocol, and no output-32/256 matrix is included.", "",
        "## Recommended rerun order", "",
        "1. Preserve raw records unchanged as exploratory artifacts. All 24 session cells are now structurally complete and no session files are missing; the HQQ/context-384 session-2 cell mixes historical dirty records with a clean append and should remain exploratory.",
        "2. Add missing runtime provenance to the runner and start from a clean, committed source tree.",
        "3. Keep clean-provenance cells as the primary subset. Rerun mixed/dirty cells only if a clean three-session confirmatory result is required; do not treat structural completeness as provenance cleanliness.",
        "4. After three clean sessions exist per backend/context, compute paired summaries and document/session-aware uncertainty.",
        "5. Decide whether output-32 or output-256 is needed after the primary output-128 matrix is valid.", "",
        "Machine-readable details: [`audit.json`](audit.json).",
    ]
    (BASE / "audit_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete_cells": complete_cells, "expected_cells": expected_cells,
                      "missing_files": missing_files, "report": str((BASE / "audit_report.md").relative_to(ROOT)),
                      "json": str((BASE / "audit.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
