#!/usr/bin/env python3
"""Execute a locked cross-model NF4 shape-sensitivity protocol."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from run_b1_block_pilot import runtime_identity
from probe_quantized_linear_shape import (
    compare,
    run_variant,
    singleton_linear_rows,
)
from run_kv_factorial_intervention import restore_static_cache, snapshot_cache


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha256_bytes(protocol_path.read_bytes())
    cohort_path = Path(protocol["cohort_file"])
    cohort_sha = sha256_bytes(cohort_path.read_bytes())
    if cohort_sha != protocol["cohort_sha256"]:
        raise ValueError("cohort file hash does not match locked protocol")
    cohort = load_jsonl(cohort_path)
    prompt_map = {(row["task"], row["id"]): row["prompt"] for row in cohort}
    declared = {(row["task"], row["id"]): row for row in protocol["prompts"]}
    if set(prompt_map) != set(declared):
        raise ValueError("cohort prompt IDs do not match locked protocol")
    for key, prompt in prompt_map.items():
        if sha256_bytes(prompt.encode()) != declared[key]["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch for {key}")

    output_dir = Path(args.output_dir)
    result_dir = output_dir / "results"
    manifest_dir = output_dir / "manifests"
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for model_spec in protocol["models"]:
        slug = model_spec["slug"]
        model_id = model_spec["id"]
        revision = model_spec["revision"]
        pending = []
        for prompt_spec in protocol["prompts"]:
            for m in protocol["M_values"]:
                safe_id = prompt_spec["id"].replace("/", "-")
                output_path = result_dir / f"{slug}__{prompt_spec['task']}__{safe_id}__m{m}.json"
                if not output_path.exists():
                    pending.append((prompt_spec, int(m), output_path))
                else:
                    all_results.append(json.loads(output_path.read_text()))
        if not pending:
            print(f"SKIP {slug}: all locked configurations already have artifacts", flush=True)
            continue

        print(f"LOAD {slug}: {model_id}@{revision}; pending={len(pending)}", flush=True)
        loaded = load_backend("hf_bnb_nf4", model_id, revision, args.device, "sdpa")
        model, tokenizer = loaded.model, loaded.tokenizer
        identity = runtime_identity(model, tokenizer, args.device, package_version)
        manifest = {
            "status": "completed model load for locked development sweep",
            "protocol_sha256": protocol_sha,
            "cohort_sha256": cohort_sha,
            "model": model_id,
            "revision": revision,
            "family": model_spec["family"],
            "backend": loaded.metadata,
            "device": args.device,
            "M_values": protocol["M_values"],
            "runtime_identity": identity,
        }
        manifest_path = manifest_dir / f"{slug}.json"
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            if (prior.get("protocol_sha256") != protocol_sha or
                    prior.get("revision") != revision or
                    prior.get("runtime_identity") != identity):
                raise ValueError(f"existing manifest differs from current run: {manifest_path}")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        max_m = max(protocol["M_values"])
        for prompt_spec, m, output_path in pending:
            key = (prompt_spec["task"], prompt_spec["id"])
            prompt = prompt_map[key]
            try:
                ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
                capacity = int(ids.shape[1]) + max_m + 8
                prefill = TargetKVCache(backend="static", max_capacity=capacity)
                with torch.no_grad():
                    first_logits = model(ids, past_key_values=prefill.cache,
                                         use_cache=True).logits[0, -1]
                source_snapshot = snapshot_cache(prefill.cache)

                candidate_ids = [int(first_logits.argmax())]
                candidate_cache = restore_static_cache(source_snapshot, capacity, args.device)
                with torch.no_grad():
                    for _ in range(max_m - 1):
                        token = torch.tensor([[candidate_ids[-1]]], device=args.device,
                                             dtype=ids.dtype)
                        output = model(token, past_key_values=candidate_cache.cache,
                                       use_cache=True)
                        candidate_ids.append(int(output.logits[0, -1].argmax()))

                candidates = candidate_ids[:m]
                reference_logits, reference_cache = run_variant(
                    model, source_snapshot, candidates, capacity, args.device, sequential=True)
                ordinary_logits, ordinary_cache = run_variant(
                    model, source_snapshot, candidates, capacity, args.device, sequential=False)
                with singleton_linear_rows(model) as linear_counts:
                    normalized_logits, normalized_cache = run_variant(
                        model, source_snapshot, candidates, capacity, args.device, sequential=False)

                report = {
                    "status": "completed",
                    "scope": "single-prefix shape diagnostic; not full-sequence exactness or a universal guarantee",
                    "protocol_sha256": protocol_sha,
                    "cohort_sha256": cohort_sha,
                    "model": model_id,
                    "revision": revision,
                    "family": model_spec["family"],
                    "backend": loaded.metadata,
                    "prompt_id": prompt_spec["id"],
                    "task": prompt_spec["task"],
                    "prompt_sha256": prompt_spec["prompt_sha256"],
                    "prompt_token_count": int(ids.shape[1]),
                    "prompt_token_ids_sha256": sha256_bytes(
                        json.dumps(ids[0].tolist()).encode()),
                    "M": m,
                    "candidate_ids": candidates,
                    "candidate_length": len(candidates),
                    "linear_singleton_counts": linear_counts,
                    "ordinary_batch_vs_sequential": compare(
                        ordinary_logits, ordinary_cache, reference_logits, reference_cache),
                    "singleton_linear_batch_vs_sequential": compare(
                        normalized_logits, normalized_cache, reference_logits, reference_cache),
                    "model_manifest": str(manifest_path),
                }
            except Exception as exc:  # keep failures as locked-protocol outcomes
                report = {
                    "status": "error",
                    "protocol_sha256": protocol_sha,
                    "cohort_sha256": cohort_sha,
                    "model": model_id,
                    "revision": revision,
                    "family": model_spec["family"],
                    "prompt_id": prompt_spec["id"],
                    "task": prompt_spec["task"],
                    "prompt_sha256": prompt_spec["prompt_sha256"],
                    "M": m,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            all_results.append(report)
            print(json.dumps({
                "status": report["status"], "model": model_id,
                "prompt_id": prompt_spec["id"], "M": m,
                "argmax_equal": (None if report["status"] != "completed" else
                    all(report["ordinary_batch_vs_sequential"]["argmax_equal_by_row"])),
                "logit_max_abs": (None if report["status"] != "completed" else
                    max(report["ordinary_batch_vs_sequential"]["per_row_logit_max_abs"])),
                "kv_key_max_abs": (None if report["status"] != "completed" else
                    report["ordinary_batch_vs_sequential"]["kv_key_max_abs"]),
                "artifact": str(output_path),
            }), flush=True)

        del model, tokenizer, loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Include any artifacts from prior resumable invocations.
    all_results = [json.loads(p.read_text()) for p in sorted(result_dir.glob("*.json"))]
    completed = [row for row in all_results if row.get("status") == "completed"]
    summary = {
        "status": "complete" if len(all_results) == protocol["execution_plan"]["configuration_count"]
                  else "partial",
        "protocol_sha256": protocol_sha,
        "cohort_sha256": cohort_sha,
        "expected_configurations": protocol["execution_plan"]["configuration_count"],
        "recorded_configurations": len(all_results),
        "completed_configurations": len(completed),
        "failed_configurations": len(all_results) - len(completed),
        "ordinary_nonzero_logit_drift": sum(
            max(row["ordinary_batch_vs_sequential"]["per_row_logit_max_abs"], default=0) > 0
            for row in completed),
        "ordinary_nonzero_kv_drift": sum(
            row["ordinary_batch_vs_sequential"]["kv_key_max_abs"] > 0 or
            row["ordinary_batch_vs_sequential"]["kv_value_max_abs"] > 0
            for row in completed),
        "ordinary_candidate_argmax_matches": sum(
            sum(row["ordinary_batch_vs_sequential"]["argmax_equal_by_row"])
            for row in completed),
        "ordinary_candidate_argmax_total": sum(
            len(row["ordinary_batch_vs_sequential"]["argmax_equal_by_row"])
            for row in completed),
        "singleton_linear_nonzero_kv_drift": sum(
            row["singleton_linear_batch_vs_sequential"]["kv_key_max_abs"] > 0 or
            row["singleton_linear_batch_vs_sequential"]["kv_value_max_abs"] > 0
            for row in completed),
        "limitations": protocol["decision_rules"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
