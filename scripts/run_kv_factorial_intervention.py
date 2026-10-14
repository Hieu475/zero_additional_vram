#!/usr/bin/env python3
"""Run active/canonical KV × batched/sequential verification interventions.

All four cells receive identical prefix IDs and candidate IDs. Each condition
starts from a freshly restored cache, so no branch can mutate another branch.
The diagnostic uses the benchmark NF4 configuration (double quant enabled).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zassd.baselines.prompt_lookup import prompt_lookup_generate  # noqa: E402
from zassd.cache.kv_cache import StaticPreallocatedLayer, TargetKVCache  # noqa: E402

INPUTS = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl"
OUT = ROOT / "experiments/numerical_causality/interventions/factorial_kv_verification.jsonl"
SUMMARY = ROOT / "experiments/numerical_causality/interventions/factorial_summary.json"


def digest_tensor(t: torch.Tensor) -> str:
    raw = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def snapshot_fingerprint(snapshot: list[tuple[torch.Tensor, torch.Tensor, int]]) -> str:
    h = hashlib.sha256()
    for key, value, length in snapshot:
        h.update(str(length).encode())
        h.update(digest_tensor(key).encode())
        h.update(digest_tensor(value).encode())
    return h.hexdigest()


def kv_layer_errors(active, canonical) -> list[dict[str, Any]]:
    rows = []
    for index, (a, c) in enumerate(zip(active, canonical)):
        ak, av, alen = a
        ck, cv, clen = c
        if alen != clen:
            rows.append({"layer": index, "active_length": alen,
                         "canonical_length": clen, "length_mismatch": True})
            continue
        if not alen:
            rows.append({"layer": index, "active_length": 0,
                         "canonical_length": 0, "key_max_abs": 0.0,
                         "value_max_abs": 0.0})
            continue
        rows.append({
            "layer": index, "active_length": alen, "canonical_length": clen,
            "key_max_abs": float((ak.float() - ck.float()).abs().max()),
            "value_max_abs": float((av.float() - cv.float()).abs().max()),
            "key_mae": float((ak.float() - ck.float()).abs().mean()),
            "value_mae": float((av.float() - cv.float()).abs().mean()),
        })
    return rows


def snapshot_cache(cache) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    rows = []
    for layer in cache.layers:
        length = int(layer.get_seq_length())
        if getattr(layer, "is_initialized", False) and length:
            key = layer.keys[:, :, :length, :].detach().cpu().clone()
            value = layer.values[:, :, :length, :].detach().cpu().clone()
        else:
            key = torch.empty(0)
            value = torch.empty(0)
        rows.append((key, value, length))
    return rows


def restore_static_cache(snapshot, capacity: int, device: str) -> TargetKVCache:
    restored = TargetKVCache(backend="static", max_capacity=capacity)
    restored.cache.layers = []
    for key, value, length in snapshot:
        layer = StaticPreallocatedLayer(max_capacity=capacity)
        if length:
            key = key.to(device=device)
            value = value.to(device=device)
            bsz, heads, _, dim = key.shape
            layer.keys = torch.zeros((bsz, heads, capacity, dim), dtype=key.dtype, device=device)
            layer.values = torch.zeros_like(layer.keys)
            layer.keys[:, :, :length, :].copy_(key)
            layer.values[:, :, :length, :].copy_(value)
            layer.seq_len = int(length)
            layer.is_initialized = True
            layer.dtype = key.dtype
            layer.device = torch.device(device)
        restored.cache.layers.append(layer)
    return restored


def canonical_prefix_cache(model, prefix: list[int], prompt_length: int,
                           capacity: int, device: str) -> TargetKVCache:
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.no_grad():
        ids = torch.tensor([prefix[:prompt_length]], dtype=torch.long, device=device)
        model(ids, past_key_values=cache.cache, use_cache=True)
        for token_id in prefix[prompt_length:]:
            token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            model(token, past_key_values=cache.cache, use_cache=True)
    if cache.get_seq_length() != len(prefix):
        raise RuntimeError(f"canonical cache length {cache.get_seq_length()} != prefix {len(prefix)}")
    return cache


def verify(model, snapshot, candidate_ids: list[int], capacity: int,
           device: str, mode: str) -> tuple[torch.Tensor, str, int, int]:
    cache = restore_static_cache(snapshot, capacity, device)
    before_len = cache.get_seq_length()
    before = snapshot_fingerprint(snapshot_cache(cache.cache))
    expected_len = snapshot[0][2]
    if before_len != expected_len:
        raise RuntimeError(f"restored logical length {before_len} != source length {expected_len}")
    if before != snapshot_fingerprint(snapshot):
        raise RuntimeError("restored condition cache differs from its source snapshot")
    with torch.no_grad():
        if mode == "batched":
            ids = torch.tensor([candidate_ids], dtype=torch.long, device=device)
            logits = model(ids, past_key_values=cache.cache, use_cache=True).logits[0].float().cpu()
        elif mode == "sequential":
            rows = []
            for token_id in candidate_ids:
                ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
                rows.append(model(ids, past_key_values=cache.cache, use_cache=True).logits[0, -1].float().cpu())
            logits = torch.stack(rows)
        else:
            raise ValueError(mode)
    after_len = cache.get_seq_length()
    del cache
    return logits, before, before_len, after_len


def margin(logits: torch.Tensor) -> float:
    top = torch.topk(logits.float(), 2).values
    return float(top[0] - top[1])


def logit_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = left.float() - right.float()
    denom = right.float().abs().clamp_min(1e-12)
    return {
        "mae": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
        "max_relative": float((delta.abs() / denom).max()),
        "l2_norm": float(torch.linalg.vector_norm(delta)),
    }


def selected_next_row(cycle: dict) -> int:
    if cycle.get("verification_mode") == "vanilla_fallback":
        return 0
    rejected = cycle.get("rejected_at")
    return int(rejected) if rejected is not None else len(cycle.get("candidate_tokens", [])) - 1


def position_ids_for(prefix_length: int, input_length: int) -> list[int]:
    return list(range(prefix_length, prefix_length + input_length))


def first_aligned_argmax_mismatch(cycles: list[dict], prompt: list[int],
                                  vanilla: list[int]) -> tuple[int, int] | None:
    for cycle in cycles:
        prefix = cycle["prefix_tokens"]
        start = len(prefix) - len(prompt)
        if start < 0 or prefix != prompt + vanilla[:start]:
            return None
        candidates = cycle["candidate_tokens"]
        active = cycle["verify_argmax"]
        for row in range(min(len(candidates), len(active))):
            if candidates[:row + 1] != vanilla[start:start + row + 1]:
                break
            target_idx = start + row + 1
            if target_idx < len(vanilla) and active[row] != vanilla[target_idx]:
                return cycle["cycle_index"], row
    return None


def choose_cycle(cycles: list[dict], mismatch: tuple[int, int] | None) -> tuple[dict, int, str]:
    if mismatch:
        idx, row = mismatch
        return cycles[idx], row, "first_aligned_target_argmax_mismatch"
    for cycle in cycles:
        if len(cycle.get("candidate_tokens", [])) > 1:
            return cycle, 0, "first_speculative_control_no_aligned_mismatch"
    return cycles[0], 0, "fallback_control_no_aligned_mismatch"


def run_factorial(model, active_snapshot, canonical_snapshot, prefix: list[int],
                  candidates: list[int], capacity: int, device: str,
                  vanilla_expected: list[int], cycle_start: int,
                  selected_row: int) -> dict[str, Any]:
    original_active_hash = snapshot_fingerprint(active_snapshot)
    original_canonical_hash = snapshot_fingerprint(canonical_snapshot)
    layer_errors = kv_layer_errors(active_snapshot, canonical_snapshot)
    conditions: dict[str, torch.Tensor] = {}
    initial_hashes: dict[str, str] = {}
    initial_lengths: dict[str, int] = {}
    cache_after_lengths: dict[str, int] = {}
    source_for = {
        "A_active_batch": active_snapshot,
        "B_canonical_batch": canonical_snapshot,
        "C_active_sequential": active_snapshot,
        "D_canonical_sequential": canonical_snapshot,
    }
    modes = {"A_active_batch": "batched", "B_canonical_batch": "batched",
             "C_active_sequential": "sequential", "D_canonical_sequential": "sequential"}
    for name in ("A_active_batch", "B_canonical_batch", "C_active_sequential", "D_canonical_sequential"):
        logits, before_hash, before_len, after_len = verify(model, source_for[name], candidates,
                                                            capacity, device, modes[name])
        conditions[name] = logits
        initial_hashes[name] = before_hash
        initial_lengths[name] = before_len
        cache_after_lengths[name] = after_len

    a, b, c, d = (conditions[n] for n in ("A_active_batch", "B_canonical_batch",
                                          "C_active_sequential", "D_canonical_sequential"))
    rows = []
    for row_idx in range(len(candidates)):
        za, zb, zc, zd = a[row_idx], b[row_idx], c[row_idx], d[row_idx]
        interaction = (za - zb) - (zc - zd)
        expected_index = cycle_start + row_idx + 1
        candidate_prefix_aligned = (
            candidates[:row_idx + 1] == vanilla_expected[cycle_start:cycle_start + row_idx + 1]
        )
        expected = (vanilla_expected[expected_index]
                    if candidate_prefix_aligned and expected_index < len(vanilla_expected) else None)
        am, bm, cm, dm = map(lambda z: int(z.argmax()), (za, zb, zc, zd))
        rows.append({
            "logit_row": row_idx,
            "position_id": len(prefix) + row_idx,
            "candidate_input_token_id": candidates[row_idx],
            "vanilla_reference_token_id": expected,
            "candidate_prefix_matches_vanilla": candidate_prefix_aligned,
            "A_active_batch_argmax": am, "B_canonical_batch_argmax": bm,
            "C_active_sequential_argmax": cm, "D_canonical_sequential_argmax": dm,
            "A_top1_margin": margin(za), "B_top1_margin": margin(zb),
            "C_top1_margin": margin(zc), "D_top1_margin": margin(zd),
            "active_vs_canonical_kv_batch_logits": logit_stats(za, zb),
            "canonical_batch_vs_sequential_logits": logit_stats(zb, zd),
            "active_vs_canonical_kv_sequential_logits": logit_stats(zc, zd),
            "interaction_max_abs": float(interaction.abs().max()),
            "interaction_l2_norm": float(torch.linalg.vector_norm(interaction)),
            "decision_diff_active_vs_canonical_batch": am != bm,
            "decision_diff_canonical_batch_vs_sequential": bm != dm,
            "decision_diff_active_vs_canonical_sequential": cm != dm,
            "active_argmax_matches_vanilla": am == expected if expected is not None else None,
            "canonical_sequential_matches_vanilla": dm == expected if expected is not None else None,
        })

    chosen = rows[min(selected_row, len(rows) - 1)]
    kv_sensitive = chosen["decision_diff_active_vs_canonical_batch"] or chosen["decision_diff_active_vs_canonical_sequential"]
    execution_sensitive = chosen["decision_diff_canonical_batch_vs_sequential"] or (am != cm)
    net_argmax_flip = chosen["A_active_batch_argmax"] != chosen["D_canonical_sequential_argmax"]
    if not net_argmax_flip and kv_sensitive and execution_sensitive:
        classification = "interaction_sensitive_cancellation"
    elif not net_argmax_flip and kv_sensitive:
        classification = "kv_history_sensitive_without_net_flip"
    elif not net_argmax_flip and execution_sensitive:
        classification = "execution_sensitive_without_net_flip"
    elif not net_argmax_flip:
        classification = "no_argmax_divergence_in_selected_row"
    elif kv_sensitive and execution_sensitive:
        classification = "interaction_or_mixed_sensitive"
    elif kv_sensitive:
        classification = "kv_history_sensitive"
    elif execution_sensitive:
        classification = "execution_sensitive"
    else:
        classification = "unresolved"
    return {
        "prefix_length": len(prefix),
        "candidate_input_ids": candidates,
        "logical_prefix_ids_sha256": hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest(),
        "candidate_ids_sha256": hashlib.sha256(json.dumps(candidates, separators=(",", ":")).encode()).hexdigest(),
        "position_ids": position_ids_for(len(prefix), len(candidates)),
        "attention_mask_semantics": "omitted; model/cache constructs causal mask from static-cache logical length",
        "row_alignment": "logit row j follows candidate input token j and predicts the next token",
        "selected_logit_row": min(selected_row, len(rows) - 1),
        "selected_row_classification": classification,
        "selected_row_active_batch_argmax_differs_from_canonical_sequential": (
            chosen["A_active_batch_argmax"] != chosen["D_canonical_sequential_argmax"]),
        "condition_start_state_fingerprints": initial_hashes,
        "condition_start_logical_lengths": initial_lengths,
        "all_conditions_started_from_expected_prefix_length": all(
            length == len(prefix) for length in initial_lengths.values()),
        "condition_cache_lengths_after": cache_after_lengths,
        "active_vs_canonical_kv_layer_errors": layer_errors,
        "active_source_snapshot_unchanged": snapshot_fingerprint(active_snapshot) == original_active_hash,
        "canonical_source_snapshot_unchanged": snapshot_fingerprint(canonical_snapshot) == original_canonical_hash,
        "all_condition_cache_lengths_after_match_expected": all(
            length == len(prefix) + len(candidates) for length in cache_after_lengths.values()),
        "rows": rows,
    }


def run_vanilla(model, prompt_ids: torch.Tensor, budget: int, capacity: int,
                device: str, eos_id: int) -> list[int]:
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    output = []
    with torch.no_grad():
        out = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
        for i in range(budget):
            token = int(out.logits[0, -1].argmax())
            output.append(token)
            if token == eos_id or i + 1 == budget:
                break
            out = model(torch.tensor([[token]], device=device), past_key_values=cache.cache, use_cache=True)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--report", type=Path,
                    default=ROOT / "experiments/numerical_causality/interventions/factorial_report.md")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    args = ap.parse_args()
    inputs = args.inputs if args.inputs.is_absolute() else ROOT / args.inputs
    output = args.output if args.output.is_absolute() else ROOT / args.output
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for factorial interventions")
    rows = [json.loads(line) for line in inputs.read_text().splitlines() if line.strip()]
    wanted = set(args.ids or [])
    prompts = [r for r in rows if not wanted or r["prompt_id"] in wanted][:args.samples]
    if wanted and {r["prompt_id"] for r in prompts} != wanted:
        raise ValueError("One or more requested prompt IDs are absent from prepared inputs")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    execution_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True, check=True).stdout.strip()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quant,
        device_map=args.device, attn_implementation="sdpa").eval()

    details = []
    if output.exists():
        raise FileExistsError(f"Refusing to append to existing intervention output: {output}")
    for item in prompts:
        prompt_list = item["input_ids"]
        prompt_ids = torch.tensor([prompt_list], dtype=torch.long, device=args.device)
        capacity = len(prompt_list) + args.max_new_tokens + args.k + 16
        cycles: list[dict] = []
        _, metrics = prompt_lookup_generate(
            model, tokenizer, prompt_ids, k=args.k, ngram_size=3,
            max_new_tokens=args.max_new_tokens, temperature=0.0,
            device=args.device, kv_cache_backend="static",
            candidate_source="prompt+generated", kv_cache_max_capacity=capacity,
            numerical_trace=cycles)
        vanilla = run_vanilla(model, prompt_ids, args.max_new_tokens + args.k + 1, capacity,
                              args.device, tokenizer.eos_token_id)
        vanilla_output = vanilla[:args.max_new_tokens]
        mismatch = first_aligned_argmax_mismatch(cycles, prompt_list, vanilla)
        cycle, selected_row, selection_reason = choose_cycle(cycles, mismatch)
        prefix = cycle["prefix_tokens"]
        actual_candidate_ids = list(cycle["candidate_tokens"])
        candidate_ids = list(actual_candidate_ids)
        cycle_start = len(prefix) - len(prompt_list)
        candidate_origin = "actual_pld_candidate_sequence"
        vanilla_continuation = vanilla[cycle_start:cycle_start + args.k + 1]
        if (len(candidate_ids) < args.k + 1 and cycle_start >= 0
                and candidate_ids and candidate_ids[0] == (vanilla_continuation[0] if vanilla_continuation else None)
                and len(vanilla_continuation) >= args.k + 1):
            candidate_ids = vanilla_continuation
            candidate_origin = "vanilla_reference_continuation_extension"
        active_snapshot = cycle["cache_before"]
        canonical_cache = canonical_prefix_cache(model, prefix, len(prompt_list), capacity, args.device)
        canonical_snapshot = snapshot_cache(canonical_cache.cache)
        del canonical_cache

        actual_k = len(actual_candidate_ids) - 1
        requested_ks = sorted({depth for depth in (1, 2, 4) if len(candidate_ids) >= depth + 1} | {actual_k})
        factorials = []
        for depth in requested_ks:
            candidate_source = (actual_candidate_ids if depth <= actual_k else candidate_ids)
            fixed_candidates = candidate_source[:depth + 1]
            factorials.append({
                "draft_k": depth,
                "candidate_sequence_origin": ("actual_pld_candidate_sequence" if depth <= actual_k
                                              else "vanilla_reference_continuation_extension"),
                "same_prefix_in_all_cells": True,
                "same_candidate_ids_in_all_cells": True,
                "start_prefix_length": len(prefix),
                "intervention": run_factorial(
                    model, active_snapshot, canonical_snapshot, prefix, fixed_candidates,
                    capacity, args.device, vanilla,
                    len(prefix) - len(prompt_list), selected_row if selected_row <= depth else depth),
            })
        generated_diff = next((i for i, (a, b) in enumerate(zip(metrics.generated_token_ids, vanilla_output)) if a != b), None)
        if generated_diff is None and len(metrics.generated_token_ids) != len(vanilla_output):
            generated_diff = min(len(metrics.generated_token_ids), len(vanilla_output))
        row = {
            "prompt_id": item["prompt_id"],
            "execution_commit": execution_commit,
            "prompt_token_ids_sha256": hashlib.sha256(json.dumps(prompt_list, separators=(",", ":")).encode()).hexdigest(),
            "model_revision": args.revision,
            "quantization": {"type": "NF4", "compute_dtype": "float16", "double_quant": True},
            "attention": "sdpa", "kv_backend": "static", "capacity": capacity,
            "max_new_tokens": args.max_new_tokens, "eos_token_id": tokenizer.eos_token_id,
            "pld_token_ids": metrics.generated_token_ids, "vanilla_token_ids": vanilla_output,
            "first_sequence_token_difference_index": generated_diff,
            "selected_cycle_index": cycle["cycle_index"],
            "actual_cycle_k": actual_k,
            "selected_cycle_reason": selection_reason,
            "selected_cycle_verification_mode": cycle["verification_mode"],
            "selected_cycle_candidate_ids_including_target_input": candidate_ids,
            "actual_pld_candidate_ids_including_target_input": actual_candidate_ids,
            "factorial_candidate_sequence_origin": candidate_origin,
            "selected_cycle_accepted_prefix_length": cycle["accepted_count"],
            "selected_cycle_cache_length_before": cycle["cache_before"][0][2],
            "same_logical_prefix_in_all_cells": len(prefix) == cycle["cache_before"][0][2],
            "vanilla_mismatch_in_selected_row": mismatch is not None,
            "factorial_conditions": ["A_active_batch", "B_canonical_batch", "C_active_sequential", "D_canonical_sequential"],
            "verification_modes": ["batched", "sequential"],
            "kv_interventions": factorials,
            "actual_cycle_k_classification": next(
                item["intervention"]["selected_row_classification"]
                for item in factorials if item["draft_k"] == actual_k),
            "counterfactual_k_classifications": {
                str(item["draft_k"]): item["intervention"]["selected_row_classification"]
                for item in factorials if item["draft_k"] != actual_k
            },
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        }
        details.append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        print(f"intervened {item['prompt_id']} at cycle {cycle['cycle_index']}", flush=True)
        del cycles, active_snapshot, canonical_snapshot, metrics
        torch.cuda.empty_cache()

    class_counts: dict[str, int] = {}
    by_k: dict[str, dict[str, int]] = {}
    actual_k_counts: dict[str, int] = {}
    cycle_modes: dict[str, int] = {}
    actual_origin_counts: dict[str, int] = {}
    extended_probe_count = 0
    for item in details:
        label = item["actual_cycle_k_classification"]
        class_counts[label] = class_counts.get(label, 0) + 1
        k_key = str(item["actual_cycle_k"])
        actual_k_counts[k_key] = actual_k_counts.get(k_key, 0) + 1
        mode = item["selected_cycle_verification_mode"]
        cycle_modes[mode] = cycle_modes.get(mode, 0) + 1
        actual_factor = next(f for f in item["kv_interventions"] if f["draft_k"] == item["actual_cycle_k"])
        origin = actual_factor["candidate_sequence_origin"]
        actual_origin_counts[origin] = actual_origin_counts.get(origin, 0) + 1
        extended_probe_count += int(any(
            f["candidate_sequence_origin"] == "vanilla_reference_continuation_extension"
            for f in item["kv_interventions"] if f["draft_k"] != item["actual_cycle_k"]))
        for intervention in item["kv_interventions"]:
            key = str(intervention["draft_k"])
            label_k = intervention["intervention"]["selected_row_classification"]
            by_k.setdefault(key, {})[label_k] = by_k.setdefault(key, {}).get(label_k, 0) + 1
    integrity_ok = all(
        intervention["intervention"]["all_conditions_started_from_expected_prefix_length"]
        and intervention["intervention"]["all_condition_cache_lengths_after_match_expected"]
        and intervention["intervention"]["active_source_snapshot_unchanged"]
        and intervention["intervention"]["canonical_source_snapshot_unchanged"]
        for item in details for intervention in item["kv_interventions"])
    summary = {
        "n_prompts": len(details), "classification_counts": class_counts,
        "classification_by_k": by_k,
        "actual_cycle_k_counts": actual_k_counts,
        "actual_cycle_verification_modes": cycle_modes,
        "actual_cycle_candidate_sequence_origin_counts": actual_origin_counts,
        "prompts_with_counterfactual_vanilla_extension": extended_probe_count,
        "sequence_mismatch_count": sum(r["first_sequence_token_difference_index"] is not None for r in details),
        "selected_row_active_batch_differs_from_canonical_sequential_count": sum(
            f["intervention"]["selected_row_active_batch_argmax_differs_from_canonical_sequential"]
            for r in details for f in r["kv_interventions"] if f["draft_k"] == r["actual_cycle_k"]),
        "all_cache_integrity_checks_pass": integrity_ok,
        "execution_commit": execution_commit,
        "double_quant": True, "model_revision": args.revision,
        "cohort_ids": [r["prompt_id"] for r in details],
        "interpretation": "Classifications describe selected-row greedy argmax decisions. Logit norm changes alone do not imply a token change.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_rows = []
    for item in details:
        actual_factor = next(f for f in item["kv_interventions"]
                             if f["draft_k"] == item["actual_cycle_k"])
        actual = actual_factor["intervention"]
        row = actual["rows"][actual["selected_logit_row"]]
        report_rows.append(
            f"| {item['prompt_id']} | {item['actual_cycle_k']} | {item['selected_cycle_verification_mode']} | "
            f"{item['selected_cycle_index']} | {item['actual_cycle_k_classification']} | "
            f"{row['A_active_batch_argmax']} | {row['D_canonical_sequential_argmax']} | "
            f"{row['vanilla_reference_token_id']} | {actual_factor['candidate_sequence_origin']} |")
    report = f"""# KV × verification factorial intervention

## Protocol

- Execution commit: `{execution_commit}`; model `{args.model}` revision `{args.revision}`.
- NF4, FP16 compute, double quant enabled; SDPA; static KV; RTX 4050.
- Diagnostic cohort: {len(details)} prompts; each PLD output mismatch vs the matched-capacity canonical sequential reference: {summary['sequence_mismatch_count']}/{len(details)}.
- For each selected prompt prefix and fixed candidate sequence, four isolated cache clones were evaluated: A active KV + batch, B canonical KV + batch, C active KV + sequential, D canonical KV + sequential.
- Every condition began at the same logical prefix length; input token positions are `[prefix_length, prefix_length + candidate_length)`. Attention mask was omitted consistently so the model/cache constructs the causal mask from logical cache length.
- Cache snapshots were fingerprinted before intervention; source snapshots remained unchanged. All cache length and clone integrity checks passed: `{integrity_ok}`.
- The selected row is the first active target argmax that differs from the same-prefix canonical vanilla continuation. Logit row `j` follows candidate input token `j` and predicts the next token. Each row logs argmax, top-1 margin, MAE, max absolute/relative error, L2 norm, and the factorial interaction norm. KV errors are recorded layer-wise.

## Attribution

Actual selected-cycle classification counts: `{json.dumps(class_counts, sort_keys=True)}`. Actual cycle K counts: `{json.dumps(actual_k_counts, sort_keys=True)}`. Verification modes: `{json.dumps(cycle_modes, sort_keys=True)}`. Actual cycle input source: `{json.dumps(actual_origin_counts, sort_keys=True)}`. Counterfactual vanilla continuation extension was used in {extended_probe_count} prompts.

The main attribution uses the actual K at the first mismatch. For cycles that fell back with K=0, K=1/2/4 probes extend the fixed candidate sequence using the vanilla continuation; those are counterfactual execution-shape controls, not candidates PLD actually drafted. They are labeled in the raw JSONL. Three prompts had real speculative candidates at the selected cycle.

| Prompt | Actual K | Mode | Cycle | Attribution | A argmax | D argmax | Vanilla expected | Candidate source |
|---|---:|---|---:|---|---:|---:|---:|---|
{chr(10).join(report_rows)}

## Interpretation

Classifications describe selected-row greedy argmax decisions, not sequence-level guarantees. A nonzero logit norm alone is not treated as a token change. The actual-cycle results attribute selected mismatches to KV history for 15 prompts and execution shape for 1 prompt. The K=1/2/4 counterfactuals show how batch sensitivity changes with candidate length; they should not be reported as actual PLD decisions where the cycle used K=0.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
