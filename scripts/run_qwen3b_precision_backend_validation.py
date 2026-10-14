#!/usr/bin/env python3
"""Run the feasible Qwen2.5-3B NF4 precision/backend validation matrix.

The runner records exact token IDs, per-cycle cache/logit drift against a
canonical sequential continuation, and latency for a labeled diagnostic and
held-out cohort. It intentionally does not claim full-precision guarantees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from zassd.baselines.prompt_lookup import prompt_lookup_generate  # noqa: E402
from zassd.cache.kv_cache import TargetKVCache  # noqa: E402

DEFAULT_INPUTS = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl"
DEFAULT_CONFIG = ROOT / "experiments/numerical_causality/traces/config.json"
DEFAULT_OUT = ROOT / "experiments/numerical_causality/precision_validation"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
CONFIGS = {
    "sdpa_static": {"attention": "sdpa", "cache": "static", "sequential": False},
    "eager_static": {"attention": "eager", "cache": "static", "sequential": False},
    "sdpa_dynamic": {"attention": "sdpa", "cache": "dynamic", "sequential": False},
    "sdpa_static_sequential": {"attention": "sdpa", "cache": "static", "sequential": True},
}


def sha(value: Any) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def errors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | bool]:
    delta = (a.float() - b.float()).abs()
    denom = b.float().abs().square().mean().sqrt().clamp_min(1e-12)
    return {"mae": float(delta.mean()), "max_abs": float(delta.max()),
            "relative_rms": float(delta.square().mean().sqrt() / denom),
            "max_relative": float((delta / b.float().abs().clamp_min(1e-12)).max()),
            "bitwise_equal": bool(torch.equal(a, b))}


def snapshot(cache) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    out = []
    for layer in cache.layers:
        n = int(layer.get_seq_length())
        if getattr(layer, "is_initialized", False) and n:
            k = layer.keys[:, :, :n].detach().cpu().clone()
            v = layer.values[:, :, :n].detach().cpu().clone()
        else:
            k = v = torch.empty(0)
        out.append((k, v, n))
    return out


def canonical_trace(model, ids: list[int], budget: int, device: str,
                    backend: str, capacity: int, eos: int | None):
    """Return canonical greedy output, per-input logits, and cache snapshots."""
    cache = TargetKVCache(backend=backend, max_capacity=capacity)
    toks, logits_by_input, caches = [], {}, {}
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, past_key_values=cache.cache, use_cache=True)
        # Prefill logits predict the first continuation token.
        next_id = int(out.logits[0, -1].argmax())
        logits_by_input[len(ids) - 1] = out.logits[0, -1].detach().float().cpu()
        caches[len(ids)] = snapshot(cache.cache)
        for _ in range(budget):
            toks.append(next_id)
            if next_id == eos:
                break
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(x, past_key_values=cache.cache, use_cache=True)
            logits_by_input[len(ids) + len(toks) - 1] = out.logits[0, -1].detach().float().cpu()
            next_id = int(out.logits[0, -1].argmax())
            caches[len(ids) + len(toks)] = snapshot(cache.cache)
    # Store the continuation logit vectors keyed by the absolute input position.
    # Prefill index is prompt_length-1; each subsequent token adds one entry.
    return toks, logits_by_input, caches


def compare_cycle(cycle: dict, prompt_len: int, canonical_tokens: list[int],
                  pos_logits: dict[int, torch.Tensor], canonical_caches):
    prefix = cycle["prefix_tokens"]
    generated_prefix = prefix[prompt_len:]
    if prefix[:prompt_len] == [] or generated_prefix != canonical_tokens[:len(generated_prefix)]:
        return None
    active = cycle.get("cache_before", [])
    canonical = canonical_caches.get(len(prefix))
    kv_layer = []
    if canonical is not None and len(active) == len(canonical):
        for i, (a, c) in enumerate(zip(active, canonical)):
            ak, av, an = a
            ck, cv, cn = c
            n = min(int(an), int(cn), ak.shape[-2] if ak.ndim else 0)
            if int(an) != int(cn):
                kv_layer.append({"layer": i, "length_mismatch": [int(an), int(cn)]})
            elif n:
                kv_layer.append({"layer": i, "key": errors(ak[:, :, :n], ck[:, :, :n]),
                                 "value": errors(av[:, :, :n], cv[:, :, :n])})
        kv_equal = all(x.get("key", {}).get("bitwise_equal", False)
                       and x.get("value", {}).get("bitwise_equal", False) for x in kv_layer)
    else:
        kv_equal = None
    candidate = cycle.get("candidate_tokens", [])
    active_logits = cycle.get("verify_logits")
    logit_rows = []
    if active_logits is not None:
        for row, token in enumerate(candidate[:active_logits.shape[1]]):
            absolute_input_pos = len(prefix) + row
            ref = pos_logits.get(absolute_input_pos)
            if ref is None:
                continue
            cur = active_logits[0, row].float()
            top = torch.topk(ref, 2).values
            atop = torch.topk(cur, 2).values
            logit_rows.append({"row": row, "absolute_input_position": absolute_input_pos,
                               "candidate_input_token_id": token,
                               "active_argmax": int(cur.argmax()), "canonical_argmax": int(ref.argmax()),
                               "active_top1_margin": float(atop[0] - atop[1]),
                               "canonical_top1_margin": float(top[0] - top[1]),
                               **errors(cur.cpu(), ref)})
    return {"prefix_length": len(prefix), "kv_bitwise_equal": kv_equal,
            "kv_layer_errors": kv_layer, "logit_rows": logit_rows}


def select_rows(input_path: Path, config_path: Path, diagnostic_n: int,
                heldout_n: int, ids_override: list[str] | None):
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    cfg = json.loads(config_path.read_text())
    diagnostic_ids = list(cfg.get("ids", []))
    if ids_override:
        diagnostic_ids = ids_override
    by_id = {r["prompt_id"]: r for r in rows}
    diagnostics = [by_id[x] for x in diagnostic_ids if x in by_id][:diagnostic_n]
    dset = {r["prompt_id"] for r in diagnostics}
    heldout = [r for r in rows if r["prompt_id"] not in dset][:heldout_n]
    return [(r, "diagnostic") for r in diagnostics] + [(r, "heldout") for r in heldout]


def wp2_attributions() -> dict[str, str]:
    path = ROOT / "experiments/numerical_causality/interventions/factorial_kv_verification_final_v2.jsonl"
    if not path.exists():
        return {}
    return {r["prompt_id"]: r.get("actual_cycle_k_classification", "unresolved")
            for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    ap.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--trace-config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--diagnostic", type=int, default=16)
    ap.add_argument("--heldout", type=int, default=14)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NF4 validation")
    chosen = list(CONFIGS) if args.config == "all" else [args.config]
    prompts = select_rows(args.inputs, args.trace_config, args.diagnostic, args.heldout, args.ids)
    attributions = wp2_attributions()
    if not prompts:
        raise RuntimeError("No prompt rows selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", revision=REVISION)
    all_summary = []
    for name in chosen:
        spec = CONFIGS[name]
        output = args.output_dir / f"{name}.jsonl"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct", revision=REVISION,
            quantization_config=quant, device_map=args.device,
            attn_implementation=spec["attention"]).eval()
        file_rows = []
        for idx, (row, cohort) in enumerate(prompts):
            prompt_ids = list(row["input_ids"])
            x = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
            cap = len(prompt_ids) + args.max_new_tokens + 20
            canonical, pos_logits, canonical_caches = canonical_trace(
                model, prompt_ids, args.max_new_tokens, args.device,
                spec["cache"], cap, tok.eos_token_id)
            cycles = []
            _, metrics = prompt_lookup_generate(
                model, tok, x, k=4, ngram_size=3,
                max_new_tokens=args.max_new_tokens, temperature=0.0,
                device=args.device, kv_cache_backend=spec["cache"],
                candidate_source="prompt+generated", sequential_verify=spec["sequential"],
                kv_cache_max_capacity=cap, numerical_trace=cycles)
            generated = list(metrics.generated_token_ids)
            diffs = []
            for cycle in cycles:
                d = compare_cycle(cycle, len(prompt_ids), canonical, pos_logits, canonical_caches)
                if d is not None:
                    diffs.append(d)
            first_kv = next((d for d in diffs if d["kv_bitwise_equal"] is False), None)
            first_logit = next(({"cycle_index": i, "row": r["row"], **r}
                                for i, d in enumerate(diffs) for r in d["logit_rows"]
                                if not r["bitwise_equal"]), None)
            first_token = next((i for i, (a, b) in enumerate(zip(generated, canonical)) if a != b),
                               min(len(generated), len(canonical)) if len(generated) != len(canonical) else None)
            first_kv_layer = None
            if first_kv is not None:
                first_kv_layer = next((layer["layer"] for layer in first_kv["kv_layer_errors"]
                                       if "key" in layer and (not layer["key"]["bitwise_equal"]
                                                               or not layer["value"]["bitwise_equal"])), None)
            row_out = {
                "prompt_id": row["prompt_id"], "cohort": cohort,
                "prompt_token_ids_sha256": sha(prompt_ids), "prompt_tokens": len(prompt_ids),
                "generated_token_ids": generated, "canonical_token_ids": canonical,
                "sequence_exact_match": generated == canonical,
                "first_token_divergence": first_token,
                "eos_position": (generated.index(tok.eos_token_id) if tok.eos_token_id in generated else None),
                "first_kv_divergence_cycle": (cycles[diffs.index(first_kv)]["cycle_index"] if first_kv else None),
                "first_kv_divergent_layer": first_kv_layer,
                "first_logit_divergence": first_logit,
                "wp2_factorial_attribution": attributions.get(row["prompt_id"]),
                "argmax_agreement": (sum(a == b for a, b in zip(generated, canonical)) /
                                     max(1, min(len(generated), len(canonical)))),
                "verification_latency_s": metrics.verify_time_s,
                "end_to_end_latency_s": metrics.total_time_s,
                "tokens_per_second": metrics.tokens_per_second,
                "acceptance_rate": metrics.acceptance_rate,
                "peak_vram_mb": metrics.peak_vram_mb,
                "cycle_comparisons": diffs,
            }
            file_rows.append(row_out)
            with output.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"config": name, **row_out}, separators=(",", ":")) + "\n")
            print(f"{name} {idx+1}/{len(prompts)} {row['prompt_id']} cohort={cohort} exact={row_out['sequence_exact_match']} first={first_token}", flush=True)
        n = len(file_rows)
        summary = {"config": name, "output": str(output),
                   "n": n, "cohort_counts": {c: sum(r["cohort"] == c for r in file_rows) for c in ("diagnostic", "heldout")},
                   "sequence_exact_match_rate": sum(r["sequence_exact_match"] for r in file_rows) / max(1, n),
                   "token_argmax_agreement_mean": sum(r["argmax_agreement"] for r in file_rows) / max(1, n),
                   "first_token_divergences": sum(r["first_token_divergence"] is not None for r in file_rows),
                   "mean_tps": sum(r["tokens_per_second"] for r in file_rows) / max(1, n),
                   "mean_verification_latency_s": sum(r["verification_latency_s"] for r in file_rows) / max(1, n),
                   "mean_peak_vram_mb": sum(r["peak_vram_mb"] for r in file_rows) / max(1, n)}
        (args.output_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        all_summary.append(summary)
        del model
        torch.cuda.empty_cache()
    manifest = {"execution_commit": commit, "model": "Qwen/Qwen2.5-3B-Instruct",
                "model_revision": REVISION, "tokenizer_revision": REVISION,
                "quantization": {"type": "NF4", "compute_dtype": "float16", "double_quant": True},
                "max_new_tokens": args.max_new_tokens, "K": 4, "ngram_size": 3,
                "candidate_source": "prompt+generated", "seed_note": "greedy decoding; no random sampling",
                "input_file": str(args.inputs.relative_to(ROOT)),
                "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                "trace_config_sha256": hashlib.sha256(args.trace_config.read_bytes()).hexdigest(),
                "cohort_counts": {"diagnostic": sum(c == "diagnostic" for _, c in prompts),
                                  "heldout": sum(c == "heldout" for _, c in prompts)},
                "hardware": torch.cuda.get_device_name(args.device),
                "cuda": torch.version.cuda, "torch": torch.__version__,
                "transformers": transformers.__version__, "python": platform.python_version(),
                "configs": chosen, "summaries": all_summary}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(args.output_dir, all_summary, manifest)


def write_report(outdir: Path, summaries: list[dict], manifest: dict):
    lines = ["# Qwen2.5-3B NF4 precision/backend validation", "",
             f"Execution commit: `{manifest['execution_commit']}`. Model revision: `{REVISION}`. ",
             f"Hardware: `{manifest['hardware']}`; cohort: {manifest['cohort_counts']}.",
             "NF4 double quant, FP16 compute, greedy decode, 32 token limit, K=4.", "",
             "## Results", "", "| Config | N | Exact match | Mean argmax agreement | First token divergence | TPS | Verify latency (s) | Peak VRAM (MB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['config']} | {s['n']} | {s['sequence_exact_match_rate']:.3f} | {s['token_argmax_agreement_mean']:.4f} | {s['first_token_divergences']} | {s['mean_tps']:.2f} | {s['mean_verification_latency_s']:.4f} | {s['mean_peak_vram_mb']:.0f} |")
    lines += ["", "## Interpretation and limits", "",
              "Numerical drift is recorded independently from argmax and sequence divergence. Any nonzero logit/KV error is not itself evidence of changed output; exact-match is only for this labeled cohort and 32-token greedy run. It does not establish a lossless guarantee.",
              "The held-out rows are unselected rows from the same prepared-input artifact, not an external dataset. FP16/BF16 unquantized and FP32 controls are not measured by this NF4 GPU matrix; they remain open because this run targets the 6 GB RTX 4050.",
              "WP2 factorial attribution is linked for diagnostic prompts where a matching prior intervention exists; held-out prompts have no causal label from WP2.",
              "Raw per-prompt measurements and software/hardware manifest are stored beside this report.", ""]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
