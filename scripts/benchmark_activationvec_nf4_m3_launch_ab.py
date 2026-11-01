#!/usr/bin/env python3
"""Same-process A/B of default vs shape-tuned M=3 launches under B1 fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

import audit_activationvec_no_proposal_fallback as fallback
import audit_bnb_nf4_activationvec_block_integration as integration
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from run_strong_baseline_comparison import run_vanilla_timed
from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate


ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_no_proposal_fallback_manifest.json"
MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_nf4_m3_launch_ab_manifest_v1.json"


def default_config(_module):
    return 8, 8


def tuned_config(module):
    if int(module.in_features) == 11008 or int(module.out_features) == 11008:
        return 4, 4
    return 8, 8


def pin_manifest():
    data = json.loads(BASE_MANIFEST.read_text())
    data["classification"] = "same-process paired default-versus-tuned NF4 M=3 launch experiment with B1 no-proposal fallback; source-pinned; development only"
    for name in ("scripts/audit_bnb_nf4_activationvec_block_integration.py",
                 "scripts/benchmark_activationvec_nf4_m3_launch_ab.py",
                 "scripts/tune_activationvec_nf4_m3_launch.py"):
        data["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    data["launch_ab"] = {
        "default": {"out_tile": 8, "num_warps": 8},
        "tuned": {"wide_mlp_out_tile": 4, "wide_mlp_num_warps": 4,
                  "other_out_tile": 8, "other_num_warps": 8},
        "selector": "in_features == 11008 or out_features == 11008",
    }
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
    fallback.MANIFEST = MANIFEST


def select_schedule(config_fn):
    fallback.triton_m3_overrides = lambda model: integration.triton_m3_overrides(
        model, config_selector=config_fn
    )


def observe(states, prompt_len):
    last = {"n": -1}
    def callback(n, cache):
        n = int(n)
        for prefix in range(last["n"] + 1, n + 1):
            states[prefix] = diagnostic_cache_prefix_digest(cache, prompt_len + prefix)
        last["n"] = max(last["n"], n)
    return callback


def generate_fallback(model, tokenizer, prompt_ids, common, config_fn, state_store=None):
    select_schedule(config_fn)
    control = {"no_proposal": False, "no_proposal_cycles": 0, "fallback_forwards": 0}
    kwargs = {"state_observer": observe(state_store, int(prompt_ids.shape[-1]))} if state_store is not None else {}
    with fallback.no_proposal_native_singleton_hook(model, control), torch.inference_mode():
        _, metrics = b1_fixed_row_block_generate(model, tokenizer, prompt_ids, **common, **kwargs)
    return metrics, control


def mean_ratio(numerators, denominators):
    return [a / b for a, b in zip(numerators, denominators)]


def phase_times(metrics):
    total = float(metrics.total_time_s)
    prefill = float(metrics.prefill_time_s)
    draft = float(metrics.draft_time_s)
    verify = float(metrics.verify_time_s)
    return {
        "prefill_s": prefill,
        "draft_s": draft,
        "verify_s": verify,
        "wall_after_prefill_s": max(0.0, total - prefill),
        "unattributed_s": total - prefill - draft - verify,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--timing-rounds", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pin_manifest()
    fallback.validate_manifest()

    rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in rows if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                  device="cuda:0", kv_cache_max_capacity=prompt_len + args.max_new_tokens + 8,
                  physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy="fixed_rows", trace=True)
    single_states, default_states, tuned_states = {}, {}, {}
    with torch.inference_mode():
        _, singleton = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **{k: v for k, v in common.items() if k != "trace"},
            state_observer=observe(single_states, prompt_len))
    default_metrics, default_control = generate_fallback(
        model, tokenizer, prompt_ids, common, default_config, default_states)
    tuned_metrics, tuned_control = generate_fallback(
        model, tokenizer, prompt_ids, common, tuned_config, tuned_states)

    ref_tokens = singleton.generated_token_ids
    exactness = {}
    for name, metrics, states in (("default", default_metrics, default_states),
                                  ("tuned", tuned_metrics, tuned_states)):
        shared = sorted(set(states) & set(single_states))
        exactness[name] = {
            "tokens_match_b1_singleton": metrics.generated_token_ids == ref_tokens,
            "state_domains_equal": set(states) == set(single_states),
            "state_mismatches": [prefix for prefix in shared if states[prefix] != single_states[prefix]],
            "shared_prefix_count": len(shared),
        }
    if not all(v["tokens_match_b1_singleton"] and v["state_domains_equal"] and not v["state_mismatches"]
               for v in exactness.values()):
        raise RuntimeError(f"B1 exactness gate failed: {exactness}")

    methods = ("original_bnb", "bnb_singleton", "fallback_default", "fallback_tuned")
    rounds = []
    for index in range(args.timing_rounds):
        shift = index % len(methods)
        order = methods[shift:] + methods[:shift]
        sample = {"round": index, "method_order": list(order), "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb":
                ids, prefill, decode, total = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0")
                entry = {"token_ids": ids, "total_s": total / 1000,
                         "prefill_s": prefill / 1000, "decode_s": decode / 1000}
            elif method == "bnb_singleton":
                with torch.inference_mode():
                    _, metrics = b1_fixed_row_generate(
                        model, tokenizer, prompt_ids,
                        **{k: v for k, v in common.items() if k != "trace"})
                entry = {"token_ids": metrics.generated_token_ids,
                         "total_s": metrics.total_time_s,
                         **phase_times(metrics)}
            else:
                config = default_config if method == "fallback_default" else tuned_config
                metrics, control = generate_fallback(model, tokenizer, prompt_ids, common, config)
                entry = {"token_ids": metrics.generated_token_ids,
                         "total_s": metrics.total_time_s,
                         **phase_times(metrics),
                         "fallback_counts": control,
                         "drafted": metrics.draft_tokens,
                         "accepted": metrics.accepted_draft_tokens,
                         "cycles": metrics.cycles}
            sample["methods"][method] = entry
        rounds.append(sample)

    times = {method: [r["methods"][method]["total_s"] for r in rounds] for method in methods}
    ratios = {
        "tuned_over_default_speedup": mean_ratio(times["fallback_default"], times["fallback_tuned"]),
        "default_over_bnb_speedup": mean_ratio(times["original_bnb"], times["fallback_default"]),
        "tuned_over_bnb_speedup": mean_ratio(times["original_bnb"], times["fallback_tuned"]),
        "tuned_over_singleton_speedup": mean_ratio(times["bnb_singleton"], times["fallback_tuned"]),
    }
    timed_token_match = {
        method: sum(r["methods"][method]["token_ids"] == ref_tokens
                    for r in rounds if method in r["methods"])
        for method in methods
    }
    payload = {
        "schema": "zassd-activationvec-nf4-m3-launch-ab-v1",
        "classification": "same-process paired default-vs-tuned launch comparison with B1 state gate",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "prompt_length": prompt_len, "max_new_tokens": args.max_new_tokens,
        "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3,
                     "attention": "bnb_rowwise", "fallback": "native BNB M=1 on no-proposal cycles"},
        "launch_configs": {"default": "8x8 for all M=3 quantized linears",
                           "tuned": "4x4 iff K or N is 11008; otherwise 8x8"},
        "exactness_preflight": exactness,
        "fallback_counts_preflight": {"default": default_control, "tuned": tuned_control},
        "timed_token_match_rounds": timed_token_match,
        "paired_rounds": rounds,
        "ratio_samples": ratios,
        "ratio_geomeans": {name: statistics.geometric_mean(values) for name, values in ratios.items()},
        "manifest": str(MANIFEST),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "bitsandbytes": __import__("importlib.metadata", fromlist=["version"]).version("bitsandbytes"),
                    "triton": __import__("triton").__version__},
        "limitations": ["single development prompt and process", "original BNB comparison is latency/token only, not KV identity",
                        "GSM8K held-out indices 300-349 excluded"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id, "exactness": exactness,
                      "timed_token_match_rounds": timed_token_match,
                      "ratio_geomeans": payload["ratio_geomeans"],
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
