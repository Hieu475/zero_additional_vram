#!/usr/bin/env python3
"""Same-process A/B: activationvec M=3 vs packed fixed-three-row NF4 linear."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
import audit_nf4_singleton_k2_r3_closed_loop as fixedthree_schedule
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from benchmark_bnb_nf4_activationvec_closed_loop import (
    activationvec_schedule_hook, run_vanilla_timed,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_fixedthree_linear_manifest.json"
FIXED_THREE = "triton_nf4_fixed_three_rows_group_scale_packed"


def validate_manifest() -> None:
    data = json.loads(MANIFEST.read_text())
    if data.get("schema") != "zassd-activationvec-fixedthree-linear-manifest-v1":
        raise RuntimeError("bad fixed-three-row manifest schema")
    for relpath, expected in data["source_sha256"].items():
        p = ROOT / relpath
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source pin mismatch: {relpath}")
    own = "scripts/audit_activationvec_vs_fixedthree_linear_paired.py"
    if data["source_sha256"].get(own) != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("fixed-three-row A/B runner is not pinned")


def observe(store: dict[int, str], prompt_len: int):
    last = {"n": -1}

    def callback(n, cache):
        n = int(n)
        for prefix in range(last["n"] + 1, n + 1):
            store[prefix] = diagnostic_cache_prefix_digest(cache, prompt_len + prefix)
        last["n"] = max(last["n"], n)

    return callback


def main() -> None:
    validate_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timing-rounds", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if args.max_new_tokens < 4 or args.timing_rounds < 1:
        raise ValueError("require max_new_tokens >= 4 and timing_rounds >= 1")
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if row["id"] == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + args.max_new_tokens + 8
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                  device="cuda:0", kv_cache_max_capacity=capacity,
                  physical_rows=3, attention_strategy="bnb_rowwise")

    stores = {name: {} for name in ("bnb_singleton", "activationvec",
                                    "fixedthree_singleton", "fixedthree")}
    with torch.inference_mode():
        _, bnb_singleton = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **common,
            quantized_linear_strategy="fixed_rows",
            state_observer=observe(stores["bnb_singleton"], prompt_len))
        with activationvec_schedule_hook(model):
            _, activationvec = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                quantized_linear_strategy="fixed_rows",
                state_observer=observe(stores["activationvec"], prompt_len))
        fixedthree_schedule.STRATEGY = FIXED_THREE
        with fixedthree_schedule.singleton_kernel_schedule():
            _, fixedthree_singleton = b1_fixed_row_generate(
                model, tokenizer, prompt_ids, **common,
                quantized_linear_strategy=FIXED_THREE,
                state_observer=observe(stores["fixedthree_singleton"], prompt_len))
            _, fixedthree = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                quantized_linear_strategy=FIXED_THREE,
                state_observer=observe(stores["fixedthree"], prompt_len))

    tokens = {"bnb_singleton": bnb_singleton.generated_token_ids,
              "activationvec": activationvec.generated_token_ids,
              "fixedthree_singleton": fixedthree_singleton.generated_token_ids,
              "fixedthree": fixedthree.generated_token_ids}
    shared = sorted(set.intersection(*(set(value) for value in stores.values())))
    mismatches = {
        "activationvec": [prefix for prefix in shared
                          if stores["activationvec"][prefix] != stores["bnb_singleton"][prefix]],
        "fixedthree": [prefix for prefix in shared
                       if stores["fixedthree"][prefix] != stores["fixedthree_singleton"][prefix]],
    }

    token_match = {
        "activationvec": tokens["activationvec"] == tokens["bnb_singleton"],
        "fixedthree": tokens["fixedthree"] == tokens["fixedthree_singleton"],
    }
    activation_domains_equal = set(stores["activationvec"]) == set(stores["bnb_singleton"])
    fixedthree_domains_equal = set(stores["fixedthree"]) == set(stores["fixedthree_singleton"])
    state_exact = {
        "activationvec": activation_domains_equal and not mismatches["activationvec"],
        "fixedthree": fixedthree_domains_equal and not mismatches["fixedthree"],
    }
    if not all(token_match.values()) or not all(state_exact.values()):
        failed = {
            "schema": "zassd-activationvec-vs-fixedthree-linear-paired-v1",
            "classification": "exactness gate failed; latency was not measured or interpreted",
            "model": MODEL, "revision": REVISION,
            "gpu": torch.cuda.get_device_name(0), "prompt_id": args.prompt_id,
            "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
            "max_new_tokens": args.max_new_tokens,
            "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3,
                         "attention": "bnb_rowwise"},
            "exactness": {
                "bnb_singleton_tokens": tokens["bnb_singleton"],
                "fixedthree_singleton_tokens": tokens["fixedthree_singleton"],
                "activationvec_tokens": tokens["activationvec"],
                "fixedthree_tokens": tokens["fixedthree"],
                "token_match_singleton": token_match,
                "shared_state_prefixes": shared,
                "state_mismatches": mismatches,
                "state_domains_equal": {"activationvec_vs_bnb": activation_domains_equal,
                                         "fixedthree_vs_own": fixedthree_domains_equal},
                "state_exact_reference": state_exact,
            },
            "manifest": str(MANIFEST),
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "source": str(Path(__file__).resolve()),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope_limits": ["Finite development prompt; not a proof.",
                             "GSM8K held-out indices 300-349 excluded."],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(failed, indent=2) + "\n")
        print(json.dumps({"prompt_id": args.prompt_id, "exactness": failed["exactness"],
                          "output": str(out)}, indent=2))
        return

    methods = ("original_bnb_greedy", "bnb_singleton", "activationvec",
               "fixedthree_singleton", "fixedthree")
    paired = []
    for round_index in range(args.timing_rounds):
        order = list(methods[round_index % len(methods):] + methods[:round_index % len(methods)])
        result = {"round": round_index, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb_greedy":
                ids, prefill, decode, total = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0")
                result["methods"][method] = {"token_ids": ids, "total_s": total / 1000,
                                             "prefill_s": prefill / 1000, "decode_s": decode / 1000}
            elif method == "bnb_singleton":
                with torch.inference_mode():
                    _, metrics = b1_fixed_row_generate(
                        model, tokenizer, prompt_ids, **common,
                        quantized_linear_strategy="fixed_rows")
                result["methods"][method] = {"token_ids": metrics.generated_token_ids,
                                             "total_s": metrics.total_time_s}
            elif method == "fixedthree_singleton":
                fixedthree_schedule.STRATEGY = FIXED_THREE
                with fixedthree_schedule.singleton_kernel_schedule(), torch.inference_mode():
                    _, metrics = b1_fixed_row_generate(
                        model, tokenizer, prompt_ids, **common,
                        quantized_linear_strategy=FIXED_THREE)
                result["methods"][method] = {"token_ids": metrics.generated_token_ids,
                                             "total_s": metrics.total_time_s}
            else:
                if method == "activationvec":
                    with activationvec_schedule_hook(model), torch.inference_mode():
                        _, metrics = b1_fixed_row_block_generate(
                            model, tokenizer, prompt_ids, **common,
                            quantized_linear_strategy="fixed_rows")
                else:
                    fixedthree_schedule.STRATEGY = FIXED_THREE
                    with fixedthree_schedule.singleton_kernel_schedule(), torch.inference_mode():
                        _, metrics = b1_fixed_row_block_generate(
                            model, tokenizer, prompt_ids, **common,
                            quantized_linear_strategy=FIXED_THREE)
                result["methods"][method] = {"token_ids": metrics.generated_token_ids,
                                             "total_s": metrics.total_time_s,
                                             "verify_s": metrics.verify_time_s,
                                             "cycles": metrics.cycles,
                                             "draft_tokens": metrics.draft_tokens,
                                             "accepted_draft_tokens": metrics.accepted_draft_tokens}
            if method == "activationvec":
                reference = tokens["bnb_singleton"]
            elif method == "fixedthree":
                reference = tokens["fixedthree_singleton"]
            else:
                reference = None
            if reference is not None:
                result[f"{method}_matches_own_reference"] = (
                    result["methods"][method]["token_ids"] == reference)
        paired.append(result)

    ratios = {
        "fixedthree_over_activationvec": [r["methods"]["activationvec"]["total_s"] /
                                          r["methods"]["fixedthree"]["total_s"] for r in paired],
        "activationvec_over_bnb": [r["methods"]["original_bnb_greedy"]["total_s"] /
                                    r["methods"]["activationvec"]["total_s"] for r in paired],
        "fixedthree_over_bnb": [r["methods"]["original_bnb_greedy"]["total_s"] /
                                r["methods"]["fixedthree"]["total_s"] for r in paired],
        "activationvec_over_singleton": [r["methods"]["bnb_singleton"]["total_s"] /
                                         r["methods"]["activationvec"]["total_s"] for r in paired],
        "fixedthree_over_singleton": [r["methods"]["bnb_singleton"]["total_s"] /
                                      r["methods"]["fixedthree"]["total_s"] for r in paired],
        "fixedthree_over_own_singleton": [r["methods"]["fixedthree_singleton"]["total_s"] /
                                          r["methods"]["fixedthree"]["total_s"] for r in paired],
    }
    result = {
        "schema": "zassd-activationvec-vs-fixedthree-linear-paired-v1",
        "classification": "same-process development comparison of two fixed-row NF4 block kernels",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id, "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "max_new_tokens": args.max_new_tokens, "timing_rounds": args.timing_rounds,
        "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3, "attention": "bnb_rowwise"},
        "exactness": {
            "bnb_singleton_tokens": tokens["bnb_singleton"],
            "fixedthree_singleton_tokens": tokens["fixedthree_singleton"],
            "activationvec_tokens_match_bnb_singleton": tokens["activationvec"] == tokens["bnb_singleton"],
            "fixedthree_tokens_match_own_singleton": tokens["fixedthree"] == tokens["fixedthree_singleton"],
            "shared_state_prefixes": shared,
            "activationvec_state_mismatches": mismatches["activationvec"],
            "fixedthree_state_mismatches": mismatches["fixedthree"],
            "activationvec_state_domain_equal_bnb": activation_domains_equal,
            "fixedthree_state_domain_equal_own": fixedthree_domains_equal,
        },
        "acceptance": {"activationvec": {"accepted": activationvec.accepted_draft_tokens,
                                           "drafted": activationvec.draft_tokens},
                       "fixedthree": {"accepted": fixedthree.accepted_draft_tokens,
                                      "drafted": fixedthree.draft_tokens}},
        "ratio_medians": {key: statistics.median(values) for key, values in ratios.items()},
        "ratios_per_round": ratios,
        "paired_rounds": paired,
        "manifest": str(MANIFEST),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope_limits": ["One process per prompt, four paired timing rounds.",
                         "Activationvec is checked against BNB singleton; fixedthree against its matching custom singleton schedule.",
                         "Development prompts only; GSM8K held-out indices 300-349 excluded."],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id, "exactness": result["exactness"],
                      "acceptance": result["acceptance"], "ratio_medians": result["ratio_medians"],
                      "output": str(out)}, indent=2))
    if not (result["exactness"]["activationvec_tokens_match_bnb_singleton"]
            and result["exactness"]["fixedthree_tokens_match_own_singleton"]
            and result["exactness"]["activationvec_state_domain_equal_bnb"]
            and result["exactness"]["fixedthree_state_domain_equal_own"]
            and not result["exactness"]["activationvec_state_mismatches"]
            and not result["exactness"]["fixedthree_state_mismatches"]):
        raise RuntimeError("A block kernel diverged from the fixed-row singleton token/KV reference")


if __name__ == "__main__":
    main()
