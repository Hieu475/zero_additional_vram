#!/usr/bin/env python3
"""Stress rollback/EOS rejection followed by native no-proposal fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from audit_activationvec_no_proposal_fallback import (
    MANIFEST as FALLBACK_MANIFEST,
    no_proposal_native_singleton_hook,
    validate_manifest as validate_fallback_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_no_proposal_fallback_branches_manifest.json"


def validate_manifest() -> None:
    data = json.loads(MANIFEST.read_text())
    if data.get("schema") != "zassd-activationvec-no-proposal-fallback-branches-manifest-v1":
        raise RuntimeError("invalid branch-audit manifest")
    for rel, expected in data["source_sha256"].items():
        path = ROOT / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source pin mismatch: {rel}")
    validate_fallback_manifest()


def make_observer(store: dict[int, str], prompt_len: int):
    last = {"n": -1}

    def observe(n, cache):
        n = int(n)
        for prefix in range(last["n"] + 1, n + 1):
            store[prefix] = diagnostic_cache_prefix_digest(cache, prompt_len + prefix)
        last["n"] = max(last["n"], n)

    return observe


class EOSOverrideTokenizer:
    """Test-only tokenizer proxy that marks a chosen continuation token as EOS."""

    def __init__(self, base, eos_token_id: int):
        self._base = base
        self.eos_token_id = int(eos_token_id)

    def __getattr__(self, name):
        return getattr(self._base, name)


def run_accepted_eos_fixture(model, tokenizer, prompt_ids, reference_tokens, prompt_len,
                             *, max_new_tokens: int):
    import zassd.decoding.b1_exact as b1_module

    eos = int(reference_tokens[1])
    fixture_tokenizer = EOSOverrideTokenizer(tokenizer, eos)
    ref_states: dict[int, str] = {}
    common = dict(k=2, ngram_size=3, max_new_tokens=max_new_tokens, device="cuda:0",
                  kv_cache_max_capacity=prompt_len + max_new_tokens + 8,
                  physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy="fixed_rows")
    with torch.inference_mode():
        _, ref_metrics = b1_fixed_row_generate(
            model, fixture_tokenizer, prompt_ids, **common,
            state_observer=make_observer(ref_states, prompt_len))

    original_find = b1_module.find_candidate_tokens
    call_count = {"n": 0}

    def propose_one_eos(*_args, **_kwargs):
        call_count["n"] += 1
        return [eos] if call_count["n"] == 1 else []

    b1_module.find_candidate_tokens = propose_one_eos
    candidate_states: dict[int, str] = {}
    control = {"no_proposal": False, "no_proposal_cycles": 0, "fallback_forwards": 0}
    try:
        with no_proposal_native_singleton_hook(model, control), torch.inference_mode():
            _, candidate_metrics = b1_fixed_row_block_generate(
                model, fixture_tokenizer, prompt_ids, **common, trace=True,
                state_observer=make_observer(candidate_states, prompt_len))
    finally:
        b1_module.find_candidate_tokens = original_find

    shared = sorted(set(ref_states) & set(candidate_states))
    first_cycle = candidate_metrics.cycle_trace[0] if candidate_metrics.cycle_trace else {}
    return {
        "name": "accept_terminal_eos_without_fallback",
        "fixture_eos_token_id": eos,
        "reference_tokens": ref_metrics.generated_token_ids,
        "candidate_tokens": candidate_metrics.generated_token_ids,
        "token_match_reference": candidate_metrics.generated_token_ids == ref_metrics.generated_token_ids,
        "reference_state_prefixes": sorted(ref_states),
        "candidate_state_prefixes": sorted(candidate_states),
        "state_domains_match_reference": set(ref_states) == set(candidate_states),
        "state_mismatches_reference": [
            prefix for prefix in shared if ref_states[prefix] != candidate_states[prefix]],
        "state_exact_reference": (
            set(ref_states) == set(candidate_states)
            and all(ref_states[prefix] == candidate_states[prefix] for prefix in shared)),
        "first_cycle": first_cycle,
        "fallback_counts": control,
        "proposal_calls": call_count["n"],
        "accepted_terminal_eos": (
            bool(candidate_metrics.generated_token_ids)
            and candidate_metrics.generated_token_ids[-1] == eos
            and first_cycle.get("rejected_at") is None
            and int(first_cycle.get("accepted", 0)) == 1),
        "cycle_trace": candidate_metrics.cycle_trace,
    }


def run_scenario(model, tokenizer, prompt_ids, reference_tokens, prompt_len,
                 *, scenario_name: str, first_cycle_draft: list[int], max_new_tokens: int):
    scripted = {"calls": 0}
    original_find = __import__("zassd.decoding.b1_exact", fromlist=["find_candidate_tokens"]).find_candidate_tokens

    def scripted_find(*_args, **_kwargs):
        call = scripted["calls"]
        scripted["calls"] += 1
        return list(first_cycle_draft) if call == 0 else []

    b1_module = __import__("zassd.decoding.b1_exact", fromlist=["find_candidate_tokens"])
    b1_module.find_candidate_tokens = scripted_find
    states: dict[int, str] = {}
    control = {"no_proposal": False, "no_proposal_cycles": 0, "fallback_forwards": 0}
    common = dict(k=2, ngram_size=3, max_new_tokens=max_new_tokens, device="cuda:0",
                  kv_cache_max_capacity=prompt_len + max_new_tokens + 8,
                  physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy="fixed_rows", trace=True)
    try:
        with no_proposal_native_singleton_hook(model, control), torch.inference_mode():
            _, metrics = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                state_observer=make_observer(states, prompt_len))
    finally:
        b1_module.find_candidate_tokens = original_find

    target_ref = reference_tokens[:len(metrics.generated_token_ids)]
    token_match = metrics.generated_token_ids == target_ref
    # The caller compares these cache digests against the canonical B1 reference.
    first_cycle = metrics.cycle_trace[0] if metrics.cycle_trace else {}
    return {
        "name": scenario_name,
        "draft_script_first_cycle": first_cycle_draft,
        "generated_token_ids": metrics.generated_token_ids,
        "reference_token_ids": target_ref,
        "token_match_reference": token_match,
        "candidate_state_prefixes": sorted(states),
        "candidate_state_digests": states,
        "first_cycle": first_cycle,
        "cycle_trace": metrics.cycle_trace,
        "fallback_counts": control,
        "scripted_proposal_calls": scripted["calls"],
        "accepted_draft_tokens": metrics.accepted_draft_tokens,
        "draft_tokens": metrics.draft_tokens,
    }


def main() -> None:
    validate_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    parser.add_argument("--prompt-id", default="HumanEval/116")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_new_tokens < 5:
        raise ValueError("max_new_tokens must be at least 5 for rollback/fallback sequence")
    rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in rows if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + args.max_new_tokens + 8
    ref_states: dict[int, str] = {}
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens, device="cuda:0",
                  kv_cache_max_capacity=capacity, physical_rows=3,
                  attention_strategy="bnb_rowwise", quantized_linear_strategy="fixed_rows")
    with torch.inference_mode():
        _, ref_metrics = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **common,
            state_observer=make_observer(ref_states, prompt_len))
    ref_tokens = ref_metrics.generated_token_ids
    if len(ref_tokens) < 4:
        raise RuntimeError("reference output is too short for scheduled rollback fixture")
    eos = int(tokenizer.eos_token_id)
    wrong_first = (ref_tokens[1] + 1) % int(model.config.vocab_size)
    while wrong_first == ref_tokens[1] or wrong_first == eos:
        wrong_first = (wrong_first + 1) % int(model.config.vocab_size)
    rejected_eos = eos if eos != ref_tokens[2] else (eos + 1) % int(model.config.vocab_size)
    scenarios = [
        run_scenario(model, tokenizer, prompt_ids, ref_tokens, prompt_len,
                     scenario_name="accept_one_reject_eos_then_empty_proposals",
                     first_cycle_draft=[ref_tokens[1], rejected_eos],
                     max_new_tokens=args.max_new_tokens),
        run_scenario(model, tokenizer, prompt_ids, ref_tokens, prompt_len,
                     scenario_name="reject_first_and_eos_suffix_then_empty_proposals",
                     first_cycle_draft=[wrong_first, eos],
                     max_new_tokens=args.max_new_tokens),
    ]
    accepted_eos = run_accepted_eos_fixture(
        model, tokenizer, prompt_ids, ref_tokens, prompt_len,
        max_new_tokens=args.max_new_tokens)
    for scenario in scenarios:
        scenario["state_mismatches_reference"] = [
            prefix for prefix, digest in scenario["candidate_state_digests"].items()
            if ref_states.get(int(prefix)) != digest
        ]
        scenario["state_domains_match_reference"] = set(map(int, scenario["candidate_state_digests"])) == set(ref_states)
        scenario["state_exact_reference"] = (
            scenario["state_domains_match_reference"] and not scenario["state_mismatches_reference"])
        del scenario["candidate_state_digests"]

    rejected_then_fallback_passed = all(
        scenario["token_match_reference"] and scenario["state_exact_reference"]
        and scenario["fallback_counts"]["no_proposal_cycles"] == scenario["fallback_counts"]["fallback_forwards"]
        and scenario["fallback_counts"]["fallback_forwards"] > 0
        and scenario["first_cycle"].get("rejected_at") is not None
        for scenario in scenarios
    )
    passed = (rejected_then_fallback_passed
              and accepted_eos["token_match_reference"]
              and accepted_eos["state_exact_reference"]
              and accepted_eos["accepted_terminal_eos"]
              and accepted_eos["fallback_counts"]["fallback_forwards"] == 0)
    payload = {
        "schema": "zassd-activationvec-no-proposal-fallback-branches-v1",
        "classification": "development-only controlled proposal scripts; finite EOS-reject/rollback then no-proposal fallback checks",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "max_new_tokens": args.max_new_tokens,
        "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3,
                     "attention": "bnb_rowwise", "fallback": "native BNB M=1 only when proposal list is empty"},
        "reference_tokens": ref_tokens,
        "reference_prefixes": sorted(ref_states),
        "scenarios": scenarios + [accepted_eos],
        "all_scenarios_pass": passed,
        "manifest": str(MANIFEST),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "fallback_manifest": str(FALLBACK_MANIFEST),
        "fallback_manifest_sha256": hashlib.sha256(FALLBACK_MANIFEST.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["Controlled n-gram proposal script, not natural proposal distribution.",
                   "Accepted-EOS case uses a test-only tokenizer EOS override; rejected-EOS cases use the model's actual tokenizer EOS.",
                   "One development prompt and one GPU; GSM8K held-out indices 300-349 excluded."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id, "all_pass": passed,
                      "scenarios": [{k: v for k, v in row.items() if k in (
                          "name", "token_match_reference", "state_exact_reference",
                          "first_cycle", "fallback_counts", "accepted_draft_tokens", "draft_tokens")}
                                    for row in scenarios],
                      "output": str(args.output)}, indent=2))
    if not passed:
        raise RuntimeError("fallback did not preserve B1 across controlled rollback scenario")


if __name__ == "__main__":
    main()
