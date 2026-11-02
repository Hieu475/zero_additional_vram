#!/usr/bin/env python3
"""B2 reject-path audit: activationvec block + rollback vs BNB greedy singleton state."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import triton

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION, original_bnb_forward
from audit_bnb_nf4_activationvec_block_integration import triton_m3_overrides
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest


@contextmanager
def block_ctx(model):
    with (original_bnb_forward(model),
          fixed_row_schedule_for_model(model, physical_rows=3, attention_strategy="bnb_rowwise",
                                       quantized_linear_strategy="fixed_rows"),
          triton_m3_overrides(model)):
        yield


def reference(model, prompt_ids, capacity, steps=6):
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    plen = int(prompt_ids.shape[-1])
    tokens, logits_after, digests = [], [], {}
    with torch.inference_mode(), original_bnb_forward(model):
        logits = model(prompt_ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
        digests[0] = diagnostic_cache_prefix_digest(cache, plen)
        for i in range(steps):
            token = logits.argmax(dim=-1)
            tokens.append(int(token.item()))
            logits = model(token, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
            logits_after.append(logits.detach().clone())
            digests[i + 1] = diagnostic_cache_prefix_digest(cache, plen + i + 1)
    return tokens, logits_after, digests


def run_block(model, cache, inputs, device):
    tensor = torch.tensor([inputs], device=device, dtype=torch.long)
    with torch.inference_mode(), block_ctx(model):
        logits = model(tensor, past_key_values=cache.cache, use_cache=True).logits[0]
    return logits.argmax(dim=-1).tolist()


def commit(cache, plen, committed_before, inputs, preds):
    """Mirror b1_fixed_row_block_generate acceptance; return (committed_count, pending, rejected_at)."""
    drafts = inputs[1:]
    prefix_len = plen + committed_before
    rejected_at = None
    for pos, draft in enumerate(drafts):
        if preds[pos] != draft:
            rejected_at = pos
            break
    if rejected_at is not None:
        cache.crop(prefix_len + rejected_at + 1)
        return committed_before + rejected_at + 1, preds[rejected_at], rejected_at
    return committed_before + len(inputs), preds[-1], None


def next_step_matches(model, cache, pending, expected_logits, device):
    tensor = torch.tensor([[pending]], device=device, dtype=torch.long)
    with torch.inference_mode(), original_bnb_forward(model):
        logits = model(tensor, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
    return torch.equal(logits, expected_logits)


def wrong(token, vocab):
    return (token + 1) % vocab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip()]
    prompt = next((r for r in records if str(r["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to(device)
    plen = int(prompt_ids.shape[-1])
    capacity = plen + 24
    vocab = int(model.config.vocab_size)
    t, logits_after, ref_digests = reference(model, prompt_ids, capacity)

    scenarios = {
        "reject_at_0": [[t[0], wrong(t[1], vocab), wrong(t[2], vocab)]],
        "reject_at_1": [[t[0], t[1], wrong(t[2], vocab)]],
        "all_accept": [[t[0], t[1], t[2]]],
        "reject_at_0_then_partial_accept_reject_at_1": [
            [t[0], wrong(t[1], vocab), wrong(t[2], vocab)],
            [t[1], t[2], wrong(t[3], vocab)]],
        "all_accept_then_reject_at_0": [
            [t[0], t[1], t[2]],
            [t[3], wrong(t[4], vocab), wrong(t[5], vocab)]],
    }
    results = {}
    for name, cycles in scenarios.items():
        cache = TargetKVCache(backend="static", max_capacity=capacity)
        with torch.inference_mode(), block_ctx(model):
            model(prompt_ids, past_key_values=cache.cache, use_cache=True)
        committed, steps = 0, []
        for inputs in cycles:
            preds = run_block(model, cache, inputs, device)
            committed, pending, rejected_at = commit(cache, plen, committed, inputs, preds)
            digest_equal = diagnostic_cache_prefix_digest(cache, plen + committed) == ref_digests[committed]
            length_ok = cache.get_seq_length() == plen + committed
            pending_ok = pending == t[committed]
            steps.append({"inputs": inputs, "predictions": preds, "rejected_at": rejected_at,
                          "committed": committed, "pending": pending, "pending_matches_bnb": pending_ok,
                          "kv_prefix_equal": digest_equal, "cache_length_ok": length_ok})
        last_ok = next_step_matches(model, cache, steps[-1]["pending"], logits_after[committed], device)
        results[name] = {"steps": steps, "next_bnb_m1_logits_bitwise_equal": last_ok,
                         "passed": last_ok and all(s["pending_matches_bnb"] and s["kv_prefix_equal"]
                                                   and s["cache_length_ok"] for s in steps)}

    result = {
        "schema": "zassd-activationvec-reject-path-b2-v1",
        "classification": "development-only reject/rollback B2 state audit; finite evidence, not a proof",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id, "prompt_length": plen,
        "reference_tokens": t, "scenarios": results,
        "all_passed": all(s["passed"] for s in results.values()),
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                    "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["One prompt per invocation, forced draft tokens, K=2/R=3.",
                         "Next-step check uses plain BNB M=1 from the rolled-back cache.",
                         "GSM8K held-out indices 300-349 excluded."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {n: s["passed"] for n, s in v.items()} if k == "scenarios" else v
                      for k, v in result.items() if k in ("prompt_id", "scenarios", "all_passed")}, indent=2))
    if not result["all_passed"]:
        raise RuntimeError("Activationvec reject path diverged from BNB singleton state")


if __name__ == "__main__":
    main()
