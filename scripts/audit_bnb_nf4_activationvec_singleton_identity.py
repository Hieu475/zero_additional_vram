#!/usr/bin/env python3
"""Compare one fused M=3 Triton verification block with greedy BNB singleton."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
import triton

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.decoding.speculative import diagnostic_cache_digest
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION, original_bnb_forward
from audit_bnb_nf4_activationvec_block_integration import triton_m3_overrides


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((item for item in records if str(item["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    capacity = int(prompt_ids.shape[-1]) + 8

    singleton_cache = TargetKVCache(backend="static", max_capacity=capacity)
    singleton_logits_rows = []
    forced_tokens = []
    with torch.inference_mode(), original_bnb_forward(model):
        logits = model(prompt_ids, past_key_values=singleton_cache.cache, use_cache=True).logits[:, -1:, :]
        for _ in range(3):
            token = logits.argmax(dim=-1)
            forced_tokens.append(int(token.item()))
            logits = model(token, past_key_values=singleton_cache.cache, use_cache=True).logits[:, -1:, :]
            singleton_logits_rows.append(logits.detach().clone())
    singleton_logits = torch.cat(singleton_logits_rows, dim=1)
    singleton_kv_digest = diagnostic_cache_digest(singleton_cache)
    singleton_lengths = [int(layer.get_seq_length()) for layer in singleton_cache.cache.layers]

    candidate_cache = TargetKVCache(backend="static", max_capacity=capacity)
    block_tokens = torch.tensor([forced_tokens], device="cuda:0", dtype=torch.long)
    with (torch.inference_mode(), original_bnb_forward(model),
          fixed_row_schedule_for_model(model, physical_rows=3, attention_strategy="bnb_rowwise",
                                      quantized_linear_strategy="fixed_rows"),
          triton_m3_overrides(model)):
        model(prompt_ids, past_key_values=candidate_cache.cache, use_cache=True)
        candidate_logits = model(block_tokens, past_key_values=candidate_cache.cache, use_cache=True).logits.detach()
    candidate_kv_digest = diagnostic_cache_digest(candidate_cache)
    candidate_lengths = [int(layer.get_seq_length()) for layer in candidate_cache.cache.layers]

    logits_equal = torch.equal(singleton_logits, candidate_logits)
    kv_equal = singleton_kv_digest == candidate_kv_digest
    result = {
        "schema": "zassd-bnb-fused-nf4-activationvec-singleton-identity-v1",
        "classification": "development-only one-cycle B2 identity check; finite evidence, not a proof",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_length": int(prompt_ids.shape[-1]), "forced_bnb_singleton_tokens": forced_tokens,
        "physical_rows": 3, "patched_linear_count": len(model.model.layers) * 7,
        "singleton_logits_shape": list(singleton_logits.shape), "block_logits_shape": list(candidate_logits.shape),
        "logits_bitwise_equal": logits_equal,
        "logits_max_abs_diff": float((singleton_logits.float() - candidate_logits.float()).abs().max().item()),
        "singleton_logits_sha256": tensor_sha256(singleton_logits),
        "block_logits_sha256": tensor_sha256(candidate_logits),
        "singleton_argmax_tokens": singleton_logits.argmax(dim=-1).reshape(-1).tolist(),
        "block_argmax_tokens": candidate_logits.argmax(dim=-1).reshape(-1).tolist(),
        "argmax_tokens_equal": torch.equal(singleton_logits.argmax(dim=-1), candidate_logits.argmax(dim=-1)),
        "singleton_kv_prefix_sha256": singleton_kv_digest,
        "block_kv_prefix_sha256": candidate_kv_digest,
        "kv_prefix_bitwise_equal": kv_equal,
        "singleton_cache_lengths": singleton_lengths,
        "block_cache_lengths": candidate_lengths,
        "cache_lengths_equal": singleton_lengths == candidate_lengths,
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "integration_source": "scripts/audit_bnb_nf4_activationvec_block_integration.py",
        "kernel_source": "scripts/benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2.py",
        "runtime": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"), "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["One three-token greedy prefix and one M=3 verify block, followed by next-logit comparison.",
                         "Uses the fixed-row BNB-rowwise attention contract and M=3 Triton linears.",
                         "No rejection/rollback, EOS, pending-token, long closed-loop, or end-to-end latency coverage.",
                         "One prompt, one GPU/toolchain; GSM8K held-out indices 300-349 excluded."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not logits_equal or not kv_equal or not result["cache_lengths_equal"]:
        raise RuntimeError("Triton M=3 block diverged from greedy BNB singleton logits or active KV")


if __name__ == "__main__":
    main()
