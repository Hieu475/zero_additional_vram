#!/usr/bin/env python3
"""Compare a native-BNB M=3 block with all Qwen linears replaced by Triton."""

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
from zassd.decoding.speculative import diagnostic_cache_digest
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION, original_bnb_forward
from benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 import _fused_nf4_m3_kernel


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def module_for(layer, projection):
    if projection in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return getattr(layer.self_attn, projection)
    return getattr(layer.mlp, projection)


@contextmanager
def triton_m3_overrides(model, config_selector=None):
    saved = []
    for layer in model.model.layers:
        for projection in PROJECTIONS:
            module = module_for(layer, projection)
            original_forward = module.forward
            state = module.weight.quant_state
            n, k = int(module.out_features), int(module.in_features)
            out_tile, num_warps = ((8, 8) if config_selector is None
                                   else config_selector(module))
            packed = module.weight.data.reshape(-1).contiguous().view(torch.uint32)
            bias = module.bias.to(torch.float16).contiguous() if module.bias is not None else torch.empty((1,), device="cuda:0", dtype=torch.float16)
            kernel_args = (packed, state.absmax.contiguous(), state.state2.absmax.contiguous(), state.state2.code.contiguous(), state.offset.float().contiguous(), state.code.contiguous(), bias)
            grid = (triton.cdiv(n, out_tile),)

            def override(input_tensor, *args, _module=module, _original=original_forward,
                         _state=state, _kernel_args=kernel_args, _grid=grid, _k=k, _n=n,
                         _out_tile=out_tile, _num_warps=num_warps,
                         **kwargs):
                if input_tensor.ndim == 3 and input_tensor.shape[0] == 1 and input_tensor.shape[1] == 3:
                    original_shape = input_tensor.shape
                    x_raw = input_tensor.reshape(-1, original_shape[-1])
                    x = x_raw.to(_module.compute_dtype).contiguous()
                    output = torch.empty((3, _n), device=x.device, dtype=torch.float16)
                    _fused_nf4_m3_kernel[_grid](x, *_kernel_args, output, _k, _n, int(_state.blocksize),
                                                _module.bias is not None, OUT_TILE=_out_tile,
                                                num_warps=_num_warps,
                                                enable_fp_fusion=False)
                    return output.to(input_tensor.dtype).reshape(*original_shape[:-1], _n)
                return _original(input_tensor, *args, **kwargs)

            saved.append((module, original_forward))
            module.forward = override
    try:
        yield
    finally:
        for module, original_forward in reversed(saved):
            module.forward = original_forward


def run_model_block(model, prompt_ids, forced_tokens, use_triton):
    cache = TargetKVCache(backend="static", max_capacity=int(prompt_ids.shape[-1]) + 8)
    tokens = torch.tensor([forced_tokens], device=prompt_ids.device, dtype=torch.long)
    with torch.inference_mode(), original_bnb_forward(model):
        if use_triton:
            with triton_m3_overrides(model):
                model(prompt_ids, past_key_values=cache.cache, use_cache=True)
                output = model(tokens, past_key_values=cache.cache, use_cache=True)
        else:
            model(prompt_ids, past_key_values=cache.cache, use_cache=True)
            output = model(tokens, past_key_values=cache.cache, use_cache=True)
    return output.logits.detach(), diagnostic_cache_digest(cache), cache


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

    token_cache = TargetKVCache(backend="static", max_capacity=int(prompt_ids.shape[-1]) + 8)
    with torch.inference_mode(), original_bnb_forward(model):
        logits = model(prompt_ids, past_key_values=token_cache.cache, use_cache=True).logits[:, -1:, :]
        forced_tokens = []
        for _ in range(3):
            token = logits.argmax(dim=-1)
            forced_tokens.append(int(token.item()))
            logits = model(token, past_key_values=token_cache.cache, use_cache=True).logits[:, -1:, :]

    bnb_logits, bnb_kv_digest, bnb_cache = run_model_block(model, prompt_ids, forced_tokens, use_triton=False)
    triton_logits, triton_kv_digest, triton_cache = run_model_block(model, prompt_ids, forced_tokens, use_triton=True)
    bnb_lengths = [int(layer.get_seq_length()) for layer in bnb_cache.cache.layers]
    triton_lengths = [int(layer.get_seq_length()) for layer in triton_cache.cache.layers]
    result = {
        "schema": "zassd-bnb-fused-nf4-activationvec-block-integration-v1",
        "classification": "development-only one-block integration check; not a closed-loop decoder proof",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_length": int(prompt_ids.shape[-1]), "forced_tokens": forced_tokens,
        "physical_rows": 3, "patched_linear_count": len(model.model.layers) * len(PROJECTIONS),
        "logits_shape": list(bnb_logits.shape), "logits_bitwise_equal": torch.equal(bnb_logits, triton_logits),
        "logits_max_abs_diff": float((bnb_logits.float() - triton_logits.float()).abs().max().item()),
        "argmax_tokens_bnb": bnb_logits.argmax(dim=-1).reshape(-1).tolist(),
        "argmax_tokens_triton": triton_logits.argmax(dim=-1).reshape(-1).tolist(),
        "kv_prefix_sha256_bnb": bnb_kv_digest, "kv_prefix_sha256_triton": triton_kv_digest,
        "kv_prefix_bitwise_equal": bnb_kv_digest == triton_kv_digest,
        "bnb_cache_lengths": bnb_lengths, "triton_cache_lengths": triton_lengths,
        "cache_lengths_equal": bnb_lengths == triton_lengths,
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_source": "scripts/benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2.py",
        "runtime": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"), "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["One teacher-forced M=3 verification block following native-BNB prefill.",
                         "Does not measure end-to-end decode latency or reject/rollback state-machine branches.",
                         "One prompt, one GPU/toolchain; GSM8K held-out indices 300-349 excluded."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["logits_bitwise_equal"] or not result["kv_prefix_bitwise_equal"] or not result["cache_lengths_equal"]:
        raise RuntimeError("Integrated M=3 Triton block diverged from native BNB logits or active KV")


if __name__ == "__main__":
    main()
