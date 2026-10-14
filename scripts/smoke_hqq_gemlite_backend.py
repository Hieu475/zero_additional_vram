#!/usr/bin/env python3
"""Smoke-test HQQ W4A16 + GemLite inside the existing HF/ZASSD model path."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hqq.utils.patching import prepare_for_inference  # noqa: E402
from zassd.baselines.prompt_lookup import prompt_lookup_generate  # noqa: E402
from zassd.cache.kv_cache import TargetKVCache  # noqa: E402
from zassd.decoding.speculative import self_speculative_generate  # noqa: E402
from zassd.models.layer_manager import LayerManager  # noqa: E402
from zassd.models.model_adapter import ModelAdapter  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
DEFAULT_INPUTS = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl"
DEFAULT_OUT = ROOT / "experiments/backend_validation/compatibility/hqq_gemlite_smoke.json"
CKA90_SKIP = [4, 5, 6, 7]


def generate_vanilla(model: Any, prompt_ids: torch.Tensor, eos_id: int | None,
                     budget: int, capacity: int, device: str) -> dict[str, Any]:
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    generated: list[int] = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
        torch.cuda.synchronize(device)
        prefill_s = time.perf_counter() - start
        next_token = int(out.logits[0, -1].argmax().item())
        decode_start = time.perf_counter()
        while len(generated) < budget:
            generated.append(next_token)
            if next_token == eos_id or len(generated) >= budget:
                break
            token = torch.tensor([[next_token]], dtype=torch.long, device=device)
            out = model(token, past_key_values=cache.cache, use_cache=True)
            next_token = int(out.logits[0, -1].argmax().item())
        torch.cuda.synchronize(device)
        total_s = time.perf_counter() - start
    return {"token_ids": generated, "prefill_s": prefill_s,
            "decode_s": max(total_s - prefill_s, 0.0), "total_s": total_s,
            "tokens_per_second": len(generated) / max(total_s, 1e-12),
            "final_kv_length": cache.get_seq_length(),
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--prompt-id", default="qasper_119_ctx384")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    result: dict[str, Any] = {
        "model": MODEL, "model_revision": REVISION,
        "backend": "Transformers HQQ W4A16 group_size=64 axis=1 + GemLite",
        "hardware": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(args.device)) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "hqq_version": importlib.metadata.version("hqq"),
        "gemlite_version": importlib.metadata.version("gemlite"),
        "prompt_id": args.prompt_id, "max_new_tokens": args.max_new_tokens,
        "checks": {},
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        rows = [json.loads(x) for x in args.inputs.read_text().splitlines() if x.strip()]
        row = next(r for r in rows if r["prompt_id"] == args.prompt_id)
        ids = list(row["input_ids"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        qconfig = HqqConfig(nbits=4, group_size=64, axis=1)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, quantization_config=qconfig,
                dtype=torch.float16, device_map=args.device,
                attn_implementation="sdpa").eval()
            prepare_for_inference(model, backend="gemlite")
            result["loader_path"] = "transformers_hqq_quantizer"
        except NotImplementedError as exc:
            # Transformers 5.14 exposes HqqConfig but its get_quantize_ops()
            # path raises NotImplementedError. The supported HQQ/GemLite
            # module patcher can preserve the HF model tree as a fallback.
            result["transformers_loader_blocker"] = str(exc)
            from gemlite.helper import A16W4_HQQ_INT, patch_model

            model = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=REVISION, dtype=torch.float16,
                device_map="cpu", low_cpu_mem_usage=True,
                attn_implementation="sdpa").eval()
            model = patch_model(
                model, device=args.device,
                processor=A16W4_HQQ_INT(device=args.device, dtype=torch.float16),
                group_size=64)
            result["loader_path"] = "manual_hqq_gemlite_linear_patch"
        adapter = ModelAdapter(model)
        manager = LayerManager(adapter)
        hqq_linear_count = sum(type(m).__name__ == "HQQLinear" for m in model.modules())
        gemlite_linear_count = sum("GemLite" in type(m).__name__ for m in model.modules())
        result["checks"]["model_load_and_gemlite_patch"] = {
            "passed": hqq_linear_count > 0 or gemlite_linear_count > 0,
            "architecture": type(model).__name__,
            "layer_count": adapter.num_layers,
            "hqq_linear_count": hqq_linear_count,
            "gemlite_named_module_count": gemlite_linear_count,
            "module_names_sample": [name for name, mod in model.named_modules()
                                    if "GemLite" in type(mod).__name__][:5],
        }
        prompt = torch.tensor([ids], dtype=torch.long, device=args.device)
        capacity = len(ids) + args.max_new_tokens + 20
        vanilla = generate_vanilla(model, prompt, tokenizer.eos_token_id,
                                   args.max_new_tokens, capacity, args.device)
        result["checks"]["vanilla_static_kv"] = {
            "passed": len(vanilla["token_ids"]) > 0 and vanilla["final_kv_length"] == len(ids) + len(vanilla["token_ids"]) - 1,
            **vanilla,
        }
        _, pld_metrics = prompt_lookup_generate(
            model, tokenizer, prompt, k=4, ngram_size=3,
            max_new_tokens=args.max_new_tokens, temperature=0.0,
            device=args.device, kv_cache_backend="static",
            candidate_source="prompt+generated", kv_cache_max_capacity=capacity)
        pld_ids = list(pld_metrics.generated_token_ids)
        result["checks"]["pld_k4_batched_verification"] = {
            "ran": len(pld_ids) > 0,
            "within_backend_exact_vs_vanilla": pld_ids == vanilla["token_ids"],
            "first_divergence_vs_vanilla": next((i for i, (a, b) in enumerate(zip(pld_ids, vanilla["token_ids"])) if a != b),
                                                 min(len(pld_ids), len(vanilla["token_ids"])) if len(pld_ids) != len(vanilla["token_ids"]) else None),
            "token_ids": pld_ids,
            "verification_cycles": int(pld_metrics.num_verification_cycles),
        }
        _, cka_metrics = self_speculative_generate(
            model, tokenizer, manager, CKA90_SKIP, prompt, k=2,
            max_new_tokens=args.max_new_tokens, temperature=0.0,
            device=args.device, kv_cache_backend="static",
            kv_cache_max_capacity=capacity, draft_mode="layer_skip",
            config_name="cka_90")
        cka_ids = list(cka_metrics.generated_token_ids)
        result["checks"]["cka90_k2_layer_skip"] = {
            "ran": len(cka_ids) > 0,
            "skip_indices": CKA90_SKIP,
            "within_backend_exact_vs_vanilla": cka_ids == vanilla["token_ids"],
            "first_divergence_vs_vanilla": next((i for i, (a, b) in enumerate(zip(cka_ids, vanilla["token_ids"])) if a != b),
                                                 min(len(cka_ids), len(vanilla["token_ids"])) if len(cka_ids) != len(vanilla["token_ids"]) else None),
            "token_ids": cka_ids,
            "verification_cycles": int(cka_metrics.num_verification_cycles),
        }
        result["passed"] = all(
            check.get("passed", check.get("ran", False))
            and check.get("within_backend_exact_vs_vanilla", True)
            for check in result["checks"].values())
    except Exception as exc:
        result["passed"] = False
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "error": result.get("error"),
                      "checks": {k: {x: v for x, v in val.items() if x in ("passed", "ran", "layer_count", "within_backend_exact_vs_vanilla")}
                                 for k, val in result["checks"].items()}}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
