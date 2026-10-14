#!/usr/bin/env python3
"""Probe local backend availability and record compatibility evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
OUT = ROOT / "experiments/backend_validation/compatibility"
SOURCES = {
    "hf_hqq": "https://huggingface.co/docs/transformers/v5.13.0/en/quantization/hqq",
    "hqq_readme": "https://github.com/dropbox/hqq/blob/master/Readme.md",
    "hqq_torchao_adapter": "https://github.com/dropbox/hqq/blob/master/hqq/backends/torchao.py",
    "gemlite_readme": "https://github.com/dropbox/gemlite",
    "torchao_inference": "https://docs.pytorch.org/ao/stable/workflows/inference.html",
    "torchao_hqq_proto": "https://github.com/pytorch/ao/tree/main/torchao/prototype/hqq",
    "vllm_quantization": "https://docs.vllm.ai/en/stable/features/quantization/",
}


def package_info(name: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(name)
        installed = True
    except importlib.metadata.PackageNotFoundError:
        version, installed = None, False
    return {"installed": installed, "version": version}


def local_model_config() -> dict[str, Any]:
    try:
        cfg = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        return {"available_in_local_hf_cache": True, "model_type": cfg.model_type,
                "architectures": getattr(cfg, "architectures", None),
                "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
                "hidden_size": getattr(cfg, "hidden_size", None)}
    except Exception as exc:  # the audit must still run when weights/config are absent
        return {"available_in_local_hf_cache": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    adapter_source = (ROOT / "src/zassd/models/model_adapter.py").read_text()
    layer_manager_source = (ROOT / "src/zassd/models/layer_manager.py").read_text()
    custom_cache_source = (ROOT / "src/zassd/cache/kv_cache.py").read_text()
    speculative_source = (ROOT / "src/zassd/decoding/speculative.py").read_text()
    model_config = local_model_config()
    smoke_path = OUT / "hqq_gemlite_smoke.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    smoke_passed = bool(smoke and smoke.get("passed"))
    packages = {name: package_info(name) for name in (
        "bitsandbytes", "hqq", "gemlite", "torchao", "vllm", "tensorrt-llm")}
    local_interfaces = {
        "qwen_layer_path_supported": "Qwen2ForCausalLM" in adapter_source and "model.layers" in adapter_source,
        "layer_skip_context_manager_present": "def skip_layers" in layer_manager_source,
        "custom_cache_accepts_hf_cache_object": "class TargetKVCache" in custom_cache_source,
        "speculative_code_passes_custom_cache": "past_key_values=target_kv.cache" in speculative_source,
    }
    candidates = [
        {"backend": "HF + bitsandbytes NF4", "local_status": "tested_baseline",
         "gpu_qwen_support": "confirmed_by_existing_runs", "layer_skip_custom_verification": "confirmed_in_current_code",
         "notes": "Reference stack; 6 GB compatibility already demonstrated by WP1-WP4."},
        {"backend": "HF + HQQ W4A16 + GemLite", "local_status": "smoke_passed_via_manual_linear_patch" if smoke_passed else ("dependencies_missing" if not packages["hqq"]["installed"] or not packages["gemlite"]["installed"] else "installed_not_smoke_tested"),
         "gpu_qwen_support": "RTX_4050_Qwen2.5-3B_smoke_passed" if smoke_passed else "plausible_but_smoke_test_required",
         "layer_skip_custom_verification": "PLD_K4_and_CKA90_K2_static_KV_smoke_passed" if smoke_passed else "architecture_and_HF_cache_should_be_preserved; runtime_unverified",
         "notes": "HQQ replaces torch.nn.Linear modules and documents FP16, 4-bit, axis=1; GemLite uses Triton low-bit kernels. Built-in Transformers HQQ loading is not implemented in Transformers 5.14.1; a manual HQQ/GemLite linear patch retained the Qwen2 module tree and passed one-prompt vanilla, PLD K4 and CKA90/K2 smoke."},
        {"backend": "HF + HQQ torchao_int4 fused kernel", "local_status": "dependencies_missing" if not packages["hqq"]["installed"] or not packages["torchao"]["installed"] else "installed_not_smoke_tested",
         "gpu_qwen_support": "Ada_tested_by_HQQ_adapter_source",
         "layer_skip_custom_verification": "architecture_should_be_preserved; runtime_unverified",
         "notes": "HQQ's adapter says it was tested on Ada, matching RTX 4050's SM 8.9, but requires BF16 compute, axis=1, W4 and supported group sizes. This changes compute precision from the FP16 reference."},
        {"backend": "HF + torch.compile on BNB NF4", "local_status": "compile_not_attempted",
         "gpu_qwen_support": "same_model_and_gpu_as_baseline",
         "layer_skip_custom_verification": "dynamic_control_flow_and_cache_recompile_risk",
         "notes": "Preserves weight format, but BNB custom linear ops and speculative shape changes require a compile smoke test; not selected before kernel backend audit."},
        {"backend": "vLLM", "local_status": "installed" if packages["vllm"]["installed"] else "not_installed",
         "gpu_qwen_support": "package_present_but_model_and_6GB_smoke_unverified",
         "layer_skip_custom_verification": "custom_layer_skip_and_cache_intervention_not_established",
         "notes": "Use as serving-stack comparison later; not a controlled first integration for current layer-skip path."},
        {"backend": "TensorRT-LLM", "local_status": "installed" if packages["tensorrt-llm"]["installed"] else "not_installed",
         "gpu_qwen_support": "not_locally_verified",
         "layer_skip_custom_verification": "requires_engine_level_integration",
         "notes": "Not selected for first second-backend experiment because it would change cache and verification ownership."},
    ]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    result = {
        "audit_kind": "static_and_local_environment_screening",
        "execution_commit": commit,
        "model": MODEL, "model_revision": REVISION,
        "hardware": {"gpu": gpu, "compute_capability": list(cap) if cap else None,
                     "cuda_available": torch.cuda.is_available(),
                     "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "transformers": transformers.__version__,
                     "packages": packages},
        "model_config": model_config,
        "current_code_interfaces": local_interfaces,
        "candidates": candidates,
        "runtime_smoke": ({
            "artifact": str(smoke_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
            "passed": smoke_passed,
            "loader_path": smoke.get("loader_path"),
            "hqq_version": smoke.get("hqq_version"),
            "gemlite_version": smoke.get("gemlite_version"),
            "vanilla_static_kv_passed": smoke["checks"]["vanilla_static_kv"]["passed"],
            "pld_k4_ran_and_exact": (smoke["checks"]["pld_k4_batched_verification"]["ran"]
                                       and smoke["checks"]["pld_k4_batched_verification"]["within_backend_exact_vs_vanilla"]),
            "cka90_k2_ran_and_exact": (smoke["checks"]["cka90_k2_layer_skip"]["ran"]
                                        and smoke["checks"]["cka90_k2_layer_skip"]["within_backend_exact_vs_vanilla"]),
            "peak_allocated_mb": smoke["checks"]["vanilla_static_kv"].get("peak_allocated_mb"),
        } if smoke else None),
        "recommended_next_candidate": "HQQ W4A16 + GemLite linear kernels in the existing Transformers Qwen2 model",
        "decision": ("HQQ+GemLite passes the one-prompt runtime compatibility gate through its manual Transformers linear patch: Qwen2.5-3B loaded on RTX 4050, 36-layer access remained intact, static KV vanilla ran, and PLD K4 plus CKA90/K2 completed with exact token agreement against this backend's vanilla output. The built-in Transformers HQQ quantization_config route is blocked by an unimplemented get_quantize_ops() in Transformers 5.14.1; preserve the manual patch path and verify across more prompts before performance integration." if smoke_passed else "Candidate passes API-structure screening but is NOT yet cleared for integration. Install/probe HQQ+GemLite, load Qwen2.5-3B on this SM 8.9 GPU, then test vanilla generation, custom static KV update, intermediate-layer access and batched verification in smoke tests."),
        "sources": SOURCES,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Backend compatibility audit", "",
             f"Model revision: `{REVISION}`. GPU: `{gpu}` (compute capability `{cap}`).",
             f"Software: PyTorch `{torch.__version__}`, CUDA `{torch.version.cuda}`, Transformers `{transformers.__version__}`.", "",
             "## Local probe", "",
             f"Model config: `{model_config}`.",
             f"Current ZASSD APIs: `{local_interfaces}`.", "",
             "| Candidate | Local state | RTX 4050 / Qwen support | Layer skipping + custom verification |",
             "|---|---|---|---|"]
    for c in candidates:
        lines.append(f"| {c['backend']} | {c['local_status']} | {c['gpu_qwen_support']} | {c['layer_skip_custom_verification']} |")
    lines += ["", "## Decision", "", result["decision"], "",
              f"Runtime smoke record: `{result['runtime_smoke']}`.", "",
              "HQQ+GemLite is the leading candidate because it replaces linear layers while preserving the Hugging Face model tree, intermediate layer list and cache API. Official documentation covers HQQ 4-bit FP16 at axis=1 and GemLite Triton kernels, while the local smoke supplies the RTX 4050/Qwen2.5-3B evidence.",
              "HQQ's torchao fused INT4 adapter has explicit Ada testing but requires BF16 compute, so it is a useful alternative rather than a same-precision comparison.",
              "The smoke is a compatibility result only. Its first-use latency includes JIT/kernel setup and CPU-side quantization; do not use it as a paired performance result.", "",
              "## Primary sources", ""]
    for name, url in SOURCES.items():
        lines.append(f"- [{name}]({url})")
    lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"output_dir": str(args.output_dir), "candidate": result["recommended_next_candidate"],
                      "gpu": gpu, "compute_capability": cap, "model_config": model_config}, indent=2))


if __name__ == "__main__":
    main()
