#!/usr/bin/env python3
"""Compare experimental fixed-row B1 blocks with its singleton reference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
from collections.abc import Mapping

import torch

try:
    from run_strong_baseline_comparison import run_vanilla_timed
except ModuleNotFoundError:
    from scripts.run_strong_baseline_comparison import run_vanilla_timed
from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from zassd.decoding.speculative import diagnostic_cache_digest
from build_b1_reference_contract import canonical_hash, reference_payload


INPUTS = {
    "gsm8k": "data/benchmarks/gsm8k_eval.jsonl",
    "humaneval": "data/benchmarks/humaneval_eval.jsonl",
    "cnndm": "data/benchmarks/cnndm_eval.jsonl",
}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostic_cache_prefix_digest(target_kv, prefix_length: int) -> str:
    """Hash a logical active-cache prefix without changing cache lengths."""
    digest = hashlib.sha256()
    for layer in target_kv.cache.layers:
        active_length = layer.get_seq_length()
        if prefix_length > active_length:
            raise ValueError(
                f"requested cache prefix {prefix_length} exceeds active length {active_length}"
            )
        digest.update(prefix_length.to_bytes(8, "little"))
        if not getattr(layer, "is_initialized", False):
            continue
        for tensor in (layer.keys, layer.values):
            raw_bytes = tensor[:, :, :prefix_length, :].detach().contiguous().view(torch.uint8)
            digest.update(raw_bytes.cpu().numpy().tobytes())
    return digest.hexdigest()


def quantized_model_state_sha256(model, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash model tensors and BNB quantization metadata without a whole-model copy."""
    digest = hashlib.sha256()

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        digest.update(f"tensor:{name}:{tuple(tensor.shape)}:{tensor.dtype}:{tensor.layout}\n".encode())
        value = tensor.detach().contiguous()
        raw = value.reshape(-1).view(torch.uint8).reshape(-1)
        chunk_size = max(1, chunk_bytes)
        for start in range(0, raw.numel(), chunk_size):
            cpu_chunk = raw[start:start + chunk_size].to(device="cpu")
            digest.update(cpu_chunk.numpy().tobytes())

    def add_value(name: str, value, seen: set[int]) -> None:
        if isinstance(value, torch.Tensor):
            add_tensor(name, value)
        elif value is None or isinstance(value, (str, int, float, bool, torch.dtype)):
            digest.update(f"value:{name}:{type(value).__name__}:{value!r}\n".encode())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                add_value(f"{name}.{key}", value[key], seen)
        elif isinstance(value, (tuple, list)):
            digest.update(f"sequence:{name}:{type(value).__name__}:{len(value)}\n".encode())
            for index, item in enumerate(value):
                add_value(f"{name}[{index}]", item, seen)
        elif hasattr(value, "__dict__"):
            identity = id(value)
            if identity in seen:
                digest.update(f"cycle:{name}:{type(value).__module__}.{type(value).__qualname__}\n".encode())
                return
            seen.add(identity)
            digest.update(f"object:{name}:{type(value).__module__}.{type(value).__qualname__}\n".encode())
            for key, item in sorted(vars(value).items()):
                add_value(f"{name}.{key}", item, seen)
        else:
            digest.update(f"repr:{name}:{type(value).__module__}.{type(value).__qualname__}:{value!r}\n".encode())

    for name, parameter in sorted(model.named_parameters()):
        add_tensor(f"parameter.{name}", parameter)
    for name, buffer in sorted(model.named_buffers()):
        add_tensor(f"buffer.{name}", buffer)
    seen: set[int] = set()
    for module_name, module in sorted(model.named_modules()):
        weight = getattr(module, "weight", None)
        quant_state = getattr(weight, "quant_state", None)
        if quant_state is not None:
            add_value(f"quant_state.{module_name}.weight", quant_state, seen)
    return digest.hexdigest()


def runtime_identity(model, tokenizer, device: str, package_version) -> dict:
    props = torch.cuda.get_device_properties(device)
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    driver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_index}",
             "--query-gpu=driver_version,compute_cap,uuid",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        fields = [value.strip() for value in result.stdout.strip().split(",")]
        if len(fields) == 3:
            driver = {"version": fields[0], "compute_capability": fields[1], "uuid": fields[2]}
    except (OSError, subprocess.CalledProcessError):
        pass

    config_json = json.dumps(model.config.to_dict(), sort_keys=True, default=str,
                             separators=(",", ":"))
    vocab_json = json.dumps(sorted(tokenizer.get_vocab().items()), ensure_ascii=False,
                            separators=(",", ":"))
    chat_template = getattr(tokenizer, "chat_template", None)
    tokenizer_identity = {
        "class": type(tokenizer).__name__,
        "name_or_path": tokenizer.name_or_path,
        "length": len(tokenizer), "vocab_size": tokenizer.vocab_size,
        "vocab_sha256": hashlib.sha256(vocab_json.encode()).hexdigest(),
        "special_tokens_map": tokenizer.special_tokens_map,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_sha256": None if chat_template is None else
            hashlib.sha256(chat_template.encode()).hexdigest(),
    }
    adapter_module = {
        "qwen2": "src/zassd/certification/qwen2_padded.py",
        "llama": "src/zassd/certification/llama_padded.py",
        "phi3": "src/zassd/certification/phi3_padded.py",
        "olmo2": "src/zassd/certification/olmo2_padded.py",
    }.get(model.config.model_type)
    source_files = tuple(path for path in (
        adapter_module,
        "src/zassd/certification/fixed_row_dispatch.py",
        "src/zassd/certification/model_capabilities.py",
        "src/zassd/decoding/b1_exact.py",
        "src/zassd/decoding/ngram.py",
        "src/zassd/cache/kv_cache.py",
        "src/zassd/backends/base.py",
        "scripts/run_b1_block_pilot.py",
    ) if path is not None)
    source_hashes = {name: file_sha256(name) for name in source_files}
    backend_flags = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
    }
    identity = {
        "gpu": {"name": props.name, "index": device_index,
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory, "driver": driver},
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": package_version("transformers"),
        "bitsandbytes": package_version("bitsandbytes"),
        "model_type": model.config.model_type,
        "model_config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "model_attention_implementation": getattr(model.config, "_attn_implementation", None),
        "tokenizer": tokenizer_identity, "backend_flags": backend_flags,
        "source_sha256": source_hashes,
        "quantized_state_sha256": quantized_model_state_sha256(model),
    }
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-file", default=None,
                        help="Optional JSONL development prompts with id and prompt fields")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--model-dtype", choices=("config", "float16"), default="config",
                        help="Optional unquantized-module/activation dtype experiment")
    parser.add_argument("--reuse-linear-padding-buffer", action="store_true",
                        help="Experimental Llama-only reusable linear padding buffers")
    parser.add_argument("--quantized-linear-strategy", choices=("fixed_rows", "rowwise"), default="fixed_rows",
                        help="Experimental Llama BNB linear path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-task", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Starting row in each task's development file")
    parser.add_argument("--session-index", type=int, default=0,
                        help="Independent process-session index; rotates method order across sessions")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--physical-rows", type=int, default=9,
                        help="Pinned B1 reference M dimension; keep fixed while tuning K")
    parser.add_argument("--attention-strategy", choices=("rowwise", "padded"), default="rowwise")
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--audit-state", action="store_true")
    parser.add_argument("--reference-contract", default=None,
                        help="Fail closed unless the loaded runtime matches this pinned B1 reference")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    if (not torch.cuda.is_available() or args.k < 1 or args.max_new_tokens < 1 or
            args.per_task < 1 or args.start_index < 0):
        raise ValueError("CUDA and positive k/output budget are required")
    if args.physical_rows < args.k + 1:
        raise ValueError("physical-rows must be at least k+1")

    prompts = []
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        data = [json.loads(line) for line in prompt_path.read_text().splitlines() if line.strip()]
        selected = data[args.start_index:args.start_index + args.per_task]
        if len(selected) != args.per_task:
            raise ValueError(f"{prompt_path} has fewer than {args.per_task} rows from start index {args.start_index}")
        for row in selected:
            prompt_text = row.get("prompt")
            if not isinstance(prompt_text, str) or not prompt_text:
                raise ValueError(f"prompt row {row.get('id')} has no nonempty prompt string")
            prompts.append((str(row.get("task", prompt_path.stem)), str(row["id"]), prompt_text))
        input_hashes = {str(prompt_path): file_sha256(prompt_path)}
    else:
        for task, path in INPUTS.items():
            data = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
            selected = data[args.start_index:args.start_index + args.per_task]
            if len(selected) != args.per_task:
                raise ValueError(f"{path} has fewer than {args.per_task} rows from start index {args.start_index}")
            prompts.extend((task, str(row["id"]), row["prompt"]) for row in selected)
        input_hashes = {task: file_sha256(path) for task, path in INPUTS.items()}
    model_dtype = torch.float16 if args.model_dtype == "float16" else None
    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa",
                          model_dtype=model_dtype)
    model, tokenizer = loaded.model, loaded.tokenizer

    def package_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    manifest = {
        "manifest_schema": "b1-fixed-row-v2",
        "status": "experimental; finite diagnostic, not a proof",
        "reference": "greedy singleton under the same architecture-specific fixed-row operator schedule",
        "second_baseline": "original BNB greedy, for compatibility measurement only",
        "model": args.model, "revision": args.revision,
        "model_dtype_override": args.model_dtype,
        "reuse_linear_padding_buffer": args.reuse_linear_padding_buffer,
        "quantized_linear_strategy": args.quantized_linear_strategy,
        "backend": loaded.metadata, "gpu": torch.cuda.get_device_name(args.device),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": package_version("transformers"),
        "bitsandbytes": package_version("bitsandbytes"),
        "k": args.k, "physical_rows": args.physical_rows,
        "attention_strategy": args.attention_strategy,
        "reuse_linear_padding_buffer": args.reuse_linear_padding_buffer,
        "quantized_linear_strategy": args.quantized_linear_strategy,
        "ngram_size": 3,
        "reference_schedule": f"{model.config.model_type}_fixed_row_{args.attention_strategy}_singleton",
        "cache_backend": "StaticPreallocatedKVCache",
        "greedy_tie_rule": "torch.argmax_first_index",
        "max_new_tokens": args.max_new_tokens, "capacity": args.capacity,
        "per_task": args.per_task, "start_index": args.start_index,
        "session_index": args.session_index,
        "prompt_file": args.prompt_file,
        "eos_token_id": tokenizer.eos_token_id,
        "audit_state": args.audit_state,
        "state_audit_granularity": "every_output_prefix" if args.audit_state else None,
        "input_hashes": input_hashes,
    }
    runtime = runtime_identity(model, tokenizer, args.device, package_version)
    manifest["runtime_identity"] = runtime
    if args.reference_contract:
        reference_path = Path(args.reference_contract)
        reference = json.loads(reference_path.read_text())
        expected = {key: value for key, value in reference.items()
                    if key not in {"reference_contract_id", "provenance"}}
        actual = reference_payload(manifest)
        if actual != expected or canonical_hash(expected) != reference.get("reference_contract_id"):
            mismatches = [key for key in expected if expected[key] != actual.get(key)]
            if canonical_hash(expected) != reference.get("reference_contract_id"):
                mismatches.append("reference_contract_id integrity")
            raise ValueError(
                f"runtime does not satisfy {reference_path}: {', '.join(mismatches)}"
            )
        manifest["reference_contract_id"] = reference["reference_contract_id"]
        manifest["reference_contract_path"] = str(reference_path)
    contract_payload = {
        "model": args.model, "revision": args.revision,
        "backend": loaded.metadata, "runtime_identity": runtime,
        "k": args.k, "physical_rows": args.physical_rows,
        "attention_strategy": args.attention_strategy,
        "reuse_linear_padding_buffer": args.reuse_linear_padding_buffer,
        "quantized_linear_strategy": args.quantized_linear_strategy,
        "ngram_size": 3, "cache_backend": "StaticPreallocatedKVCache",
        "greedy_tie_rule": "torch.argmax_first_index",
        "capacity": args.capacity, "max_new_tokens": args.max_new_tokens,
        "input_hashes": input_hashes, "prompt_file": args.prompt_file,
        "per_task": args.per_task, "start_index": args.start_index,
    }
    manifest["contract_id"] = hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True, default=str,
                   separators=(",", ":")).encode()
    ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows = []
    for index, (task, prompt_id, prompt) in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cpu()
        if ids.shape[1] + args.max_new_tokens > args.capacity:
            raise ValueError(f"{prompt_id} exceeds KV capacity")
        results = {}
        methods = ["original_greedy", "b1_reference", "b1_block"]
        order_offset = (index + args.session_index) % len(methods)
        methods = methods[order_offset:] + methods[:order_offset]
        for method in methods:
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            if method == "original_greedy":
                tokens, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
                    model, tokenizer, ids, args.max_new_tokens, args.device)
                results[method] = {"token_ids": tokens, "total_s": total_ms / 1000,
                                  "prefill_s": prefill_ms / 1000,
                                  "decode_s": decode_ms / 1000}
            elif method == "b1_reference":
                _, metrics = b1_fixed_row_generate(
                    model, tokenizer, ids, k=args.k,
                    max_new_tokens=args.max_new_tokens, device=args.device,
                    kv_cache_max_capacity=args.capacity,
                    physical_rows=args.physical_rows,
                    attention_strategy=args.attention_strategy,
                    reuse_linear_padding_buffer=args.reuse_linear_padding_buffer,
                    quantized_linear_strategy=args.quantized_linear_strategy)
                results[method] = {"token_ids": metrics.generated_token_ids,
                                   "total_s": metrics.total_time_s,
                                   "prefill_s": metrics.prefill_time_s,
                                   "target_forwards": metrics.target_forwards,
                                   "target_rows": metrics.target_rows,
                                   "physical_rows": metrics.physical_rows}
            else:
                _, metrics = b1_fixed_row_block_generate(
                    model, tokenizer, ids, k=args.k,
                    max_new_tokens=args.max_new_tokens, device=args.device,
                    kv_cache_max_capacity=args.capacity,
                    physical_rows=args.physical_rows,
                    attention_strategy=args.attention_strategy,
                    reuse_linear_padding_buffer=args.reuse_linear_padding_buffer,
                    quantized_linear_strategy=args.quantized_linear_strategy,
                    trace=True)
                results[method] = {"token_ids": metrics.generated_token_ids,
                                   "total_s": metrics.total_time_s,
                                   "prefill_s": metrics.prefill_time_s,
                                   "verify_s": metrics.verify_time_s,
                                   "target_forwards": metrics.target_forwards,
                                   "target_rows": metrics.target_rows,
                                   "physical_rows": metrics.physical_rows,
                                   "draft_tokens": metrics.draft_tokens,
                                   "accepted_draft_tokens": metrics.accepted_draft_tokens,
                                   "cycles": metrics.cycle_trace}
            results[method]["peak_allocated_mb"] = torch.cuda.max_memory_allocated(args.device) / 1e6
            token_ids = results[method]["token_ids"]
            if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                terminal_reason = "eos"
            elif len(token_ids) == args.max_new_tokens:
                terminal_reason = "budget"
            else:
                terminal_reason = "unexpected_stop"
            results[method]["terminal_reason"] = terminal_reason

        row = {
            "task": task, "id": prompt_id,
            "input_token_ids_sha256": hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
            "prompt_token_count": int(ids.shape[1]),
            "session_index": args.session_index,
            "method_order": methods, "results": results,
            "block_matches_b1": results["b1_block"]["token_ids"] == results["b1_reference"]["token_ids"],
            "block_matches_original_bnb": results["b1_block"]["token_ids"] == results["original_greedy"]["token_ids"],
            "block_speedup_vs_b1": results["b1_reference"]["total_s"] / results["b1_block"]["total_s"],
            "block_speedup_vs_original": results["original_greedy"]["total_s"] / results["b1_block"]["total_s"],
        }
        if args.audit_state:
            reference_states, block_states = [], []
            b1_fixed_row_generate(
                model, tokenizer, ids, k=args.k, max_new_tokens=args.max_new_tokens,
                device=args.device, kv_cache_max_capacity=args.capacity,
                physical_rows=args.physical_rows,
                attention_strategy=args.attention_strategy,
                state_observer=lambda n, c: reference_states.append((n, diagnostic_cache_digest(c))),
            )
            def observe_all_block_prefixes(n, cache):
                if not block_states:
                    prefix_counts = [0]
                else:
                    prefix_counts = range(block_states[-1][0] + 1, n + 1)
                for prefix_count in prefix_counts:
                    block_states.append((
                        prefix_count,
                        diagnostic_cache_prefix_digest(cache, int(ids.shape[1]) + prefix_count),
                    ))

            b1_fixed_row_block_generate(
                model, tokenizer, ids, k=args.k, max_new_tokens=args.max_new_tokens,
                device=args.device, kv_cache_max_capacity=args.capacity,
                physical_rows=args.physical_rows,
                attention_strategy=args.attention_strategy,
                state_observer=observe_all_block_prefixes,
            )
            ref_map, block_map = dict(reference_states), dict(block_states)
            shared = sorted(set(ref_map) & set(block_map))
            row["state_audit"] = {
                "shared_prefixes": len(shared),
                "all_shared_states_equal": all(ref_map[n] == block_map[n] for n in shared),
                "state_exact_at_all_reference_prefixes": set(ref_map) == set(block_map) and
                    all(ref_map[n] == block_map[n] for n in ref_map),
                "reference_prefix_counts": sorted(ref_map),
                "block_prefix_counts": sorted(block_map),
                "reference_state_trace": [
                    {"prefix_length": n, "cache_sha256": ref_map[n]}
                    for n in sorted(ref_map)
                ],
            }
        rows.append(row)
        with (output / "rows.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"{index + 1}/{len(prompts)} {task} {prompt_id}: "
              f"B1={row['block_matches_b1']}, BNB={row['block_matches_original_bnb']}, "
              f"speedup={row['block_speedup_vs_b1']:.3f}", flush=True)

    summary = {
        "n": len(rows),
        "exact_vs_b1_singleton": sum(row["block_matches_b1"] for row in rows),
        "exact_vs_original_bnb": sum(row["block_matches_original_bnb"] for row in rows),
        "mean_speedup_vs_b1_singleton": sum(row["block_speedup_vs_b1"] for row in rows) / len(rows),
        "mean_speedup_vs_original_bnb": sum(row["block_speedup_vs_original"] for row in rows) / len(rows),
        "scope": "diagnostic pilot; empirical match does not establish universal operator equivalence",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
