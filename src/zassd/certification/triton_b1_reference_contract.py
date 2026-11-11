"""Live validation and immutable handles for the experimental Triton B1 contract."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
ATTENTION_STRATEGY = "triton_batched"
LINEAR_STRATEGY = "triton_nf4_group_scale_packed_rows2_n16_warp1_k256"


class B1ReferenceContractError(RuntimeError):
    """The loaded model or selected decoding schedule violates the contract."""


@dataclass(frozen=True)
class VerifiedTritonB1Contract:
    """A handle issued only after live model/runtime/source identity passes."""

    contract_id: str
    contract_path: str
    model_revision: str
    model_state_sha256: str
    verified_source_count: int
    _model_object_id: int
    _tokenizer_object_id: int


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantized_model_state_sha256(model, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash model tensors and BNB quantization metadata without a full copy."""
    digest = hashlib.sha256()

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        digest.update(f"tensor:{name}:{tuple(tensor.shape)}:{tensor.dtype}:{tensor.layout}\n".encode())
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), max(1, chunk_bytes)):
            digest.update(raw[start:start + chunk_bytes].to(device="cpu").numpy().tobytes())

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
            if id(value) in seen:
                digest.update(f"cycle:{name}:{type(value).__module__}.{type(value).__qualname__}\n".encode())
                return
            seen.add(id(value))
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
        quant_state = getattr(getattr(module, "weight", None), "quant_state", None)
        if quant_state is not None:
            add_value(f"quant_state.{module_name}.weight", quant_state, seen)
    return digest.hexdigest()


def _live_identity(model, tokenizer, device: str) -> dict[str, Any]:
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
        "length": len(tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "vocab_sha256": hashlib.sha256(vocab_json.encode()).hexdigest(),
        "special_tokens_map": tokenizer.special_tokens_map,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_sha256": None if chat_template is None else
            hashlib.sha256(chat_template.encode()).hexdigest(),
    }
    flags = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
    }
    return {
        "model_type": model.config.model_type,
        "model_config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "quantized_state_sha256": _quantized_model_state_sha256(model),
        "tokenizer": tokenizer_identity,
        "runtime": {
            "gpu": {"name": props.name, "index": device_index,
                    "compute_capability": [props.major, props.minor],
                    "total_memory_bytes": props.total_memory, "driver": driver},
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
            "triton": importlib.metadata.version("triton"),
            "model_attention_implementation": getattr(model.config, "_attn_implementation", None),
            "backend_flags": flags,
        },
    }


def verify_triton_b1_reference_contract(
    model, tokenizer, contract_path: str | Path, *, device: str = "cuda:0",
) -> VerifiedTritonB1Contract:
    """Validate live identity, schedule, and pinned sources or raise closed."""
    path = Path(contract_path)
    contract = json.loads(path.read_text())
    payload = {key: value for key, value in contract.items() if key != "contract_id"}
    reasons: list[str] = []
    if _canonical_hash(payload) != contract.get("contract_id"):
        reasons.append("contract_id_integrity")
    if contract.get("schema") != "zassd-b1-triton-reference-v1":
        reasons.append("schema")
    if contract.get("model", {}).get("id") != MODEL_ID:
        reasons.append("model_id")
    if contract.get("model", {}).get("revision") != MODEL_REVISION:
        reasons.append("contract_revision")
    if getattr(model.config, "_commit_hash", None) != MODEL_REVISION:
        reasons.append("loaded_revision")

    schedule = contract.get("operator_schedule", {})
    if (schedule.get("physical_rows") != 2 or
            schedule.get("attention", {}).get("strategy") != ATTENTION_STRATEGY or
            schedule.get("quantized_linear", {}).get("strategy") != LINEAR_STRATEGY):
        reasons.append("operator_schedule")
    decoder = contract.get("decoder", {})
    if decoder.get("k") != 1 or decoder.get("ngram_size") != 3:
        reasons.append("decoder_schedule")

    live = _live_identity(model, tokenizer, device)
    if contract.get("model", {}).get("model_type") != live["model_type"]:
        reasons.append("model_type")
    if contract.get("model", {}).get("config_sha256") != live["model_config_sha256"]:
        reasons.append("model_config_sha256")
    if contract.get("weights", {}).get("quantized_state_sha256") != live[
        "quantized_state_sha256"
    ]:
        reasons.append("quantized_state_sha256")
    if contract.get("tokenizer") != live["tokenizer"]:
        reasons.append("tokenizer_identity")
    for key, value in live["runtime"].items():
        if contract.get("runtime", {}).get(key) != value:
            reasons.append(f"runtime.{key}")

    source_mismatches = []
    for name, expected in sorted(contract.get("source_sha256", {}).items()):
        source = Path(name)
        if not source.is_file() or _file_sha256(source) != expected:
            source_mismatches.append(name)
    if source_mismatches:
        reasons.append("source_hashes")
    if reasons:
        raise B1ReferenceContractError(
            "Triton B1 reference contract rejected: " + ", ".join(reasons)
        )
    return VerifiedTritonB1Contract(
        contract_id=contract["contract_id"],
        contract_path=str(path),
        model_revision=MODEL_REVISION,
        model_state_sha256=live["quantized_state_sha256"],
        verified_source_count=len(contract.get("source_sha256", {})),
        _model_object_id=id(model),
        _tokenizer_object_id=id(tokenizer),
    )


def require_verified_triton_b1_contract(
    value: VerifiedTritonB1Contract, model, tokenizer,
) -> None:
    if not isinstance(value, VerifiedTritonB1Contract):
        raise B1ReferenceContractError(
            "pass a VerifiedTritonB1Contract returned by live contract validation"
        )
    if value.model_revision != MODEL_REVISION or not value.contract_id:
        raise B1ReferenceContractError("verified contract handle is incomplete")
    if value._model_object_id != id(model) or value._tokenizer_object_id != id(tokenizer):
        raise B1ReferenceContractError(
            "verified contract handle belongs to a different model/tokenizer instance"
        )
