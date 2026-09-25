"""Real hardware telemetry collection and offline closed-loop replay for controller validation.

Demonstrates the trajectory of controller adaptations:
  Telemetry Trace (Temp, Power, VRAM, Entropy) -> Controller Action Sequence (S_t, K_t).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zassd.cache.kv_cache import TargetKVCache
from zassd.controllers.entropy import compute_entropy
from zassd.controllers.hardware_controller import (
    ControllerAction,
    HardwareAwareJointController,
    HardwareState,
)
from zassd.models.loader import load_model, load_tokenizer
from zassd.profiling.gpu import GPUProfiler
from zassd.profiling.memory import get_vram_usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def collect_real_generation_telemetry(
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
    num_steps: int = 25,
    device: str = "cuda",
) -> list[dict[str, float]]:
    """Execute real GPU generation and log telemetry trace per step."""
    logger.info(f"Loading model {model_name_or_path} for real telemetry capture...")
    model = load_model(model_name_or_path, quantize=True, bits=4)
    tokenizer = load_tokenizer(model_name_or_path)
    model.eval()

    profiler = GPUProfiler()

    prompt = (
        "Discuss in detail the architectural tradeoffs between memory bandwidth, "
        "quantization precision, and cache coherence in distributed edge computing."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]

    target_kv = TargetKVCache()
    with torch.no_grad():
        out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
    curr_logit = out.logits[0, -1, :].float()
    curr_token = int(curr_logit.argmax(dim=-1).item())

    telemetry_trace: list[dict[str, float]] = []

    logger.info(f"Generating {num_steps} tokens with real hardware telemetry logging...")
    for step in range(1, num_steps + 1):
        # Sample hardware state right before generation step
        snap = profiler.snapshot()
        vram_info = get_vram_usage()

        power_w = snap.get("power_w", 50.0)
        temp_c = snap.get("temperature_c", 55.0)
        vram_mb = vram_info.get("allocated_mb", 2000.0)
        entropy = float(compute_entropy(curr_logit).item())

        trace_item = {
            "step": step,
            "gpu_temperature_c": round(temp_c, 1),
            "gpu_power_w": round(power_w, 1),
            "vram_used_mb": round(vram_mb, 1),
            "entropy": round(entropy, 3),
        }
        telemetry_trace.append(trace_item)

        # Autoregressive forward pass
        token_tensor = torch.tensor([[curr_token]], device=device)
        with torch.no_grad():
            fwd_out = model(token_tensor, past_key_values=target_kv.cache, use_cache=True)
        torch.cuda.synchronize()

        curr_logit = fwd_out.logits[0, -1, :].float()
        curr_token = int(curr_logit.argmax(dim=-1).item())
        if curr_token == tokenizer.eos_token_id:
            break

    return telemetry_trace


def replay_telemetry_through_controller(
    telemetry_trace: list[dict[str, float]],
    model_name: str = "qwen25_3b",
    out_dir: str = "experiments/13_telemetry_replay",
) -> list[dict[str, Any]]:
    """Replay collected telemetry trace through HardwareAwareJointController."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    configs = {
        "cka_90": [3, 7, 12, 19],
        "cka_83": [3, 7, 12, 16, 20, 24],
        "cka_75": [3, 5, 7, 9, 11, 13, 16, 18, 21],
        "cka_60": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        "cka_50": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21],
    }

    controller = HardwareAwareJointController(
        candidate_layer_configs=configs,
        model_name=model_name,
        candidate_k_values=[1, 2, 3, 4],
    )

    trajectory: list[dict[str, Any]] = []

    for item in telemetry_trace:
        step = item["step"]
        temp = item["gpu_temperature_c"]
        power = item["gpu_power_w"]
        vram = item["vram_used_mb"]
        entropy = item["entropy"]

        hw_state = HardwareState(
            vram_used_mb=vram,
            gpu_power_w=power,
            gpu_temperature_c=temp,
        )

        action: ControllerAction = controller.select_action(
            entropy=entropy,
            last_accepted=1,
            last_proposed=2,
            draft_ms=18.0,
            verify_ms=23.0,
            hardware_state=hw_state,
        )

        record = {
            "step": step,
            "temp_c": temp,
            "power_w": power,
            "vram_gb": round(vram / 1024.0, 2),
            "entropy": entropy,
            "selected_config": action.config_name,
            "selected_k": action.draft_length,
            "predicted_utility": round(action.predicted_utility, 3),
            "expected_cycle_ms": round(action.expected_cycle_ms, 2),
            "expected_energy_j_tok": round(action.expected_energy_j_tok, 3),
        }
        trajectory.append(record)

    out_file = Path(out_dir) / "trajectory_replay_summary.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "num_steps": len(trajectory),
                "telemetry_trace": telemetry_trace,
                "trajectory": trajectory,
            },
            f,
            indent=2,
        )
    logger.info(f"Trajectory replay summary saved to {out_file}")

    print("\n" + "=" * 92)
    print(f"HARDWARE-AWARE CONTROLLER TELEMETRY REPLAY TRAJECTORY ({model_name.upper()})")
    print("=" * 92)
    print(
        f"{'Step':<5} | {'Temp (°C)':<9} | {'Power (W)':<9} | {'VRAM (GB)':<9} | "
        f"{'Entropy':<8} | {'Selected S':<10} | {'K':<3} | {'Utility':<8} | {'Cycle (ms)':<10}"
    )
    print("-" * 92)
    for r in trajectory:
        print(
            f"{r['step']:<5} | {r['temp_c']:>8.1f}° | {r['power_w']:>8.1f}W | {r['vram_gb']:>8.2f}G | "
            f"{r['entropy']:>8.2f} | {r['selected_config']:<10} | {r['selected_k']:<3} | "
            f"{r['predicted_utility']:>+7.2f} | {r['expected_cycle_ms']:>8.2f}ms"
        )
    print("=" * 92)

    return trajectory


def run_full_telemetry_experiment():
    trace = collect_real_generation_telemetry(num_steps=20)
    replay_telemetry_through_controller(trace)


if __name__ == "__main__":
    run_full_telemetry_experiment()
