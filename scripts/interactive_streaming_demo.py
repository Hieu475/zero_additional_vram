"""Phase 14 / Việc 8 — Interactive Streaming Generation Demo with Real-Time Systems Telemetry.

Demonstrates end-to-end ZASSD system pipeline:
HF Model (4-bit NF4) -> ZASSD -> Runtime HardwareAware Controller -> Streaming Token Generation

Real-time Dashboard UI displays:
- Current mode (CKA configuration)
- Active K (draft speculation window)
- Generation Throughput (tok/s)
- Speculative Acceptance Rate (%)
- Peak VRAM Footprint (GB)
- Operating Power (W)
- Energy efficiency (J/token)
- Live streamed token output
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Generator

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from zassd.cache.kv_cache import TargetKVCache
from zassd.controllers.entropy import compute_entropy
from zassd.controllers.hardware_controller import HardwareAwareJointController, HardwareState
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.profiling.gpu import GPUProfiler
from zassd.profiling.memory import get_vram_usage, reset_vram_stats
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Canonical CKA configs
CONFIG_DEFINITIONS = {
    "cka_90": [4, 5, 6, 7],
    "cka_83": [4, 5, 6, 7, 12, 13],
    "cka_75": [3, 4, 5, 6, 7, 10, 11, 12, 13],
    "cka_60": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 21],
    "cka_50": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
}


def stream_self_speculative_generate(
    model,
    tokenizer,
    layer_mgr: LayerManager,
    controller: HardwareAwareJointController,
    prompt: str,
    max_new_tokens: int = 256,
    device: str = "cuda:0",
) -> Generator[tuple[str, dict], None, None]:
    """Stream tokens token-by-token with real-time systems telemetry."""
    gpu_profiler = GPUProfiler()

    # Collect comprehensive stop token IDs across architectures (Qwen, LLaMA)
    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    for stop_str in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        tid = tokenizer.convert_tokens_to_ids(stop_str)
        if tid is not None and isinstance(tid, int) and tid != tokenizer.unk_token_id:
            stop_token_ids.add(tid)

    # Format with Chat Template if available and not already formatted
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template and "<|im_start|>" not in prompt and "<|start_header_id|>" not in prompt:
        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable and precise AI assistant specializing in computer science, machine learning, and systems architecture.",
            },
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]

    torch.cuda.synchronize()
    gen_start_time = time.perf_counter()
    e_start = gpu_profiler.get_total_energy_mj()

    # Canonical Prefill
    target_kv = TargetKVCache()
    with torch.no_grad():
        prefill_out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
    target_prefix_logit = prefill_out.logits[0, -1, :]
    curr_target_tok = int(target_prefix_logit.argmax(dim=-1).item())
    curr_entropy = float(compute_entropy(target_prefix_logit.float()).item())

    emitted_tokens: list[int] = []
    total_draft_tokens = 0
    total_accepted_tokens = 0
    current_prefix_len = prompt_len

    last_accepted = 1
    last_proposed = 2
    last_draft_ms = 18.0
    last_verify_ms = 22.0

    while len(emitted_tokens) < max_new_tokens:
        rem_tokens = max_new_tokens - len(emitted_tokens)
        if rem_tokens <= 0:
            break

        # Controller selects action (S_t, K_t) dynamically
        hw_snap = gpu_profiler.snapshot()
        vram_info = get_vram_usage()
        hw_state = HardwareState(
            vram_used_mb=vram_info.get("allocated_mb", 2000.0),
            gpu_power_w=hw_snap.get("power_w", 55.0),
            gpu_temperature_c=hw_snap.get("temperature_c", 55.0),
        )
        controller.hardware_override = hw_state

        action = controller.select_action(
            entropy=curr_entropy,
            last_accepted=last_accepted,
            last_proposed=last_proposed,
            draft_ms=last_draft_ms,
            verify_ms=last_verify_ms,
        )

        step_k = min(action.draft_length, rem_tokens)
        active_skip = action.skip_indices
        cfg_name = action.config_name

        # Fork ephemeral draft KV
        draft_kv = target_kv.fork_ephemeral_draft_kv()

        # Draft Phase
        t_d0 = time.perf_counter()
        draft_tokens: list[int] = []
        with torch.no_grad():
            with layer_mgr.skip_layers(active_skip):
                curr_d = torch.tensor([[curr_target_tok]], device=device)
                for _ in range(step_k):
                    d_out = model(curr_d, past_key_values=draft_kv, use_cache=True)
                    d_next = int(d_out.logits[0, -1, :].argmax(dim=-1).item())
                    draft_tokens.append(d_next)
                    curr_d = torch.tensor([[d_next]], device=device)
                    if d_next in stop_token_ids:
                        break

        torch.cuda.synchronize()
        last_draft_ms = (time.perf_counter() - t_d0) * 1000.0
        actual_k = len(draft_tokens)
        total_draft_tokens += actual_k

        # Parallel Target Verify Phase
        t_v0 = time.perf_counter()
        verify_inputs = [curr_target_tok] + draft_tokens
        cand_tensor = torch.tensor([verify_inputs], device=device)
        with torch.no_grad():
            v_out = model(cand_tensor, past_key_values=target_kv.cache, use_cache=True)
        torch.cuda.synchronize()
        last_verify_ms = (time.perf_counter() - t_v0) * 1000.0

        # Verification & Acceptance
        cycle_emitted = [curr_target_tok]
        rejected_at = None
        num_accepted = 0
        stopped = (curr_target_tok in stop_token_ids)

        if not stopped:
            for i in range(actual_k):
                pred = int(v_out.logits[0, i, :].argmax(dim=-1).item())
                if pred == draft_tokens[i]:
                    cycle_emitted.append(draft_tokens[i])
                    num_accepted += 1
                    if draft_tokens[i] in stop_token_ids:
                        stopped = True
                        break
                else:
                    target_kv.crop(current_prefix_len + len(cycle_emitted))
                    curr_target_tok = pred
                    rejected_at = i
                    break

            if rejected_at is None and not stopped:
                bonus_tok = int(v_out.logits[0, actual_k, :].argmax(dim=-1).item())
                curr_target_tok = bonus_tok
                if bonus_tok in stop_token_ids:
                    stopped = True

        total_accepted_tokens += num_accepted
        last_accepted = num_accepted
        last_proposed = actual_k

        emitted_tokens.extend(cycle_emitted)
        current_prefix_len += len(cycle_emitted)

        # Compute dynamic telemetry
        elapsed_s = time.perf_counter() - gen_start_time
        curr_tps = len(emitted_tokens) / max(1e-3, elapsed_s)
        curr_acc = (total_accepted_tokens / max(1, total_draft_tokens)) * 100.0
        vram_mb = vram_info.get("peak_mb", 2000.0)
        power_w = hw_snap.get("power_w", 55.0)

        e_curr = gpu_profiler.get_total_energy_mj()
        if e_start is not None and e_curr is not None and e_curr >= e_start:
            energy_j = ((e_curr - e_start) / 1000.0) / max(1, len(emitted_tokens))
        else:
            energy_j = (power_w * elapsed_s) / max(1, len(emitted_tokens))

        clean_cycle = [t for t in cycle_emitted if t not in stop_token_ids]
        chunk_text = tokenizer.decode(clean_cycle, skip_special_tokens=True)

        telemetry = {
            "mode": cfg_name.upper().replace("_", "-"),
            "k": step_k,
            "tokens_per_second": round(curr_tps, 1),
            "acceptance_rate_pct": round(curr_acc, 1),
            "vram_gb": round(vram_mb / 1024.0, 2),
            "power_w": round(power_w, 1),
            "energy_j_token": round(energy_j, 2),
            "num_tokens": len(emitted_tokens),
        }

        yield chunk_text, telemetry

        if stopped or any(t in stop_token_ids for t in cycle_emitted) or curr_target_tok in stop_token_ids:
            break


def build_dashboard(
    prompt: str, generated_text: str, telemetry: dict[str, Any]
) -> Layout:
    """Construct Rich layout displaying live telemetry dashboard and streaming text."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="telemetry", size=7),
        Layout(name="content", ratio=1),
    )

    header_text = Text()
    header_text.append(" ZASSD: Zero-Additional-VRAM Self-Speculative Decoding Interactive Demo\n", style="bold cyan")
    header_text.append("Hardware: NVIDIA GeForce RTX 4050 Laptop (6GB, 80W) | Execution: 4-bit NF4 Quantization", style="dim white")
    layout["header"].update(Panel(header_text, style="cyan"))

    # Telemetry Table
    table = Table(title="Live Systems Telemetry (HardwareAwareJointController)", expand=True)
    table.add_column("Current Mode", justify="center")
    table.add_column("K (Draft)", justify="center", style="bold yellow")
    table.add_column("Tokens/s", justify="center", style="bold magenta")
    table.add_column("Acceptance", justify="center", style="bold blue")
    table.add_column("VRAM", justify="center", style="bold red")
    table.add_column("Power", justify="center", style="bold yellow")
    table.add_column("Energy / Token", justify="center", style="bold green")

    table.add_row(
        f"{telemetry.get('mode', 'CKA-83')}",
        f"{telemetry.get('k', 2)}",
        f"{telemetry.get('tokens_per_second', 35.4)} tok/s",
        f"{telemetry.get('acceptance_rate_pct', 86.0)}%",
        f"{telemetry.get('vram_gb', 1.99)} GB",
        f"{telemetry.get('power_w', 58.0)} W",
        f"{telemetry.get('energy_j_token', 1.77)} J/tok",
    )
    layout["telemetry"].update(Panel(table, style="white"))

    # Content
    content_text = Text()
    content_text.append(f"Prompt: {prompt}\n\n", style="bold yellow")
    content_text.append("Streaming Generation:\n", style="bold white")
    content_text.append(generated_text, style="bold green")
    layout["content"].update(Panel(content_text, title="Streaming Output", style="green"))

    return layout


def main() -> None:
    parser = argparse.ArgumentParser(description="ZASSD Interactive Streaming Generation Demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", type=str, default="Explain the concept of speculative decoding in large language models and why it can accelerate inference without changing output quality.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate (default: 256)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    console = Console()

    console.print("\nLoading model & initializing runtime controller...")
    model = load_model(args.model, quantize=True, bits=4)
    tokenizer = load_tokenizer(args.model)
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    cost_model = MeasuredActionCostModel.from_files()

    controller = HardwareAwareJointController(
        candidate_layer_configs=CONFIG_DEFINITIONS,
        cost_model=cost_model,
        max_vram_mb=5500.0,
        power_budget_w=80.0,
        temp_threshold_c=80.0,
    )

    console.print("System initialized. Starting live streaming demonstration...\n")

    full_generated_text = ""
    latest_telemetry = {
        "mode": "CKA-83",
        "k": 2,
        "tokens_per_second": 0.0,
        "acceptance_rate_pct": 0.0,
        "vram_gb": 1.99,
        "power_w": 55.0,
        "energy_j_token": 0.0,
    }

    with Live(build_dashboard(args.prompt, full_generated_text, latest_telemetry), console=console, refresh_per_second=8) as live:
        for chunk, telemetry in stream_self_speculative_generate(
            model=model,
            tokenizer=tokenizer,
            layer_mgr=layer_mgr,
            controller=controller,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        ):
            full_generated_text += chunk
            latest_telemetry = telemetry
            live.update(build_dashboard(args.prompt, full_generated_text, latest_telemetry))
            time.sleep(0.02)

    console.print("\nStreaming Generation Complete.")
    console.print(
        f"Final Telemetry: Mode={latest_telemetry['mode']} | "
        f"K={latest_telemetry['k']} | TPS={latest_telemetry['tokens_per_second']} tok/s | "
        f"Acceptance={latest_telemetry['acceptance_rate_pct']}% | "
        f"VRAM={latest_telemetry['vram_gb']} GB | Power={latest_telemetry['power_w']} W | "
        f"Energy={latest_telemetry['energy_j_token']} J/tok\n"
    )


if __name__ == "__main__":
    main()
