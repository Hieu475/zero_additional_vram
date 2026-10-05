"""Unified Command-Line Interface for ZASSD (Zero-Additional-VRAM Self-Speculative Decoding).

Provides subcommands:
  - zassd benchmark: Run standardized inference benchmarks across methods and models.
  - zassd audit: Run mathematical exactness and numerical verification audits.
  - zassd demo: Launch interactive streaming generation demo.
  - zassd app: Run thin application layer (chat/summarize/qa/code) or serve HTTP API.
  - zassd generate-tables: Generate verified LaTeX paper tables and macros from raw results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from zassd.baselines.knapspec import KnapSpecController
from zassd.baselines.prompt_lookup import prompt_lookup_generate
from zassd.baselines.specbound import SpecBoundController
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter

logger = logging.getLogger("zassd")

SKIP_STRATEGIES = {
    "mid_12": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "mid_10": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "mid_8": [14, 15, 16, 17, 18, 19, 20, 21],
    "empirical_6": [4, 5, 19, 20, 22, 23],
    "empirical_8": [4, 5, 19, 20, 22, 23, 24, 25],
    "empirical_10": [4, 5, 9, 13, 19, 20, 22, 23, 24, 25],
    "cka_75": [4, 8, 12, 16, 20, 24, 28, 32],
    "cka_50": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
}


def run_vanilla(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: str = "cuda:0",
) -> tuple[str, float, int]:
    """Execute standard autoregressive decoding with TargetKVCache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    cache = TargetKVCache(backend="static")
    emitted_ids: list[int] = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(inputs.input_ids, past_key_values=cache.cache, use_cache=True)
        curr = out.logits[:, -1:, :].argmax(dim=-1)
        curr_id = int(curr.item())
        emitted_ids.append(curr_id)

        for _ in range(max_new_tokens - 1):
            if curr_id == tokenizer.eos_token_id:
                break
            out = model(curr, past_key_values=cache.cache, use_cache=True)
            if temperature == 0.0:
                curr = out.logits[:, -1:, :].argmax(dim=-1)
            else:
                probs = torch.softmax(out.logits[0, -1, :].float() / temperature, dim=-1)
                curr = torch.multinomial(probs, num_samples=1).view(1, 1)
            curr_id = int(curr.item())
            emitted_ids.append(curr_id)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_time = t1 - t0
    tps = len(emitted_ids) / total_time if total_time > 0 else 0.0
    text = tokenizer.decode(emitted_ids, skip_special_tokens=True)
    return text, tps, len(emitted_ids)


def handle_benchmark(args: argparse.Namespace) -> None:
    """Execute benchmark subcommand."""
    device = args.device
    print(f"\n[ZASSD Benchmark] Loading model: {args.model} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    is_prequantized = "bnb-4bit" in args.model.lower() or "4bit" in args.model.lower()

    if is_prequantized:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device)
    elif args.quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map=device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device,
            torch_dtype=torch.float16,
        )
    model.eval()

    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)

    # Determine skip indices
    if args.skip_strategy in SKIP_STRATEGIES:
        skip_indices = SKIP_STRATEGIES[args.skip_strategy]
    elif args.skip_indices:
        skip_indices = [int(x.strip()) for x in args.skip_indices.split(",")]
    else:
        skip_indices = SKIP_STRATEGIES["mid_12"]

    # Determine prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            if args.prompts_file.endswith(".jsonl"):
                prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
            else:
                prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "Explain the concept of speculative decoding in large language models and why it accelerates inference.",
            "Write a quicksort implementation in Python with clean comments and edge case handling.",
            "Summarize the key architectural trade-offs between Multi-Head Attention and Grouped-Query Attention.",
        ]

    print(f"[ZASSD Benchmark] Running method: '{args.method}' across {len(prompts)} prompt(s) (k={args.k}) ...")

    # Warmup (not timed): eliminate cold-start CUDA allocation bias for both paths
    try:
        run_vanilla(model, tokenizer, "Warmup: explain gravity briefly.",
                    max_new_tokens=16, temperature=args.temperature, device=device)
    except Exception:
        pass

    # Measure Vanilla baseline
    vanilla_tps_list = []
    for p in prompts:
        _, tps, _ = run_vanilla(model, tokenizer, p, max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
        vanilla_tps_list.append(tps)
    baseline_tps = float(np.mean(vanilla_tps_list))

    results = []
    for p_idx, prompt in enumerate(prompts):
        if args.method == "vanilla":
            text, tps, num_toks = run_vanilla(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
            res = {"prompt_idx": p_idx, "tps": tps, "tokens": num_toks, "acc_rate": 0.0, "speedup": 1.0}
        elif args.method == "prompt_lookup":
            text, m = prompt_lookup_generate(model, tokenizer, prompt, k=args.k, max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
            res = {"prompt_idx": p_idx, "tps": m.tokens_per_second, "tokens": m.total_tokens, "acc_rate": m.acceptance_rate, "speedup": m.tokens_per_second / baseline_tps}
        elif args.method == "knapspec":
            ctrl = KnapSpecController(total_layers=adapter.num_layers, budget_ratio=0.75, fixed_k=args.k)
            text, m = self_speculative_generate(model, tokenizer, layer_mgr, ctrl.skip_indices, prompt, k=args.k, controller=ctrl, max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
            res = {"prompt_idx": p_idx, "tps": m.tokens_per_second, "tokens": m.total_tokens, "acc_rate": m.acceptance_rate, "speedup": m.tokens_per_second / baseline_tps}
        elif args.method == "hybrid":
            text, m = self_speculative_generate(model, tokenizer, layer_mgr, skip_indices, prompt, k=args.k, draft_mode="hybrid", max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
            res = {"prompt_idx": p_idx, "tps": m.tokens_per_second, "tokens": m.total_tokens, "acc_rate": m.acceptance_rate, "speedup": m.tokens_per_second / baseline_tps, "pld_cycles": m.pld_cycles, "layer_skip_cycles": m.layer_skip_cycles}
        elif args.method == "routed":
            from zassd.routing.hybrid_router import HybridDraftRouter
            from zassd.profiling.action_cost_model import MeasuredActionCostModel
            _cm = MeasuredActionCostModel.for_model(args.model)
            _router = HybridDraftRouter(cost_model=_cm, pld_k=4)
            _cfg = args.skip_strategy if args.skip_strategy in _cm.known_kept_layers else "cka_75"
            text, m = self_speculative_generate(model, tokenizer, layer_mgr, skip_indices, prompt, k=args.k, draft_mode="routed", router=_router, config_name=_cfg, max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
            res = {"prompt_idx": p_idx, "tps": m.tokens_per_second, "tokens": m.total_tokens, "acc_rate": m.acceptance_rate, "speedup": m.tokens_per_second / baseline_tps, "pld_cycles": m.pld_cycles, "layer_skip_cycles": m.layer_skip_cycles, "router_stats": _router.stats.to_dict()}
        elif args.method == "specbound":
            ctrl = SpecBoundController(skip_indices=skip_indices, k_min=1, k_max=max(2, args.k), initial_k=args.k)
            text, m = self_speculative_generate(model, tokenizer, layer_mgr, skip_indices, prompt, k=args.k, controller=ctrl, max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
            res = {"prompt_idx": p_idx, "tps": m.tokens_per_second, "tokens": m.total_tokens, "acc_rate": m.acceptance_rate, "speedup": m.tokens_per_second / baseline_tps}
        else:  # "zassd"
            text, m = self_speculative_generate(model, tokenizer, layer_mgr, skip_indices, prompt, k=args.k, max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
            res = {"prompt_idx": p_idx, "tps": m.tokens_per_second, "tokens": m.total_tokens, "acc_rate": m.acceptance_rate, "speedup": m.tokens_per_second / baseline_tps}

        results.append(res)
        print(f"  Prompt #{p_idx+1}: {res['tps']:.2f} tok/s | Acc: {res['acc_rate']*100:.1f}% | Rel: {res['speedup']:.3f}x")

    mean_tps = float(np.mean([r["tps"] for r in results]))
    mean_acc = float(np.mean([r["acc_rate"] for r in results]))
    mean_speedup = mean_tps / baseline_tps

    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY: {args.method.upper()} on {args.model}")
    print(f"  Vanilla Target Throughput: {baseline_tps:.2f} tok/s")
    print(f"  Method Mean Throughput   : {mean_tps:.2f} tok/s")
    print(f"  Relative Speedup         : {mean_speedup:.3f}x")
    print(f"  Mean Acceptance Rate     : {mean_acc*100:.1f}%")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "model": args.model,
                "method": args.method,
                "k": args.k,
                "skip_strategy": args.skip_strategy,
                "vanilla_tps": baseline_tps,
                "method_tps": mean_tps,
                "speedup": mean_speedup,
                "mean_acc": mean_acc,
                "results": results,
            }, f, indent=2)
        print(f"[ZASSD] Saved benchmark results to {args.output}")


def handle_app(args: argparse.Namespace) -> None:
    """Thin application layer: tasks + optional stdlib HTTP server."""
    from zassd.app.assistant import IntelligentAssistant

    if args.list_tasks:
        from zassd.app.prompts import TASK_DESCRIPTIONS

        print("Available tasks:")
        for name, desc in TASK_DESCRIPTIONS.items():
            print(f"  {name:<10} {desc}")
        return

    if args.serve:
        from zassd.app.server import serve

        serve(port=args.port, model_name=args.model, method=args.method)
        return

    assistant = IntelligentAssistant(
        model_name=args.model,
        method=args.method,  # type: ignore[arg-type]
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        k=args.k,
    )
    if getattr(args, "interactive", False):
        assistant.run_interactive()
        return
    if args.docs_file:
        import json as _json

        with open(args.docs_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    assistant.store.add(str(obj.get("id", f"doc{i}")), obj.get("text", line))
                except Exception:
                    assistant.store.add(f"doc{i}", line)

    if args.compare:
        results = assistant.compare(args.task, args.input or "", context=args.context or "")  # type: ignore[arg-type]
        print(f"\n[ZASSD App] compare task='{args.task}' input_len={len(args.input or '')}")
        for r in results:
            print(f"  [{r.method_used}] {r.tps:.1f} tok/s | {r.latency_s:.2f}s | {r.text[:160]}")
        return

    if args.task == "qa":
        res = assistant.ask(args.input or "")
    elif args.task == "summarize":
        res = assistant.summarize(args.input or "")
    elif args.task == "code":
        res = assistant.code_assist(args.input or "", context=args.context or "")
    else:
        res = assistant.generate("chat", args.input or "", context=args.context or "")
    print(f"\n[ZASSD App] task={res.task} method={res.method_used} "
          f"({res.tps:.1f} tok/s, {res.latency_s:.2f}s)")
    print("-" * 60)
    print(res.text)


def handle_tables(args: argparse.Namespace) -> None:
    """Execute generate-tables subcommand."""
    import subprocess
    cmd = [sys.executable, "scripts/generate_paper_tables.py"]
    subprocess.run(cmd, check=True)


def handle_demo(args: argparse.Namespace) -> None:
    """Launch interactive streaming demo."""
    import subprocess
    cmd = [sys.executable, "scripts/interactive_streaming_demo.py"]
    subprocess.run(cmd)


def handle_audit(args: argparse.Namespace) -> None:
    """Launch numerical equivalence audit."""
    import subprocess
    cmd = [sys.executable, "scripts/audit_numerical_equivalence.py"]
    subprocess.run(cmd)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="zassd",
        description="ZASSD: Zero-Additional-VRAM Self-Speculative Decoding CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Subcommand: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run speculative decoding benchmarks")
    bench_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID")
    bench_parser.add_argument(
        "--method",
        type=str,
        default="zassd",
        choices=["vanilla", "zassd", "prompt_lookup", "hybrid", "routed", "knapspec", "specbound"],
        help="Decoding method to benchmark",
    )
    bench_parser.add_argument("--k", type=int, default=2, help="Speculative draft length K")
    bench_parser.add_argument(
        "--skip-strategy",
        type=str,
        default="mid_12",
        choices=["mid_12", "mid_10", "mid_8", "empirical_6", "empirical_8", "empirical_10", "cka_75", "cka_50"],
        help="Predefined layer skipping strategy",
    )
    bench_parser.add_argument("--skip-indices", type=str, default=None, help="Comma-separated custom layer indices to skip")
    bench_parser.add_argument("--prompt", type=str, default=None, help="Single prompt string to evaluate")
    bench_parser.add_argument("--prompts-file", type=str, default=None, help="Path to prompts file (.txt or .jsonl)")
    bench_parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum new tokens per prompt")
    bench_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    bench_parser.add_argument("--no-quantize", dest="quantize", action="store_false", help="Disable 4-bit NF4 quantization")
    bench_parser.add_argument("--device", type=str, default="cuda:0", help="Execution device")
    bench_parser.add_argument("--output", type=str, default=None, help="JSON file path to save benchmark results")
    bench_parser.set_defaults(quantize=True)

    # Subcommand: audit
    subparsers.add_parser("audit", help="Run numerical exactness and GEMM noise envelope audit")

    # Subcommand: demo
    subparsers.add_parser("demo", help="Launch interactive streaming demo")

    # Subcommand: generate-tables
    subparsers.add_parser("generate-tables", help="Generate verified LaTeX tables and macros for manuscript")

    # Subcommand: app (thin application layer for thesis demo)
    app_parser = subparsers.add_parser("app", help="Run end-user tasks (chat/summarize/qa/code) or serve HTTP API")
    app_parser.add_argument("--task", type=str, default="chat", choices=["chat", "summarize", "qa", "code"])
    app_parser.add_argument("--input", type=str, default="Hello, what can you do on a 6GB laptop GPU?")
    app_parser.add_argument("--context", type=str, default="")
    app_parser.add_argument("--docs-file", type=str, default=None, help="JSONL docs file {id, text} for QA")
    app_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    app_parser.add_argument("--method", type=str, default="auto",
                            choices=["auto", "vanilla", "prompt_lookup", "zassd", "routed", "hybrid"])
    app_parser.add_argument("--k", type=int, default=4)
    app_parser.add_argument("--max-new-tokens", type=int, default=128)
    app_parser.add_argument("--temperature", type=float, default=0.0)
    app_parser.add_argument("--device", type=str, default="cuda:0")
    app_parser.add_argument("--compare", action="store_true", help="Run same prompt via vanilla/prompt_lookup/routed")
    app_parser.add_argument("--interactive", action="store_true", help="Multi-turn REPL keeping history (/reset, /quit)")
    app_parser.add_argument("--serve", action="store_true", help="Serve stdlib HTTP JSON API")
    app_parser.add_argument("--port", type=int, default=8000)
    app_parser.add_argument("--list-tasks", action="store_true")

    args = parser.parse_args()

    if args.subcommand == "benchmark":
        handle_benchmark(args)
    elif args.subcommand == "audit":
        handle_audit(args)
    elif args.subcommand == "demo":
        handle_demo(args)
    elif args.subcommand == "generate-tables":
        handle_tables(args)
    elif args.subcommand == "app":
        handle_app(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
