"""Phase 2 — Layer Skipping Baseline Benchmark.

Measures speed AND output quality when skipping layers.

For each layer configuration:
  - Speed:   tok/s, TTFT, TPOT, peak VRAM
  - Quality: Greedy token agreement, KL divergence, cosine similarity

Qwen2.5-3B has 36 transformer blocks.
Configurations tested:
  75% kept (27 layers)  — skip  9 layers
  50% kept (18 layers)  — skip 18 layers
  25% kept ( 9 layers)  — skip 27 layers

Strategies: static-even, random (3 seeds), contiguous-middle

Usage:
    python scripts/benchmark_layer_skip.py
    python scripts/benchmark_layer_skip.py --runs 10  # quick test
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.models.layer_manager import LayerManager
from zassd.layer_selection.random import select_random_layers
from zassd.layer_selection.static import (
    select_even_layers,
    select_middle_skip,
)
from zassd.profiling.memory import get_vram_usage, reset_vram_stats
from zassd.utils.config import load_config
from zassd.utils.seed import set_seed
from zassd.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer configurations
# ---------------------------------------------------------------------------

def build_skip_configs(num_layers: int) -> list[dict]:
    """Build all layer-skip configurations to test."""
    configs: list[dict] = []

    # --- 75 % kept  (skip ~25 %) -----------------------------------
    # static even: skip every 4th layer
    skip_25 = [i for i in range(num_layers) if i % 4 == 1]  # skip indices 1,5,9,...
    configs.append({
        "name": "static_75",
        "kept": sorted(set(range(num_layers)) - set(skip_25)),
        "skipped": skip_25,
        "description": "75% kept, skip every 4th layer",
    })

    for seed in [42, 123, 2026]:
        kept, skipped = select_random_layers(
            num_layers, skip_ratio=0.25, seed=seed
        )
        configs.append({
            "name": f"random_75_s{seed}",
            "kept": kept,
            "skipped": skipped,
            "description": f"75% kept, random seed={seed}",
        })

    # --- 50 % kept  (skip ~50 %) -----------------------------------
    # static even: skip odd-indexed layers
    skip_50_even = [i for i in range(1, num_layers, 2)]
    keep_50_even = [i for i in range(0, num_layers, 2)]
    configs.append({
        "name": "static_50",
        "kept": keep_50_even,
        "skipped": skip_50_even,
        "description": "50% kept, skip odd-indexed layers",
    })

    for seed in [42, 123, 2026]:
        kept, skipped = select_random_layers(
            num_layers, skip_ratio=0.50, seed=seed
        )
        configs.append({
            "name": f"random_50_s{seed}",
            "kept": kept,
            "skipped": skipped,
            "description": f"50% kept, random seed={seed}",
        })

    # contiguous middle
    mid_start = num_layers // 4           # layer 9
    mid_end = mid_start + num_layers // 2  # layer 27
    kept_mid, skipped_mid = select_middle_skip(
        num_layers, mid_start, mid_end
    )
    configs.append({
        "name": "contiguous_50_mid",
        "kept": kept_mid,
        "skipped": skipped_mid,
        "description": f"50% kept, skip contiguous middle [{mid_start}..{mid_end})",
    })

    # --- 25 % kept  (skip ~75 %) -----------------------------------
    # static: keep every 4th layer
    keep_25 = [i for i in range(num_layers) if i % 4 == 0]
    skip_75 = sorted(set(range(num_layers)) - set(keep_25))
    configs.append({
        "name": "static_25",
        "kept": keep_25,
        "skipped": skip_75,
        "description": "25% kept, keep every 4th layer",
    })

    for seed in [42, 123, 2026]:
        kept, skipped = select_random_layers(
            num_layers, skip_ratio=0.75, seed=seed
        )
        configs.append({
            "name": f"random_25_s{seed}",
            "kept": kept,
            "skipped": skipped,
            "description": f"25% kept, random seed={seed}",
        })

    return configs


# ---------------------------------------------------------------------------
# Quality measurement
# ---------------------------------------------------------------------------

def measure_quality(
    model,
    tokenizer,
    adapter: ModelAdapter,
    layer_mgr: LayerManager,
    skip_indices: list[int],
    prompts: list[dict],
    reference_outputs: list[dict],
    max_new_tokens: int = 128,
    device: str = "cuda:0",
) -> dict:
    """Measure output quality vs full model reference.

    Returns dict with:
      - greedy_match_rate: fraction of tokens matching reference
      - avg_kl_divergence: KL(p_full || p_skip) on first output token
      - avg_cosine_similarity: cosine of logits on first output token
      - avg_top1_agreement: top-1 token agreement on first output token
    """
    kl_divs = []
    cosine_sims = []
    top1_agreements = []
    token_match_rates = []

    num_eval = min(len(prompts), len(reference_outputs), 10)

    for i in range(num_eval):
        prompt_text = prompts[i]["prompt"]
        ref_output = reference_outputs[i].get("output", "")

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # --- First-token logit comparison ---
        with torch.no_grad():
            # Full model logits
            full_out = model(input_ids=inputs["input_ids"])
            full_logits = full_out.logits[:, -1, :].float()

            # Skipped model logits
            with layer_mgr.skip_layers(skip_indices):
                skip_out = model(input_ids=inputs["input_ids"])
            skip_logits = skip_out.logits[:, -1, :].float()

        # KL divergence: KL(p_full || p_skip)
        full_log_probs = F.log_softmax(full_logits, dim=-1)
        skip_log_probs = F.log_softmax(skip_logits, dim=-1)
        full_probs = F.softmax(full_logits, dim=-1)
        kl = F.kl_div(skip_log_probs, full_probs, reduction="batchmean").item()
        kl_divs.append(kl)

        # Cosine similarity
        cos = F.cosine_similarity(full_logits, skip_logits, dim=-1).item()
        cosine_sims.append(cos)

        # Top-1 agreement
        full_top1 = full_logits.argmax(dim=-1)
        skip_top1 = skip_logits.argmax(dim=-1)
        top1_agreements.append((full_top1 == skip_top1).float().item())

        # --- Free-running token match ---
        with torch.no_grad():
            with layer_mgr.skip_layers(skip_indices):
                generated_ids = inputs["input_ids"].clone()
                past_key_values = None
                for step in range(min(max_new_tokens, 64)):
                    if past_key_values is None:
                        out = model(input_ids=generated_ids, use_cache=True)
                    else:
                        out = model(
                            input_ids=generated_ids[:, -1:],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    past_key_values = out.past_key_values
                    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_tok], dim=-1)
                    if next_tok.item() == tokenizer.eos_token_id:
                        break

        skip_text = tokenizer.decode(
            generated_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Token-level match rate
        ref_tokens = tokenizer.encode(ref_output, add_special_tokens=False)
        skip_tokens = tokenizer.encode(skip_text, add_special_tokens=False)
        min_len = min(len(ref_tokens), len(skip_tokens))
        if min_len > 0:
            matches = sum(
                1 for a, b in zip(ref_tokens[:min_len], skip_tokens[:min_len])
                if a == b
            )
            token_match_rates.append(matches / min_len)
        else:
            token_match_rates.append(0.0)

    return {
        "avg_kl_divergence": float(np.mean(kl_divs)),
        "avg_cosine_similarity": float(np.mean(cosine_sims)),
        "avg_top1_agreement": float(np.mean(top1_agreements)),
        "avg_token_match_rate": float(np.mean(token_match_rates)),
        "num_eval_prompts": num_eval,
    }


# ---------------------------------------------------------------------------
# Speed measurement
# ---------------------------------------------------------------------------

def measure_speed(
    model,
    tokenizer,
    layer_mgr: LayerManager,
    skip_indices: list[int],
    prompts: list[dict],
    max_new_tokens: int = 128,
    warmup_runs: int = 5,
    measured_runs: int = 15,
    device: str = "cuda:0",
) -> dict:
    """Measure generation speed with layer skipping."""

    # Warmup
    for i in range(warmup_runs):
        prompt_text = prompts[i % len(prompts)]["prompt"]
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            with layer_mgr.skip_layers(skip_indices):
                generated = inputs["input_ids"].clone()
                pkv = None
                for _ in range(32):
                    if pkv is None:
                        out = model(input_ids=generated, use_cache=True)
                    else:
                        out = model(
                            input_ids=generated[:, -1:],
                            past_key_values=pkv, use_cache=True,
                        )
                    pkv = out.past_key_values
                    nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, nt], dim=-1)

    # Measured runs
    results = []
    for run_idx in range(measured_runs):
        prompt_text = prompts[run_idx % len(prompts)]["prompt"]
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        reset_vram_stats(device)
        torch.cuda.synchronize()

        with torch.no_grad():
            with layer_mgr.skip_layers(skip_indices):
                generated = inputs["input_ids"].clone()
                pkv = None
                per_token_ms = []
                start = time.perf_counter()

                for step in range(max_new_tokens):
                    t0 = time.perf_counter()
                    if pkv is None:
                        out = model(input_ids=generated, use_cache=True)
                    else:
                        out = model(
                            input_ids=generated[:, -1:],
                            past_key_values=pkv, use_cache=True,
                        )
                    pkv = out.past_key_values
                    nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, nt], dim=-1)
                    per_token_ms.append((time.perf_counter() - t0) * 1000)

                    if step == 0:
                        ttft = time.perf_counter() - start

                    if nt.item() == tokenizer.eos_token_id:
                        break

                torch.cuda.synchronize()
                total_time = time.perf_counter() - start

        new_tokens = generated.shape[1] - input_len
        results.append({
            "run_idx": run_idx,
            "total_tokens": new_tokens,
            "total_time_s": total_time,
            "tokens_per_second": new_tokens / total_time if total_time > 0 else 0,
            "ttft_s": ttft,
            "tpot_ms": float(np.mean(per_token_ms[1:])) if len(per_token_ms) > 1 else 0,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        })

    # Aggregate
    tps = np.array([r["tokens_per_second"] for r in results])
    ttft = np.array([r["ttft_s"] for r in results]) * 1000
    tpot = np.array([r["tpot_ms"] for r in results])
    vram = np.array([r["peak_vram_mb"] for r in results])

    return {
        "per_run": results,
        "tokens_per_second": {"mean": float(tps.mean()), "std": float(tps.std())},
        "ttft_ms": {"mean": float(ttft.mean()), "std": float(ttft.std())},
        "tpot_ms": {"mean": float(tpot.mean()), "std": float(tpot.std())},
        "peak_vram_mb": {"mean": float(vram.mean()), "max": float(vram.max())},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Layer skipping benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="experiments/02_layer_skip")
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=str(output_dir / "benchmark.log"))
    set_seed(42)

    logger.info("=" * 60)
    logger.info("PHASE 2 — LAYER SKIPPING BASELINE")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, quantize=True, bits=args.bits)
    tokenizer = load_tokenizer(args.model)

    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    num_layers = adapter.num_layers
    logger.info(f"Model has {num_layers} transformer layers")

    # Load prompts
    prompts = []
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    if prompts_path.exists():
        with open(prompts_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
    logger.info(f"Loaded {len(prompts)} prompts")

    # Load reference outputs from Phase 1 (ctx_128)
    ref_path = Path("experiments/01_vanilla/ctx_128/outputs.jsonl")
    reference_outputs = []
    if ref_path.exists():
        with open(ref_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    reference_outputs.append(json.loads(line))
        logger.info(f"Loaded {len(reference_outputs)} reference outputs")
    else:
        logger.warning("No reference outputs found; generating fresh reference")
        # Generate reference outputs with full model
        for i in range(min(10, len(prompts))):
            inp = tokenizer(prompts[i]["prompt"], return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                gen = model.generate(
                    **inp, max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            out_text = tokenizer.decode(
                gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True
            )
            reference_outputs.append({
                "prompt_id": prompts[i]["id"],
                "output": out_text,
            })

    # --- Full model speed reference (to compute speedup) ---
    logger.info("\n--- Full model speed reference ---")
    full_speed = measure_speed(
        model, tokenizer, layer_mgr,
        skip_indices=[],  # no skipping
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    logger.info(
        f"Full model: {full_speed['tokens_per_second']['mean']:.1f} "
        f"± {full_speed['tokens_per_second']['std']:.1f} tok/s"
    )

    # Build configurations
    configs = build_skip_configs(num_layers)

    # Results table
    all_results: dict[str, dict] = {}
    all_results["full_36"] = {
        "layers_kept": num_layers,
        "layers_skipped": 0,
        "skip_ratio": 0.0,
        "speed": full_speed,
        "quality": {
            "avg_kl_divergence": 0.0,
            "avg_cosine_similarity": 1.0,
            "avg_top1_agreement": 1.0,
            "avg_token_match_rate": 1.0,
        },
    }

    for cfg in configs:
        name = cfg["name"]
        kept = cfg["kept"]
        skipped = cfg["skipped"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {name} — {cfg['description']}")
        logger.info(f"  Kept layers ({len(kept)}): {kept}")
        logger.info(f"  Skipped ({len(skipped)}): {skipped}")
        logger.info(f"{'='*60}")

        # Quality
        logger.info("Measuring quality...")
        quality = measure_quality(
            model, tokenizer, adapter, layer_mgr,
            skip_indices=skipped,
            prompts=prompts,
            reference_outputs=reference_outputs,
            max_new_tokens=args.max_new_tokens,
        )
        logger.info(
            f"  KL={quality['avg_kl_divergence']:.4f}, "
            f"Cosine={quality['avg_cosine_similarity']:.4f}, "
            f"Top1={quality['avg_top1_agreement']:.2%}, "
            f"TokenMatch={quality['avg_token_match_rate']:.2%}"
        )

        # Speed
        logger.info("Measuring speed...")
        speed = measure_speed(
            model, tokenizer, layer_mgr,
            skip_indices=skipped,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            warmup_runs=args.warmup,
            measured_runs=args.runs,
        )
        speedup = (
            speed["tokens_per_second"]["mean"]
            / full_speed["tokens_per_second"]["mean"]
        )
        logger.info(
            f"  {speed['tokens_per_second']['mean']:.1f} tok/s "
            f"(speedup: {speedup:.2f}×), "
            f"VRAM={speed['peak_vram_mb']['mean']:.0f}MB"
        )

        all_results[name] = {
            "layers_kept": len(kept),
            "layers_skipped": len(skipped),
            "skip_ratio": len(skipped) / num_layers,
            "kept_indices": kept,
            "skipped_indices": skipped,
            "description": cfg["description"],
            "speed": speed,
            "quality": quality,
            "speedup": speedup,
        }

        # Save per-config
        cfg_dir = output_dir / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "results.json", "w") as f:
            json.dump(all_results[name], f, indent=2)

        torch.cuda.empty_cache()
        gc.collect()

    # Save combined results (without per_run details for readability)
    combined = {}
    for name, res in all_results.items():
        combined[name] = {
            "layers_kept": res["layers_kept"],
            "layers_skipped": res["layers_skipped"],
            "skip_ratio": res.get("skip_ratio", 0),
            "tok_per_s_mean": res["speed"]["tokens_per_second"]["mean"],
            "tok_per_s_std": res["speed"]["tokens_per_second"]["std"],
            "ttft_ms": res["speed"]["ttft_ms"]["mean"],
            "tpot_ms": res["speed"]["tpot_ms"]["mean"],
            "peak_vram_mb": res["speed"]["peak_vram_mb"]["mean"],
            "speedup": res.get("speedup", 1.0),
            "kl_divergence": res["quality"]["avg_kl_divergence"],
            "cosine_similarity": res["quality"]["avg_cosine_similarity"],
            "top1_agreement": res["quality"]["avg_top1_agreement"],
            "token_match_rate": res["quality"]["avg_token_match_rate"],
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Print final table
    logger.info("\n" + "=" * 110)
    logger.info("LAYER SKIPPING RESULTS")
    logger.info("=" * 110)
    logger.info(
        f"{'Config':<22} | {'Layers':>6} | {'tok/s':>10} | {'Speedup':>7} | "
        f"{'VRAM(MB)':>8} | {'KL':>8} | {'Cosine':>6} | {'Top1':>6} | {'Match':>6}"
    )
    logger.info("-" * 110)

    for name in ["full_36"] + [c["name"] for c in configs]:
        r = combined[name]
        logger.info(
            f"{name:<22} | {r['layers_kept']:>6} | "
            f"{r['tok_per_s_mean']:>7.1f}±{r['tok_per_s_std']:.1f} | "
            f"{r['speedup']:>6.2f}× | "
            f"{r['peak_vram_mb']:>8.0f} | "
            f"{r['kl_divergence']:>8.4f} | "
            f"{r['cosine_similarity']:>6.4f} | "
            f"{r['top1_agreement']:>5.1%} | "
            f"{r['token_match_rate']:>5.1%}"
        )

    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
