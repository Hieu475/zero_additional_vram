"""Script to run ActionProfiler and generate empirical action_costs.json database."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_profiler import ActionProfiler
from zassd.profiling.gpu import GPUProfiler
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile candidate actions (S, K) to build action_costs.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name or path")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 6, 8], help="Draft lengths to profile")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of evaluation prompts per action")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Tokens to generate per prompt")
    parser.add_argument("--output-file", type=str, default="results/raw/action_costs.json", help="Output database file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    logger.info("Initializing Action Profiler...")

    # Load model and tokenizer
    model = load_model(args.model, quantize=True, bits=4)
    tokenizer = load_tokenizer(args.model)
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    gpu_profiler = GPUProfiler()

    # Load candidate layer skip configurations
    cka_file = Path("experiments/03_cka/benchmark_results.json")
    if cka_file.exists():
        with open(cka_file) as f:
            cka_data = json.load(f)
        cka_75_skips = cka_data.get("cka_75", {}).get("skipped_indices", [3, 5, 7, 9, 11, 13, 16, 18, 21])
        cka_50_skips = cka_data.get("cka_50", {}).get(
            "skipped_indices", [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 16, 18, 21, 23, 25, 28, 31, 33]
        )
    else:
        cka_75_skips = [3, 5, 7, 9, 11, 13, 16, 18, 21]
        cka_50_skips = [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 16, 18, 21, 23, 25, 28, 31, 33]

    candidate_configs = {
        "cka_75": cka_75_skips,
        "cka_50": cka_50_skips,
    }

    # Load fixed prompts
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))
    eval_prompts = prompts[: args.num_prompts]

    profiler = ActionProfiler(
        model=model,
        tokenizer=tokenizer,
        layer_mgr=layer_mgr,
        candidate_configs=candidate_configs,
        gpu_profiler=gpu_profiler,
        device="cuda:0",
    )

    db = profiler.profile_all(
        k_values=args.k_values,
        eval_prompts=eval_prompts,
        output_file=args.output_file,
    )

    logger.info(f"Profiling complete. Action database created at {args.output_file}")


if __name__ == "__main__":
    main()
