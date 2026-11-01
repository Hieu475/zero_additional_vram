"""Paired same-process comparison of B1 block schedules on development prompts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate

STRATEGIES = {
    "warp1": "triton_nf4_group_scale_packed_rows2_n16_warp1_k256",
    "split8": "triton_nf4_split8_downproj_warp1_k256",
}


def generate(model, tokenizer, ids, strategy, max_new_tokens, capacity):
    _, metrics = b1_fixed_row_block_generate(
        model, tokenizer, ids, k=1, ngram_size=3,
        max_new_tokens=max_new_tokens, device="cuda:0",
        kv_cache_max_capacity=capacity, physical_rows=2,
        attention_strategy="padded", quantized_linear_strategy=strategy,
    )
    torch.cuda.synchronize("cuda:0")
    return {
        "token_ids": metrics.generated_token_ids,
        "total_s": metrics.total_time_s,
        "prefill_s": metrics.prefill_time_s,
        "target_forwards": metrics.target_forwards,
        "target_rows": metrics.target_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", default="data/benchmarks/cnndm_eval.jsonl")
    parser.add_argument("--prompt-count", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompts = [json.loads(line) for line in Path(args.prompt_file).read_text().splitlines() if line.strip()]
    prompts = prompts[:args.prompt_count]
    if len(prompts) != args.prompt_count:
        raise ValueError("not enough development prompts")
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
        model_dtype=torch.float16,
    )
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    records = []
    with torch.inference_mode():
        for prompt_idx, prompt in enumerate(prompts):
            ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
            if ids.shape[1] + args.max_new_tokens > args.capacity:
                raise ValueError(f"{prompt['id']} exceeds capacity")
            for strategy in STRATEGIES.values():
                warm = generate(model, tokenizer, ids, strategy, 2, args.capacity)
                if len(warm["token_ids"]) != 2:
                    raise RuntimeError("two-token warmup failed")
            repeats = []
            for repeat in range(args.repeats):
                order = list(STRATEGIES)
                if (prompt_idx + repeat) % 2:
                    order.reverse()
                values = {
                    name: generate(model, tokenizer, ids, STRATEGIES[name],
                                   args.max_new_tokens, args.capacity)
                    for name in order
                }
                repeats.append({
                    "order": order,
                    "values": values,
                    "tokens_equal": values["warp1"]["token_ids"] == values["split8"]["token_ids"],
                    "split8_over_warp1": values["split8"]["total_s"] / values["warp1"]["total_s"],
                })
            ratios = [item["split8_over_warp1"] for item in repeats]
            records.append({
                "prompt_id": str(prompt["id"]),
                "prompt_tokens": int(ids.shape[1]),
                "repeats": repeats,
                "geomean_split8_over_warp1": math.exp(statistics.mean(map(math.log, ratios))),
                "all_tokens_equal": all(item["tokens_equal"] for item in repeats),
            })
    payload = {
        "status": "same-process paired development diagnostic; finite evidence only",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "backend": loaded.metadata,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "prompt_file": args.prompt_file, "prompt_count": args.prompt_count,
        "repeats_per_prompt": args.repeats,
        "max_new_tokens": args.max_new_tokens, "k": 1, "physical_rows": 2,
        "attention": "padded", "heldout_gsm8k_used": False,
        "results": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "all_tokens_equal": all(row["all_tokens_equal"] for row in records),
        "split8_over_warp1_by_prompt": {
            row["prompt_id"]: row["geomean_split8_over_warp1"] for row in records
        },
    }, indent=2))


if __name__ == "__main__":
    main()
