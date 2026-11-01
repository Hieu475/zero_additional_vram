#!/usr/bin/env python3
"""Summarize key operators from K2/R3 diagnostic PyTorch profiler reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_event(events: list[dict], predicate) -> dict | None:
    return next((event for event in events if predicate(event)), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path("experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/operator_profile_summary.json"),
    )
    args = parser.parse_args()
    reports = [
        args.profile_root / "profile_humaneval112_k2r3_32/profile.json",
        args.profile_root / "profile_humaneval114_k2r3_32/profile.json",
    ]
    results = []
    for path in reports:
        report = json.loads(path.read_text())
        events = report["top_self_cuda"]
        target_forwards = report["target_forwards"]
        packed = find_event(events, lambda e: e["key"] == "_packed_two_rows_kernel")
        attention = find_event(events, lambda e: e["key"] == "_fixed_row_attention_tiled")
        output_head = find_event(
            events,
            lambda e: e["key"] == "aten::mm"
            and "[3, 2048]" in e["input_shapes"]
            and "151936" in e["input_shapes"],
        )
        prefill_vocab = find_event(
            events,
            lambda e: e["key"] == "aten::mm"
            and "[2048, 151936]" in e["input_shapes"]
            and f"[{report['prompt_tokens']}, 2048]" in e["input_shapes"],
        )
        bnb_prefill = [
            e for e in events
            if e["key"] == "bitsandbytes::gemm_4bit"
            and f"[1, {report['prompt_tokens']}, " in e["input_shapes"]
        ]

        def ms(event: dict | None) -> float | None:
            return None if event is None else event["self_cuda_time_us"] / 1000

        results.append(
            {
                "prompt_id": report["prompt_id"],
                "prompt_tokens": report["prompt_tokens"],
                "profiled_output_tokens": report["output_tokens"],
                "cycles": report["cycles"],
                "accepted_draft_tokens": report["accepted_draft_tokens"],
                "draft_tokens": report["draft_tokens"],
                "target_forwards": target_forwards,
                "target_rows": report["target_rows"],
                "wall_ms": {
                    "total": report["block_total_ms"],
                    "prefill": report["prefill_ms"],
                    "draft": report["draft_ms"],
                    "verify": report["verify_ms"],
                },
                "operators": {
                    "packed_nf4_linear": {
                        "event_key": None if packed is None else packed["key"],
                        "calls": None if packed is None else packed["calls"],
                        "gpu_event_ms": ms(packed),
                        "calls_per_target_forward": None
                        if packed is None or target_forwards == 0
                        else packed["calls"] / target_forwards,
                        "gpu_event_ms_per_target_forward": None
                        if packed is None or target_forwards == 0
                        else ms(packed) / target_forwards,
                    },
                    "lm_head_block_rows": {
                        "input_shape": None if output_head is None else output_head["input_shapes"],
                        "calls": None if output_head is None else output_head["calls"],
                        "gpu_event_ms": ms(output_head),
                        "gpu_event_ms_per_target_forward": None
                        if output_head is None or target_forwards == 0
                        else ms(output_head) / target_forwards,
                    },
                    "lm_head_prefill": {
                        "input_shape": None if prefill_vocab is None else prefill_vocab["input_shapes"],
                        "calls": None if prefill_vocab is None else prefill_vocab["calls"],
                        "gpu_event_ms": ms(prefill_vocab),
                    },
                    "fixed_row_attention": {
                        "calls": None if attention is None else attention["calls"],
                        "calls_per_target_forward": None
                        if attention is None or target_forwards == 0
                        else attention["calls"] / target_forwards,
                        "gpu_event_ms": ms(attention),
                    },
                    "bnb_4bit_prefill_linears": {
                        "events": [
                            {
                                "input_shapes": event["input_shapes"],
                                "calls": event["calls"],
                                "gpu_event_ms": ms(event),
                            }
                            for event in bnb_prefill
                        ],
                        "summed_gpu_event_ms": sum(ms(e) or 0.0 for e in bnb_prefill),
                    },
                },
                "trace": str(path.parent / "trace.json"),
            }
        )

    result = {
        "schema": "zassd-k2r3-operator-profile-summary-v1",
        "status": "diagnostic profiler attribution; not a latency benchmark",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "profile": {
            "schedule": "K=2/R=3, triton_batched attention, packed NF4 rows2 N16/K256/one warp",
            "generation_budget": 32,
            "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "torch": "2.11.0+cu128",
            "triton": "3.6.0",
        },
        "interpretation_limits": [
            "Profiler instrumentation changes execution cost; use the separate closed-loop timing artifacts for performance claims.",
            "CUDA event totals across operators can overlap and should not be summed into an exact decomposition of verify wall time.",
            "The BNB linears are classified as prefill from their sequence-shaped input tensors; block verification uses three-row tensors and the packed Triton path.",
        ],
        "profiles": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "profiles": len(results)}, indent=2))
    for profile in results:
        ops = profile["operators"]
        print(
            f"{profile['prompt_id']}: target_forwards={profile['target_forwards']}, "
            f"verify={profile['wall_ms']['verify']:.1f}ms; "
            f"packed NF4={ops['packed_nf4_linear']['gpu_event_ms']:.1f}ms "
            f"({ops['packed_nf4_linear']['calls_per_target_forward']:.0f} launches/forward); "
            f"LM head={ops['lm_head_block_rows']['gpu_event_ms']:.1f}ms; "
            f"attention={ops['fixed_row_attention']['gpu_event_ms']:.1f}ms"
        )


if __name__ == "__main__":
    main()
