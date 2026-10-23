#!/usr/bin/env python3
"""Summarize backend-specific matched-prefix numerical traces."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    prompts = data.get("prompts", [])
    n = len(prompts)
    cycles = [cycle for prompt in prompts for cycle in prompt.get("cycles", [])]
    margins = [p["vanilla_top1_margin_at_divergence"] for p in prompts
               if p.get("vanilla_top1_margin_at_divergence") is not None]
    flips = [c for c in cycles if c.get("first_argmax_difference_in_cycle") is not None]
    return {
        "path": str(path), "weight_backend": data.get("weight_backend", data.get("backend")),
        "model_revision": data.get("model_revision"), "prompts": n,
        "sequence_mismatches": sum(p.get("first_token_divergence_index") is not None for p in prompts),
        "first_kv_divergence_prompts": sum(p.get("first_kv_divergence_cycle") is not None for p in prompts),
        "first_logit_divergence_prompts": sum(p.get("first_logit_divergence_cycle") is not None for p in prompts),
        "cycle_count": len(cycles), "cycles_with_batch_vs_canonical_argmax_flip": len(flips),
        "mismatch_margins": margins,
        "per_prompt": [{
            "prompt_id": p["prompt_id"], "first_kv_cycle": p.get("first_kv_divergence_cycle"),
            "first_logit_cycle": p.get("first_logit_divergence_cycle"),
            "first_sequence_mismatch": p.get("first_token_divergence_index"),
            "vanilla_margin_at_mismatch": p.get("vanilla_top1_margin_at_divergence"),
            "argmax_agreement_at_first_mismatch": p.get("argmax_agreement_at_first_divergence"),
            "mismatch_cycle": p.get("first_token_divergence_cycle"),
            "mismatch_logit_row": p.get("first_token_divergence_logit_row"),
        } for p in prompts],
    }


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args()
    payload={"backend_summaries":[summarize(p) for p in args.inputs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(args.output)

if __name__ == "__main__": main()
