"""Locate the first generated-token disagreement between experimental B1 block and BNB."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_row_kernel_v1"
SUMMARY = EXP / "generation_budget_humaneval114_115_summary.json"


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def main():
    summary = json.loads(SUMMARY.read_text())
    groups = defaultdict(list)
    for prompt_id, prompt in summary["prompts"].items():
        for budget_text, budget in prompt["budgets"].items():
            for pair in budget["paired_rounds"]:
                artifact = json.loads((EXP / pair["artifact"]).read_text())
                round_record = artifact["timing"]["paired_rounds"][pair["round"]]
                methods = round_record["methods"]
                block = methods["optimized_b1_block"]["token_ids"]
                singleton = methods["optimized_b1_singleton"]["token_ids"]
                bnb = methods["original_bnb_greedy"]["token_ids"]
                groups[(prompt_id, int(budget_text))].append({
                    "artifact": pair["artifact"],
                    "round": pair["round"],
                    "block_matches_b1": block == singleton,
                    "block_matches_bnb": block == bnb,
                    "generated_tokens_block": len(block),
                    "generated_tokens_bnb": len(bnb),
                    "first_bnb_divergence_index_zero_based": first_difference(block, bnb),
                    "matching_token_prefix_length": (
                        first_difference(block, bnb) if first_difference(block, bnb) is not None
                        else min(len(block), len(bnb))
                    ),
                })
    result = []
    for (prompt_id, budget), rounds in sorted(groups.items()):
        divergences = [r["first_bnb_divergence_index_zero_based"] for r in rounds if r["first_bnb_divergence_index_zero_based"] is not None]
        result.append({
            "prompt_id": prompt_id,
            "max_new_tokens": budget,
            "timing_rounds": len(rounds),
            "all_block_runs_match_b1_tokens": all(r["block_matches_b1"] for r in rounds),
            "bnb_token_matches": sum(r["block_matches_bnb"] for r in rounds),
            "bnb_comparisons": len(rounds),
            "first_divergence_index_range_zero_based": None if not divergences else [min(divergences), max(divergences)],
            "matching_prefix_length_range": [min(r["matching_token_prefix_length"] for r in rounds), max(r["matching_token_prefix_length"] for r in rounds)],
            "rounds": rounds,
        })
    output = {
        "status": "token-trace diagnostic only; no BNB KV/state identity claim",
        "source_summary": SUMMARY.name,
        "groups": result,
        "interpretation_limit": "First token disagreement locates B2 token divergence only; matching tokens do not establish matching BNB KV state.",
    }
    target = EXP / "bnb_divergence_horizons_humaneval_sweep.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps([{
        k: group[k] for k in (
            "prompt_id", "max_new_tokens", "bnb_token_matches", "bnb_comparisons",
            "first_divergence_index_range_zero_based", "matching_prefix_length_range",
        )
    } for group in result], indent=2))
    print(f"output={target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
