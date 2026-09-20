"""Test 2 — Self-Speculative Exactness & Numerical Stability Test.

Evaluates greedy consistency between vanilla autoregressive decoding and
self-speculative decoding across fixed benchmark prompts.

Logs detailed divergence diagnostics for numerical stability analysis:
{
  "prompt_id": 12,
  "match": false,
  "first_divergence": 17,
  "vanilla_token": 1234,
  "spec_token": 5678,
  "top1_logit": 4.31,
  "top2_logit": 4.30,
  "margin": 0.01
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from zassd.decoding.speculative import self_speculative_generate
from zassd.decoding.vanilla import vanilla_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter


@pytest.fixture(scope="module")
def setup_pipeline(session_pipeline):
    model, tok, adapter, layer_mgr = session_pipeline

    cka_file = Path("experiments/03_cka/benchmark_results.json")
    if cka_file.exists():
        with open(cka_file) as f:
            cka_data = json.load(f)
        skip_indices = cka_data.get("cka_75", {}).get("skipped_indices", [3, 5, 7, 9, 11, 13, 16, 18, 21])
    else:
        skip_indices = [3, 5, 7, 9, 11, 13, 16, 18, 21]

    prompts_path = Path("data/benchmarks/prompts.jsonl")
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))

    return model, tok, layer_mgr, skip_indices, prompts


class TestSpeculativeExactness:
    """Validate exactness and document any quantized numerical stability limits."""

    def test_exactness_on_benchmark_prompts(self, setup_pipeline):
        """Run exactness check across 20 fixed prompts and record stability log."""
        model, tok, layer_mgr, skip_indices, prompts = setup_pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        eval_prompts = prompts[:20]
        results_dir = Path("results/raw")
        results_dir.mkdir(parents=True, exist_ok=True)

        stability_records: list[dict[str, Any]] = []
        exact_matches = 0

        for p in eval_prompts:
            p_id = p["id"]
            p_text = p["prompt"]

            # Vanilla generation
            v_text, v_metrics = vanilla_generate(
                model=model,
                tokenizer=tok,
                prompt=p_text,
                max_new_tokens=48,
                temperature=0.0,
                device=device,
            )
            v_tokens = tok.encode(v_text, add_special_tokens=False)

            # Self-speculative generation
            s_text, s_metrics = self_speculative_generate(
                model=model,
                tokenizer=tok,
                layer_mgr=layer_mgr,
                skip_indices=skip_indices,
                prompt=p_text,
                k=2,
                max_new_tokens=48,
                temperature=0.0,
                device=device,
            )
            s_tokens = tok.encode(s_text, add_special_tokens=False)

            is_match = (v_text == s_text) or (v_tokens == s_tokens)
            min_len = min(len(v_tokens), len(s_tokens))

            if is_match:
                exact_matches += 1
                record = {
                    "prompt_id": p_id,
                    "match": True,
                    "first_divergence": None,
                    "vanilla_token": None,
                    "spec_token": None,
                    "top1_logit": None,
                    "top2_logit": None,
                    "margin": None,
                }
            else:
                first_div = None
                v_tok = None
                s_tok = None
                for i in range(min_len):
                    if v_tokens[i] != s_tokens[i]:
                        first_div = i
                        v_tok = v_tokens[i]
                        s_tok = s_tokens[i]
                        break
                if first_div is None:
                    first_div = min_len
                    v_tok = v_tokens[min_len] if min_len < len(v_tokens) else None
                    s_tok = s_tokens[min_len] if min_len < len(s_tokens) else None

                # Compute logits and margin at divergence position using full model
                context_ids = tok(p_text, return_tensors="pt")["input_ids"].to(device)
                if first_div > 0:
                    prefix_slice = torch.tensor([v_tokens[:first_div]], device=device)
                    context_ids = torch.cat([context_ids, prefix_slice], dim=-1)

                with torch.no_grad():
                    diag_out = model(context_ids, use_cache=False)
                div_logits = diag_out.logits[0, -1, :].float()
                top_vals, top_indices = torch.topk(div_logits, 2)
                top1_val = float(top_vals[0].item())
                top2_val = float(top_vals[1].item())
                margin = float(top1_val - top2_val)

                record = {
                    "prompt_id": p_id,
                    "match": False,
                    "first_divergence": first_div,
                    "vanilla_token": v_tok,
                    "spec_token": s_tok,
                    "top1_logit": round(top1_val, 4),
                    "top2_logit": round(top2_val, 4),
                    "margin": round(margin, 4),
                }

            stability_records.append(record)

        # Save stability analysis report
        out_path = results_dir / "exactness_stability_analysis.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "num_evaluated": len(eval_prompts),
                    "exact_matches": exact_matches,
                    "exact_match_rate": exact_matches / len(eval_prompts),
                    "records": stability_records,
                },
                f,
                indent=2,
            )

        match_rate = exact_matches / len(eval_prompts)
        print(f"\nExactness Match Rate: {exact_matches}/{len(eval_prompts)} ({match_rate:.1%})")
        print(f"Detailed stability analysis saved to {out_path}")
        assert match_rate >= 0.5, f"Exact match rate unexpectedly low: {match_rate:.1%}"
