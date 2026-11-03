#!/usr/bin/env python3
"""EOS-termination audit for the B2 loops (eager and lazy LM head) vs original BNB greedy (development only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
import benchmark_bnb_nf4_activationvec_fallback_closed_loop as eager_mod
import benchmark_b2_lazy_lm_head as lazy_mod
from _common import run_vanilla_timed

SYNTHETIC = [
    "Answer: yes<|im_end|>\nAnswer: yes<|im_end|>\nAnswer: yes<|im_end|>\nAnswer: yes",
    "The result is forty two.<|im_end|>\nThe result is forty two.<|im_end|>\nThe result is forty two.",
    "Q: 2+2? A: 4<|im_end|>\nQ: 3+3? A: 6<|im_end|>\nQ: 2+2? A: 4<|im_end|>\nQ: 3+3? A: 6<|im_end|>\nQ: 2+2? A:",
    "ok<|im_end|>\nok<|im_end|>\nok<|im_end|>\nok<|im_end|>\nok",
]


CHAT_QUESTIONS = [
    "Repeat this sentence exactly three times, one per line: The quick brown fox jumps over the lazy dog.",
    "Write the word 'banana' five times separated by commas and then stop.",
    "Count from one to ten in words, separated by commas.",
    "Give a two-sentence summary of why the sky is blue.",
    "List the days of the week, one per line.",
    "Translate 'good morning, how are you today' into French and Spanish, then stop.",
    "Say hello three times: Hello! Hello! Hello!",
    "What is 12 times 12? Answer briefly.",
]


def counting(module, counter):
    original = module.find_candidate_tokens

    def wrapped(tokens, *a, **kw):
        out = original(tokens, *a, **kw)
        counter["cycles"] += 1
        counter["eos_candidate_cycles"] += int(module._eos in out)
        return out

    module.find_candidate_tokens = wrapped
    return original


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-ids", nargs="*", default=[])
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    eos = tok.eos_token_id
    eager_mod._eos = lazy_mod._eos = eos
    chat = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) for q in CHAT_QUESTIONS]
    prompts = ([(pid, rows[pid]["prompt"]) for pid in args.prompt_ids] + [(f"synthetic_{i}", s) for i, s in enumerate(SYNTHETIC)]
               + [(f"chat_{i}", c) for i, c in enumerate(chat)])
    per = []
    for pid, text in prompts:
        ids = tok(text, return_tensors="pt")["input_ids"].to(device)
        cap = int(ids.shape[-1]) + args.max_new_tokens + 8
        ref, _, _, _ = run_vanilla_timed(model, tok, ids, args.max_new_tokens, device)
        counter = {"cycles": 0, "eos_candidate_cycles": 0}
        orig_e, orig_l = counting(eager_mod, counter), counting(lazy_mod, counter)
        try:
            eager_tokens, _ = eager_mod.generate(model, tok, ids, k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                                                 device=device, capacity=cap, fallback_on_empty=True)
            lazy_tokens, _ = lazy_mod.generate_lazy(model, tok, ids, max_new_tokens=args.max_new_tokens, device=device, capacity=cap)
        finally:
            eager_mod.find_candidate_tokens, lazy_mod.find_candidate_tokens = orig_e, orig_l
        per.append({"prompt_id": pid, "bnb_tokens": len(ref), "ended_with_eos": ref[-1] == eos,
                    "eager_matches_bnb": eager_tokens == ref, "lazy_matches_bnb": lazy_tokens == ref,
                    "eos_candidate_cycles": counter["eos_candidate_cycles"]})
        print(json.dumps(per[-1]))
    summary = {"prompts": len(per), "ended_with_eos": sum(p["ended_with_eos"] for p in per),
               "prompts_with_eos_draft": sum(p["eos_candidate_cycles"] > 0 for p in per),
               "all_match": all(p["eager_matches_bnb"] and p["lazy_matches_bnb"] for p in per)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-b2-eos-termination-v1", "summary": summary, "per_prompt": per}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
