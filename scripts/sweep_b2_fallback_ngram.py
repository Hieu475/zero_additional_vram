#!/usr/bin/env python3
"""Development-only ngram_size sweep for the B2 activationvec+fallback loop (k=2, rows=3)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from benchmark_bnb_nf4_activationvec_fallback_closed_loop import generate
from _common import run_vanilla_timed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--ngram-sizes", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timing-rounds", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    per_prompt = []
    for pid in args.prompt_ids:
        ids = tokenizer(rows[pid]["prompt"], return_tensors="pt")["input_ids"].to(device)
        capacity = int(ids.shape[-1]) + args.max_new_tokens + 8
        names = ["bnb"] + [f"ngram{n}" for n in args.ngram_sizes]
        times = {n: [] for n in names}
        stats, exact = {}, {n: True for n in names[1:]}
        for r in range(args.timing_rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            ref = None
            res = {}
            for name in order:
                if name == "bnb":
                    tokens, _, _, ms = run_vanilla_timed(model, tokenizer, ids, args.max_new_tokens, device)
                    times[name].append(ms / 1000)
                    ref = tokens
                else:
                    tokens, st = generate(model, tokenizer, ids, k=2, ngram_size=int(name[5:]),
                                          max_new_tokens=args.max_new_tokens, device=device, capacity=capacity,
                                          fallback_on_empty=True)
                    times[name].append(st["total_s"])
                    stats[name] = st
                res[name] = tokens
            for n in names[1:]:
                exact[n] = exact[n] and res[n] == ref
        med = {n: statistics.median(v) for n, v in times.items()}
        entry = {"prompt_id": pid, "exact_to_bnb": exact,
                 "speedup": {n: med["bnb"] / med[n] for n in names[1:]},
                 "acceptance": {n: (stats[n]["accepted_draft_tokens"] / stats[n]["draft_tokens"]
                                    if stats[n]["draft_tokens"] else None) for n in names[1:]},
                 "block_cycles": {n: stats[n]["block_cycles"] for n in names[1:]}}
        per_prompt.append(entry)
        print(json.dumps({"prompt_id": pid, "speedup": {k: round(v, 3) for k, v in entry["speedup"].items()},
                          "exact": all(exact.values())}))
    summary = {}
    for n in [f"ngram{x}" for x in args.ngram_sizes]:
        summary[n] = {"geomean_speedup": statistics.geometric_mean(p["speedup"][n] for p in per_prompt),
                      "prompts_below_bnb": sum(p["speedup"][n] < 1 for p in per_prompt),
                      "exact_prompts": sum(p["exact_to_bnb"][n] for p in per_prompt)}
    out = {"schema": "zassd-b2-fallback-ngram-sweep-v1", "classification": "development-only parameter sweep",
           "benchmark": args.benchmark, "k": 2, "max_new_tokens": args.max_new_tokens,
           "timing_rounds": args.timing_rounds, "summary": summary, "per_prompt": per_prompt}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
