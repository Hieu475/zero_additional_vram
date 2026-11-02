#!/usr/bin/env python3
"""Prompt-cluster bootstrap summary for benchmark_b2_fallback_vs_b1 outputs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def boot(xs, draws, rng):
    vals = sorted(gm([rng.choice(xs) for _ in xs]) for _ in range(draws))
    return vals[int(0.025 * draws)], vals[int(0.975 * draws) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = {}
    for path in args.inputs:
        d = json.loads(path.read_text())
        pp = d["per_prompt"]
        b2 = [p["b2_speedup_vs_bnb"] for p in pp]
        b1 = [p["b1_speedup_vs_bnb"] for p in pp]
        ratio = [x / y for x, y in zip(b2, b1)]
        out[path.name] = {
            "prompts": len(pp), "max_new_tokens": d["max_new_tokens"],
            "b2_geomean_vs_bnb": gm(b2), "b2_ci95": boot(b2, args.draws, rng),
            "b1_geomean_vs_bnb": gm(b1), "b1_ci95": boot(b1, args.draws, rng),
            "b2_over_b1_geomean": gm(ratio), "b2_over_b1_ci95": boot(ratio, args.draws, rng),
            "b2_exact_to_bnb": f"{d['b2_exact_to_bnb_prompts']}/{len(pp)}",
            "b1_exact_to_bnb": f"{d['b1_exact_to_bnb_prompts']}/{len(pp)}",
            "b1_block_exact_to_b1_singleton": f"{d['b1_block_exact_to_b1_singleton_prompts']}/{len(pp)}",
            "b2_prompts_below_bnb": sum(x < 1 for x in b2), "b1_prompts_below_bnb": sum(x < 1 for x in b1),
        }
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
