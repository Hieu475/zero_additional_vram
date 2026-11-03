#!/usr/bin/env python3
"""Predict B2 speedup for block sizes K+1 = 2..5 from recorded greedy sequences and a calibrated cost model (CPU only).

The proposal policy (PLD, ngram<=3) is run against the true BNB greedy continuation to count accepted tokens exactly.
Cycle cost (ms): fallback F; block = F - h + s*(rows-1) + h*(lm_head rows actually computed), h = per-row LM head cost.
The per-extra-row cost s is calibrated so that K=2 reproduces the measured B2-lazy speedup of each cohort; the same s is then
used to predict other K. Prefill is excluded (common to both sides). Development data only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from zassd.decoding.ngram import find_candidate_tokens

MODEL, REV = "Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def simulate(prompt_ids, truth, k, ngram=3):
    history = list(prompt_ids)
    n = len(truth)
    p, fallback, blocks, rows_total, head_rows, accepted_total, drafted = 0, 0, [], 0, 0, 0, 0
    while p < n:
        if p == n - 1:
            p += 1
            break
        cands = find_candidate_tokens(history + truth[:p + 1], ngram_size=ngram, max_candidates=min(k, n - p - 1))
        if not cands:
            fallback += 1
            p += 1
            continue
        a = 0
        while a < len(cands) and cands[a] == truth[p + 1 + a]:
            a += 1
        blocks.append((1 + len(cands), a + 1))
        drafted += len(cands)
        accepted_total += a
        p += 1 + a
    return fallback, blocks, accepted_total, drafted


def cycle_time(fallback, blocks, F, h, s):
    return fallback * F + sum(F - h + s * (r - 1) + h * hr for r, hr in blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", nargs=4, action="append", metavar=("NAME", "BENCHMARK", "RESULT_JSON", "MEASURED_LAZY_SPEEDUP"), required=True)
    ap.add_argument("--F", type=float, default=24.0)
    ap.add_argument("--h", type=float, default=3.3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    out = {"F_ms": args.F, "h_ms": args.h, "cohorts": {}}
    for name, bench, result, measured in args.cohort:
        rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(bench).read_text().splitlines() if l.strip())}
        data = json.loads(Path(result).read_text())["per_prompt"]
        items = []
        for p in data:
            truth = p["rounds"][0]["methods"]["original_bnb_greedy"]["tokens"]
            items.append((tok(rows[p["prompt_id"]]["prompt"])["input_ids"], truth))
        measured = float(measured)

        def mean_speedup(k, s):
            vals = []
            for ids, truth in items:
                fb, blocks, _, _ = simulate(ids, truth, k)
                vals.append(len(truth) * args.F / cycle_time(fb, blocks, args.F, args.h, s))
            return statistics.geometric_mean(vals)

        lo, hi = 0.0, 40.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if mean_speedup(2, mid) > measured:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2
        entry = {"measured_k2_lazy_speedup": measured, "calibrated_s_ms_per_extra_row": s, "prompts": len(items), "predicted": {}}
        for k in (1, 2, 3, 4):
            stats = [simulate(ids, truth, k) for ids, truth in items]
            drafted = sum(x[3] for x in stats)
            entry["predicted"][f"k{k}"] = {
                "speedup_at_calibrated_s": mean_speedup(k, s),
                "speedup_if_s_half": mean_speedup(k, s / 2),
                "acceptance": sum(x[2] for x in stats) / drafted if drafted else None,
                "mean_blocks": statistics.mean(len(x[1]) for x in stats),
                "mean_tokens_per_block": statistics.mean((sum(a for _, a in x[1])) / max(1, len(x[1])) for x in stats)}
        out["cohorts"][name] = entry
        print(name, "s=%.2f ms/row" % s, {k: round(v["speedup_at_calibrated_s"], 3) for k, v in entry["predicted"].items()},
              "| s/2:", {k: round(v["speedup_if_s_half"], 3) for k, v in entry["predicted"].items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
