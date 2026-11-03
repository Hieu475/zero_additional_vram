#!/usr/bin/env python3
"""Development-only comparison of draft-selection policies in the B2 lazy-LM-head loop (k=2, rows=3).

Policies only change which tokens are proposed; every proposal is verified by the same exact path, and output
tokens are compared with eager BNB greedy for every policy and round.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from benchmark_bnb_nf4_activationvec_fallback_closed_loop import SwitchableSchedule
from _common import run_vanilla_timed

POLICIES = {
    "base": dict(ngram_max=3, min_size=2, vote=False, min_reps=1),
    "s3": dict(ngram_max=3, min_size=3, vote=False, min_reps=1),
    "s4": dict(ngram_max=4, min_size=3, vote=False, min_reps=1),
    "s5": dict(ngram_max=5, min_size=4, vote=False, min_reps=1),
    "vote": dict(ngram_max=3, min_size=2, vote=True, min_reps=1),
    "vote_s3": dict(ngram_max=4, min_size=3, vote=True, min_reps=1),
    "rep2": dict(ngram_max=3, min_size=2, vote=False, min_reps=2),
}


def propose(tokens, max_candidates, ngram_max, min_size, vote, min_reps):
    if max_candidates <= 0:
        return []
    for size in range(min(ngram_max, len(tokens)), min_size - 1, -1):
        if size < 1:
            break
        query = tokens[-size:]
        tails = []
        for i in range(len(tokens) - size):
            if tokens[i:i + size] == query:
                tail = tokens[i + size:i + size + max_candidates]
                if tail:
                    tails.append(tail)
        if not tails:
            continue
        if size < 3 and len(tails) < min_reps:
            continue
        if vote:
            counts = Counter(t[0] for t in tails)
            best = max(counts.values())
            for tail in reversed(tails):
                if counts[tail[0]] == best:
                    return tail
        return tails[-1]
    return []


def generate_policy(model, tokenizer, ids, policy, *, max_new_tokens, device, capacity, k=2):
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    history = ids[0].tolist()
    emitted, eos = [], tokenizer.eos_token_id
    st = {"fallback": 0, "block": 0, "draft": 0, "accepted": 0}
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with SwitchableSchedule(model) as schedule, torch.inference_mode():
        schedule.use("plain")
        pending = int(model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :].argmax(dim=-1).item())
        while len(emitted) < max_new_tokens:
            if pending == eos or len(emitted) + 1 == max_new_tokens:
                emitted.append(pending)
                break
            cands = propose(history + [pending], min(k, max_new_tokens - len(emitted) - 1), **policy)
            if eos in cands:
                cands = cands[:cands.index(eos) + 1]
            if not cands:
                schedule.use("plain")
                tok = torch.tensor([[pending]], device=device, dtype=ids.dtype)
                nxt = int(model(tok, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :].argmax(dim=-1).item())
                emitted.append(pending); history.append(pending); pending = nxt
                st["fallback"] += 1
                continue
            prefix_len = cache.get_seq_length()
            has_eos = bool(cands[-1] == eos)
            inputs = [pending] + (cands[:-1] if has_eos else cands)
            schedule.use("patched")
            tensor = torch.tensor([inputs], device=device, dtype=ids.dtype)
            hidden = model.model(input_ids=tensor, past_key_values=cache.cache, use_cache=True).last_hidden_state
            preds, rej = [], None
            for row in range(len(inputs)):
                p = int(model.lm_head(hidden[:, row:row + 1, :]).argmax(dim=-1).item())
                preds.append(p)
                if row < len(cands) and p != cands[row]:
                    rej = row
                    break
            st["block"] += 1
            st["draft"] += len(cands)
            emitted.append(pending); history.append(pending)
            acc = 0
            for pos, c in enumerate(cands):
                if rej is not None and pos == rej:
                    break
                emitted.append(c); history.append(c); acc += 1
                if c == eos:
                    break
            st["accepted"] += acc
            if rej is not None:
                cache.crop(prefix_len + rej + 1); pending = preds[rej]
            elif has_eos and acc == len(cands):
                break
            else:
                pending = preds[-1]
            if emitted[-1] == eos:
                break
    torch.cuda.synchronize(device)
    st["total_s"] = time.perf_counter() - start
    return emitted[:max_new_tokens], st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timing-rounds", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    names = ["bnb"] + args.policies
    per = []
    for pid in args.prompt_ids:
        ids = tok(rows[pid]["prompt"], return_tensors="pt")["input_ids"].to(device)
        cap = int(ids.shape[-1]) + args.max_new_tokens + 8
        times = {n: [] for n in names}
        exact = {n: True for n in args.policies}
        stats = {}
        for r in range(args.timing_rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            toks = {}
            for n in order:
                if n == "bnb":
                    t, _, _, ms = run_vanilla_timed(model, tok, ids, args.max_new_tokens, device)
                    times[n].append(ms / 1000)
                else:
                    t, s = generate_policy(model, tok, ids, POLICIES[n], max_new_tokens=args.max_new_tokens, device=device, capacity=cap)
                    times[n].append(s["total_s"]); stats[n] = s
                toks[n] = t
            for n in args.policies:
                exact[n] = exact[n] and toks[n] == toks["bnb"]
        med = {n: statistics.median(v) for n, v in times.items()}
        per.append({"prompt_id": pid, "exact": exact,
                    "speedup": {n: med["bnb"] / med[n] for n in args.policies},
                    "acceptance": {n: (stats[n]["accepted"] / stats[n]["draft"] if stats[n]["draft"] else None) for n in args.policies},
                    "block_cycles": {n: stats[n]["block"] for n in args.policies},
                    "fallback_cycles": {n: stats[n]["fallback"] for n in args.policies}})
        print(json.dumps({"prompt_id": pid, "speedup": {n: round(v, 3) for n, v in per[-1]["speedup"].items()}, "exact": all(exact.values())}))
    summary = {n: {"geomean_speedup": statistics.geometric_mean(p["speedup"][n] for p in per),
                   "below_bnb": sum(p["speedup"][n] < 1 for p in per), "exact_prompts": sum(p["exact"][n] for p in per),
                   "mean_block_cycles": statistics.mean(p["block_cycles"][n] for p in per)} for n in args.policies}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-b2-draft-policies-v1", "benchmark": args.benchmark, "policies": {n: POLICIES[n] for n in args.policies},
                                       "summary": summary, "per_prompt": per}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
