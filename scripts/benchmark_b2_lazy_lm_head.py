#!/usr/bin/env python3
"""B2 fallback loop with lazy per-row LM head: rows after the first rejected draft are never projected."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.ngram import find_candidate_tokens
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_activationvec_fallback_kv_closed_loop import bnb_reference
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from benchmark_bnb_nf4_activationvec_fallback_closed_loop import SwitchableSchedule, generate
from _common import run_vanilla_timed


def generate_lazy(model, tokenizer, ids, *, max_new_tokens, device, capacity, k=2, ngram_size=3, observer=None):
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    history = ids[0].tolist()
    emitted, eos = [], tokenizer.eos_token_id
    stats = {"fallback_cycles": 0, "block_cycles": 0, "lm_head_rows": 0, "block_rows": 0}
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with SwitchableSchedule(model) as schedule, torch.inference_mode():
        schedule.use("plain")
        pending = int(model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :].argmax(dim=-1).item())
        if observer is not None:
            observer(cache)
        while len(emitted) < max_new_tokens:
            if pending == eos or len(emitted) + 1 == max_new_tokens:
                emitted.append(pending)
                break
            cands = find_candidate_tokens(history + [pending], ngram_size=ngram_size,
                                          max_candidates=min(k, max_new_tokens - len(emitted) - 1))
            if eos in cands:
                cands = cands[:cands.index(eos) + 1]
            if not cands:
                schedule.use("plain")
                tok = torch.tensor([[pending]], device=device, dtype=ids.dtype)
                nxt = int(model(tok, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :].argmax(dim=-1).item())
                emitted.append(pending); history.append(pending); pending = nxt
                stats["fallback_cycles"] += 1
                if observer is not None:
                    observer(cache)
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
            stats["block_cycles"] += 1
            stats["lm_head_rows"] += len(preds)
            stats["block_rows"] += len(inputs)
            emitted.append(pending); history.append(pending)
            acc = 0
            for pos, c in enumerate(cands):
                if rej is not None and pos == rej:
                    break
                emitted.append(c); history.append(c); acc += 1
                if c == eos:
                    break
            if rej is not None:
                cache.crop(prefix_len + rej + 1); pending = preds[rej]
            elif has_eos and acc == len(cands):
                break
            else:
                pending = preds[-1]
            if emitted[-1] == eos:
                break
            if observer is not None and len(emitted) < max_new_tokens:
                observer(cache)
    torch.cuda.synchronize(device)
    stats["total_s"] = time.perf_counter() - start
    return emitted[:max_new_tokens], stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timing-rounds", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    per = []
    for pid in args.prompt_ids:
        ids = tokenizer(rows[pid]["prompt"], return_tensors="pt")["input_ids"].to(device)
        plen = int(ids.shape[-1])
        cap = plen + args.max_new_tokens + 8
        ref_tokens, ref_digests = bnb_reference(model, ids, args.max_new_tokens, cap)
        seen = {}
        def obs(cache):
            n = cache.get_seq_length() - plen
            seen[n] = diagnostic_cache_prefix_digest(cache, plen + n)
        toks, _ = generate_lazy(model, tokenizer, ids, max_new_tokens=args.max_new_tokens, device=device, capacity=cap, observer=obs)
        kv_ok = bool(seen) and all(ref_digests.get(n) == d for n, d in seen.items())
        names = ["bnb", "b2", "b2_lazy"]
        times = {n: [] for n in names}
        exact = True
        for r in range(args.timing_rounds):
            for name in names[r % 3:] + names[:r % 3]:
                if name == "bnb":
                    t, _, _, ms = run_vanilla_timed(model, tokenizer, ids, args.max_new_tokens, device)
                    times[name].append(ms / 1000)
                elif name == "b2":
                    t, st = generate(model, tokenizer, ids, k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                                     device=device, capacity=cap, fallback_on_empty=True)
                    times[name].append(st["total_s"]); exact &= t == ref_tokens
                else:
                    t, st = generate_lazy(model, tokenizer, ids, max_new_tokens=args.max_new_tokens, device=device, capacity=cap)
                    times[name].append(st["total_s"]); exact &= t == ref_tokens; lazy_stats = st
        med = {n: statistics.median(v) for n, v in times.items()}
        per.append({"prompt_id": pid, "tokens_match_bnb": exact and toks == ref_tokens, "kv_prefixes_equal": kv_ok,
                    "observed_prefixes": len(seen), "b2_speedup": med["bnb"] / med["b2"], "lazy_speedup": med["bnb"] / med["b2_lazy"],
                    "lazy_over_b2": med["b2"] / med["b2_lazy"], "lm_head_rows": lazy_stats["lm_head_rows"], "block_rows": lazy_stats["block_rows"]})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in per[-1].items()}))
    gm = lambda key: statistics.geometric_mean(p[key] for p in per)
    summary = {"prompts": len(per), "all_tokens_match_bnb": all(p["tokens_match_bnb"] for p in per),
               "all_kv_prefixes_equal": all(p["kv_prefixes_equal"] for p in per),
               "geomean_b2_speedup": gm("b2_speedup"), "geomean_lazy_speedup": gm("lazy_speedup"), "geomean_lazy_over_b2": gm("lazy_over_b2")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-b2-lazy-lm-head-v1", "benchmark": args.benchmark, "summary": summary, "per_prompt": per}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
