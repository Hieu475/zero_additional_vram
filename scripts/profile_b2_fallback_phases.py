#!/usr/bin/env python3
"""Per-phase wall-time profile of the B2 activationvec+fallback loop (development prompts only)."""

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
from benchmark_bnb_nf4_activationvec_fallback_closed_loop import SwitchableSchedule, generate
from _common import run_vanilla_timed

PHASES = ("setup", "prefill", "lookup", "switch", "fallback_forward", "block_forward", "commit")


def timed_generate(model, tokenizer, ids, *, max_new_tokens, device, capacity, k=2, ngram_size=3):
    t = dict.fromkeys(PHASES, 0.0)
    n = {"fallback": 0, "block": 0, "block_rows": 0, "draft": 0, "accepted": 0}
    clock = time.perf_counter
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    history = ids[0].tolist()
    emitted, eos = [], tokenizer.eos_token_id
    torch.cuda.synchronize(device)
    start = clock()
    s0 = clock()
    schedule = SwitchableSchedule(model)
    schedule.__enter__()
    t["setup"] += clock() - s0
    try:
        with torch.inference_mode():
            s0 = clock(); schedule.use("plain"); t["switch"] += clock() - s0
            s0 = clock()
            pending = int(model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :].argmax(dim=-1).item())
            t["prefill"] += clock() - s0
            while len(emitted) < max_new_tokens:
                if pending == eos or len(emitted) + 1 == max_new_tokens:
                    emitted.append(pending)
                    break
                s0 = clock()
                cands = find_candidate_tokens(history + [pending], ngram_size=ngram_size,
                                              max_candidates=min(k, max_new_tokens - len(emitted) - 1))
                if eos in cands:
                    cands = cands[:cands.index(eos) + 1]
                t["lookup"] += clock() - s0
                n["draft"] += len(cands)
                if not cands:
                    s0 = clock(); schedule.use("plain"); t["switch"] += clock() - s0
                    s0 = clock()
                    tok = torch.tensor([[pending]], device=device, dtype=ids.dtype)
                    nxt = int(model(tok, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :].argmax(dim=-1).item())
                    t["fallback_forward"] += clock() - s0
                    emitted.append(pending); history.append(pending); pending = nxt
                    n["fallback"] += 1
                    continue
                prefix_len = cache.get_seq_length()
                has_eos = bool(cands[-1] == eos)
                inputs = [pending] + (cands[:-1] if has_eos else cands)
                s0 = clock(); schedule.use("patched"); t["switch"] += clock() - s0
                s0 = clock()
                tensor = torch.tensor([inputs], device=device, dtype=ids.dtype)
                preds = model(tensor, past_key_values=cache.cache, use_cache=True).logits[0].argmax(dim=-1).tolist()
                t["block_forward"] += clock() - s0
                n["block"] += 1; n["block_rows"] += len(inputs)
                s0 = clock()
                emitted.append(pending); history.append(pending)
                rej, acc = None, 0
                for pos, c in enumerate(cands):
                    if preds[pos] != c:
                        rej = pos
                        break
                    emitted.append(c); history.append(c); acc += 1
                    if c == eos:
                        break
                n["accepted"] += acc
                if rej is not None:
                    cache.crop(prefix_len + rej + 1); pending = preds[rej]
                elif has_eos and acc == len(cands):
                    t["commit"] += clock() - s0
                    break
                else:
                    pending = preds[-1]
                t["commit"] += clock() - s0
                if emitted[-1] == eos:
                    break
    finally:
        s0 = clock()
        schedule.__exit__(None, None, None)
        t["setup"] += clock() - s0
    torch.cuda.synchronize(device)
    return emitted[:max_new_tokens], clock() - start, t, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    out = []
    for pid in args.prompt_ids:
        ids = tokenizer(rows[pid]["prompt"], return_tensors="pt")["input_ids"].to(device)
        cap = int(ids.shape[-1]) + args.max_new_tokens + 8
        run_vanilla_timed(model, tokenizer, ids, args.max_new_tokens, device)
        bnb_tokens, prefill_ms, decode_ms, total_ms = run_vanilla_timed(model, tokenizer, ids, args.max_new_tokens, device)
        timed_generate(model, tokenizer, ids, max_new_tokens=args.max_new_tokens, device=device, capacity=cap)
        toks, total, t, n = timed_generate(model, tokenizer, ids, max_new_tokens=args.max_new_tokens, device=device, capacity=cap)
        ref, _ = generate(model, tokenizer, ids, k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                          device=device, capacity=cap, fallback_on_empty=True)
        out.append({"prompt_id": pid, "tokens": len(toks), "matches_bnb": toks == bnb_tokens, "matches_generate": toks == ref,
                    "bnb_total_s": total_ms / 1000, "bnb_prefill_s": prefill_ms / 1000,
                    "bnb_decode_ms_per_token": decode_ms / max(1, len(bnb_tokens) - 1),
                    "b2_total_s": total, "phases_s": t, "counts": n,
                    "fallback_ms_per_call": 1000 * t["fallback_forward"] / max(1, n["fallback"]),
                    "block_ms_per_call": 1000 * t["block_forward"] / max(1, n["block"]),
                    "tokens_per_block_cycle": (n["accepted"] + n["block"]) / max(1, n["block"]),
                    "acceptance": n["accepted"] / n["draft"] if n["draft"] else None})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[-1].items() if k not in ("phases_s",)}))
    agg = {p: statistics.mean(o["phases_s"][p] / o["b2_total_s"] for o in out) for p in PHASES}
    summary = {"mean_phase_fraction_of_b2_total": agg,
               "mean_fallback_ms_per_call": statistics.mean(o["fallback_ms_per_call"] for o in out),
               "mean_block_ms_per_call": statistics.mean(o["block_ms_per_call"] for o in out),
               "mean_bnb_decode_ms_per_token": statistics.mean(o["bnb_decode_ms_per_token"] for o in out),
               "mean_tokens_per_block_cycle": statistics.mean(o["tokens_per_block_cycle"] for o in out),
               "mean_fallback_cycle_fraction": statistics.mean(o["counts"]["fallback"] / max(1, o["counts"]["fallback"] + o["counts"]["block"]) for o in out),
               "all_match_bnb": all(o["matches_bnb"] and o["matches_generate"] for o in out)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-b2-fallback-phase-profile-v1", "benchmark": args.benchmark,
                                       "max_new_tokens": args.max_new_tokens, "summary": summary, "per_prompt": out}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
