#!/usr/bin/env python3
"""Closed-loop K=2/R=3 activationvec benchmark with native-BNB M=1 fallback on empty proposals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.decoding.ngram import find_candidate_tokens
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_bnb_nf4_activationvec_block_integration import triton_m3_overrides
from _common import run_vanilla_timed


@contextmanager
def block_ctx(model):
    with (fixed_row_schedule_for_model(model, physical_rows=3, attention_strategy="bnb_rowwise",
                                       quantized_linear_strategy="fixed_rows"),
          triton_m3_overrides(model)):
        yield


class SwitchableSchedule:
    """Enter the block schedule once; swap per-module forwards between plain BNB and the schedule."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        modules = list(self.model.modules())
        plain = [m.forward for m in modules]
        self._stack = block_ctx(self.model)
        self._stack.__enter__()
        patched = [m.forward for m in modules]
        self._entries = [(m, p, q) for m, p, q in zip(modules, plain, patched) if p != q]
        self.mode = "patched"
        return self

    def use(self, mode):
        if mode == self.mode:
            return
        for module, plain, patched in self._entries:
            module.forward = plain if mode == "plain" else patched
        self.mode = mode

    def __exit__(self, *exc):
        self.use("patched")
        return self._stack.__exit__(*exc)


def generate(model, tokenizer, prompt_ids, *, k, ngram_size, max_new_tokens, device, capacity,
             fallback_on_empty, observer=None):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with SwitchableSchedule(model) as schedule:
        tokens, stats = _generate(model, schedule, tokenizer, prompt_ids, k=k, ngram_size=ngram_size,
                                  max_new_tokens=max_new_tokens, device=device, capacity=capacity,
                                  fallback_on_empty=fallback_on_empty, observer=observer)
    torch.cuda.synchronize(device)
    stats["total_s"] = time.perf_counter() - start
    return tokens, stats


def _generate(model, schedule, tokenizer, prompt_ids, *, k, ngram_size, max_new_tokens, device, capacity,
              fallback_on_empty, observer=None):
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    history = prompt_ids[0].tolist()
    emitted: list[int] = []
    eos = tokenizer.eos_token_id
    stats = {"cycles": 0, "fallback_cycles": 0, "block_cycles": 0, "draft_tokens": 0, "accepted_draft_tokens": 0}
    schedule.use("plain")
    with torch.inference_mode():
        pending = int(model(prompt_ids, past_key_values=cache.cache,
                            use_cache=True).logits[:, -1:, :].argmax(dim=-1).item())
        if observer is not None:
            observer(cache)
        while len(emitted) < max_new_tokens:
            if pending == eos or len(emitted) + 1 == max_new_tokens:
                emitted.append(pending)
                break
            candidates = find_candidate_tokens(history + [pending], ngram_size=ngram_size,
                                               max_candidates=min(k, max_new_tokens - len(emitted) - 1))
            if eos in candidates:
                candidates = candidates[:candidates.index(eos) + 1]
            stats["cycles"] += 1
            stats["draft_tokens"] += len(candidates)
            if not candidates and fallback_on_empty:
                schedule.use("plain")
                token = torch.tensor([[pending]], device=device, dtype=prompt_ids.dtype)
                nxt = int(model(token, past_key_values=cache.cache,
                                use_cache=True).logits[:, -1:, :].argmax(dim=-1).item())
                emitted.append(pending)
                history.append(pending)
                pending = nxt
                stats["fallback_cycles"] += 1
                if observer is not None:
                    observer(cache)
                continue
            prefix_len = cache.get_seq_length()
            has_eos = bool(candidates and candidates[-1] == eos)
            inputs = [pending] + (candidates[:-1] if has_eos else candidates)
            tensor = torch.tensor([inputs], device=device, dtype=prompt_ids.dtype)
            schedule.use("patched")
            preds = model(tensor, past_key_values=cache.cache, use_cache=True).logits[0].argmax(dim=-1).tolist()
            stats["block_cycles"] += 1
            emitted.append(pending)
            history.append(pending)
            rejected_at = None
            accepted = 0
            for pos, cand in enumerate(candidates):
                if preds[pos] != cand:
                    rejected_at = pos
                    break
                emitted.append(cand)
                history.append(cand)
                accepted += 1
                if cand == eos:
                    break
            stats["accepted_draft_tokens"] += accepted
            if rejected_at is not None:
                cache.crop(prefix_len + rejected_at + 1)
                pending = preds[rejected_at]
            elif has_eos and accepted == len(candidates):
                break
            else:
                pending = preds[-1]
            if emitted[-1] == eos:
                break
            if observer is not None and len(emitted) < max_new_tokens:
                observer(cache)
    return emitted[:max_new_tokens], stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--timing-rounds", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    methods = ("original_bnb_greedy", "activationvec_no_fallback", "activationvec_fallback")
    per_prompt = []
    for pid in args.prompt_ids:
        prompt_ids = tokenizer(rows[pid]["prompt"], return_tensors="pt")["input_ids"].to(device)
        capacity = int(prompt_ids.shape[-1]) + args.max_new_tokens + 8
        common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens, device=device, capacity=capacity)
        rounds = []
        for r in range(args.timing_rounds):
            order = list(methods[r % 3:] + methods[:r % 3])
            sample = {"round": r, "order": order, "methods": {}}
            for m in order:
                if m == "original_bnb_greedy":
                    tokens, _, _, total_ms = run_vanilla_timed(model, tokenizer, prompt_ids, args.max_new_tokens, device)
                    sample["methods"][m] = {"tokens": tokens, "total_s": total_ms / 1000}
                else:
                    tokens, st = generate(model, tokenizer, prompt_ids, fallback_on_empty=(m == "activationvec_fallback"), **common)
                    sample["methods"][m] = {"tokens": tokens, **st}
            ref = sample["methods"]["original_bnb_greedy"]["tokens"]
            sample["no_fallback_matches_bnb"] = sample["methods"]["activationvec_no_fallback"]["tokens"] == ref
            sample["fallback_matches_bnb"] = sample["methods"]["activationvec_fallback"]["tokens"] == ref
            rounds.append(sample)
        med = {m: statistics.median(s["methods"][m]["total_s"] for s in rounds) for m in methods}
        fb = rounds[0]["methods"]["activationvec_fallback"]
        per_prompt.append({
            "prompt_id": pid, "prompt_length": int(prompt_ids.shape[-1]), "rounds": rounds,
            "median_s": med,
            "speedup_no_fallback_vs_bnb": med["original_bnb_greedy"] / med["activationvec_no_fallback"],
            "speedup_fallback_vs_bnb": med["original_bnb_greedy"] / med["activationvec_fallback"],
            "acceptance": fb["accepted_draft_tokens"] / fb["draft_tokens"] if fb["draft_tokens"] else None,
            "fallback_cycles": fb["fallback_cycles"], "block_cycles": fb["block_cycles"],
            "all_match_bnb": all(s["no_fallback_matches_bnb"] and s["fallback_matches_bnb"] for s in rounds),
        })
        print(json.dumps({k: per_prompt[-1][k] for k in ("prompt_id", "all_match_bnb", "acceptance",
              "speedup_no_fallback_vs_bnb", "speedup_fallback_vs_bnb", "fallback_cycles", "block_cycles")}))

    def geomean(key):
        return statistics.geometric_mean(p[key] for p in per_prompt)

    result = {
        "schema": "zassd-bnb-fused-nf4-activationvec-fallback-closed-loop-v1",
        "classification": "development-only B2 closed-loop with empty-proposal fallback",
        "model": MODEL, "revision": REVISION, "benchmark": args.benchmark,
        "max_new_tokens": args.max_new_tokens, "timing_rounds": args.timing_rounds,
        "per_prompt": per_prompt,
        "all_runs_match_bnb": all(p["all_match_bnb"] for p in per_prompt),
        "geomean_speedup_no_fallback_vs_bnb": geomean("speedup_no_fallback_vs_bnb"),
        "geomean_speedup_fallback_vs_bnb": geomean("speedup_fallback_vs_bnb"),
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                    "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["Development prompts only; GSM8K held-out indices 300-349 excluded.",
                         "Identity is checked on token IDs against original BNB greedy, not per-step KV.",
                         "The block schedule is entered once per generation (setup included in timing); plain/patched forwards are swapped between fallback and block cycles."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("all_runs_match_bnb", "geomean_speedup_no_fallback_vs_bnb",
                                             "geomean_speedup_fallback_vs_bnb")}, indent=2))
    if not result["all_runs_match_bnb"]:
        raise RuntimeError("Activationvec output diverged from original BNB greedy")


if __name__ == "__main__":
    main()
