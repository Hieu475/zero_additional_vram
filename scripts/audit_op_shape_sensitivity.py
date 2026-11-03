#!/usr/bin/env python3
"""Per-operator shape sensitivity of the unpatched HF + bitsandbytes path (development only).

For teacher-forced three-token blocks on the exact sequential state, run the standard model on the block, capture each
operator's real inputs/outputs, and compare with the same operator applied one row at a time on the same inputs.
Also compares block logits with sequentially produced logits (bitwise, argmax flips, margins).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION


def is_rmsnorm(m) -> bool:
    return "rmsnorm" in type(m).__name__.lower()


def is_quant_linear(m) -> bool:
    return isinstance(m, Linear4bit) or type(m).__module__.startswith("gemlite")


def op_class(name: str, module) -> str:
    last = name.split(".")[-1]
    if name == "lm_head":
        return "lm_head"
    if is_rmsnorm(module):
        if name == "model.norm":
            return "rmsnorm_final"
        return "rmsnorm_qk" if last in ("q_norm", "k_norm") else "rmsnorm_layer"
    if is_quant_linear(module):
        return f"linear4bit_{last}"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--block-size", type=int, default=3, help="rows M per verification block")
    ap.add_argument("--context-len", type=int, default=0,
                    help="if >0, left-pad the prompt with other benchmark prompts to ~this many tokens (keeps the prompt at the end)")
    ap.add_argument("--backend", default="hf_bnb_nf4", choices=["hf_bnb_nf4", "hqq_gemlite_w4a16"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend(args.backend, args.model, args.revision, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    M = args.block_size
    filler_ids = []
    if args.context_len > 0:
        for r in rows.values():
            filler_ids.extend(tok(r["prompt"])["input_ids"])
            if len(filler_ids) >= args.context_len:
                break
    targets = {name: m for name, m in model.named_modules()
               if is_quant_linear(m) or is_rmsnorm(m) or name == "lm_head"}
    stats = defaultdict(lambda: {"calls": 0, "calls_with_mismatch": 0, "elements": 0, "mismatches": 0, "max_abs_diff": 0.0})
    e2e = {"rows": 0, "logits_not_bitwise": 0, "argmax_flips": 0, "max_abs_logit_diff": [], "margins_of_flips": [], "margins_all": []}
    sdpa_stats = stats["sdpa_attention"]
    with torch.inference_mode():
        for pid in args.prompt_ids:
            ids = tok(rows[pid]["prompt"], return_tensors="pt")["input_ids"]
            if args.context_len > 0 and ids.shape[-1] < args.context_len:
                need = args.context_len - ids.shape[-1]
                ids = torch.cat([torch.tensor([filler_ids[:need]]), ids], dim=-1)
            ids = ids.to(device)
            cap = int(ids.shape[-1]) + args.steps + 16
            cache = TargetKVCache(backend="static", max_capacity=cap)
            logits = model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
            greedy = []
            ref_cache = TargetKVCache(backend="static", max_capacity=cap)
            lg = model(ids, past_key_values=ref_cache.cache, use_cache=True).logits[:, -1:, :]
            for _ in range(args.steps):
                t = lg.argmax(dim=-1)
                greedy.append(int(t.item()))
                lg = model(t, past_key_values=ref_cache.cache, use_cache=True).logits[:, -1:, :]
            del ref_cache
            consumed = 0
            while consumed + M <= args.steps:
                block = torch.tensor([greedy[consumed:consumed + M]], device=device)
                captured, sdpa_calls, hooks = {}, [], []

                def make_hook(name):
                    def hook(mod, inputs, output):
                        x = inputs[0]
                        if x.ndim == 3 and x.shape[1] == M:
                            captured[name] = (x.detach().clone(), output.detach().clone())
                    return hook

                for name, m in targets.items():
                    hooks.append(m.register_forward_hook(make_hook(name)))
                original_sdpa = F.scaled_dot_product_attention

                def spy(q, k, v, *a, **kw):
                    out = original_sdpa(q, k, v, *a, **kw)
                    if q.shape[-2] == M:
                        sdpa_calls.append((q.detach().clone(), k, v, dict(kw), out.detach().clone()))
                    return out

                F.scaled_dot_product_attention = spy
                try:
                    start_len = cache.get_seq_length()
                    block_logits = model(block, past_key_values=cache.cache, use_cache=True).logits
                finally:
                    F.scaled_dot_product_attention = original_sdpa
                    for h in hooks:
                        h.remove()
                cache.crop(start_len)
                for name, (x, out3) in captured.items():
                    m = targets[name]
                    per_row = torch.cat([m(x[:, i:i + 1, :].contiguous()) for i in range(M)], dim=1)
                    s = stats[op_class(name, m)]
                    neq = int((per_row != out3).sum().item())
                    s["calls"] += 1; s["calls_with_mismatch"] += int(neq > 0); s["elements"] += out3.numel(); s["mismatches"] += neq
                    if neq:
                        s["max_abs_diff"] = max(s["max_abs_diff"], float((per_row.float() - out3.float()).abs().max().item()))
                for q, k, v, kw, out3 in sdpa_calls:
                    prefix = k.shape[-2] - M
                    row_outs = []
                    for i in range(M):
                        length = prefix + i + 1
                        row_outs.append(original_sdpa(q[:, :, i:i + 1, :].contiguous(), k[:, :, :length, :].contiguous(),
                                                      v[:, :, :length, :].contiguous(), attn_mask=None, dropout_p=0.0,
                                                      scale=kw.get("scale"), is_causal=False, enable_gqa=True))
                    per_row = torch.cat(row_outs, dim=2)
                    neq = int((per_row != out3).sum().item())
                    sdpa_stats["calls"] += 1; sdpa_stats["calls_with_mismatch"] += int(neq > 0)
                    sdpa_stats["elements"] += out3.numel(); sdpa_stats["mismatches"] += neq
                    if neq:
                        sdpa_stats["max_abs_diff"] = max(sdpa_stats["max_abs_diff"], float((per_row.float() - out3.float()).abs().max().item()))
                seq_rows = []
                for i in range(M):
                    seq_rows.append(model(block[:, i:i + 1], past_key_values=cache.cache, use_cache=True).logits[:, -1, :])
                seq_logits = torch.stack(seq_rows, dim=1)
                for i in range(M):
                    a, b = block_logits[0, i].float(), seq_logits[0, i].float()
                    e2e["rows"] += 1
                    e2e["logits_not_bitwise"] += int(not torch.equal(block_logits[0, i], seq_logits[0, i]))
                    e2e["max_abs_logit_diff"].append(float((a - b).abs().max().item()))
                    top2 = torch.topk(b, 2).values
                    margin = float((top2[0] - top2[1]).item())
                    e2e["margins_all"].append(margin)
                    if int(a.argmax()) != int(b.argmax()):
                        e2e["argmax_flips"] += 1
                        e2e["margins_of_flips"].append(margin)
                consumed += M
            print(json.dumps({"prompt": pid, "rows": e2e["rows"], "flips": e2e["argmax_flips"]}), flush=True)
    diffs = e2e.pop("max_abs_logit_diff"); margins = e2e.pop("margins_all")
    e2e["median_max_abs_logit_diff"] = sorted(diffs)[len(diffs) // 2]
    e2e["max_max_abs_logit_diff"] = max(diffs)
    e2e["fraction_rows_margin_below_median_diff"] = sum(m < e2e["median_max_abs_logit_diff"] for m in margins) / len(margins)
    out = {"schema": "zassd-op-shape-sensitivity-v1", "backend": args.backend, "model": args.model, "revision": args.revision, "prompts": args.prompt_ids, "steps": args.steps, "block_size": M, "context_len": args.context_len,
           "per_op_class": {k: {**v, "mismatch_rate": v["mismatches"] / max(1, v["elements"]),
                                "call_mismatch_rate": v["calls_with_mismatch"] / max(1, v["calls"])} for k, v in sorted(stats.items())},
           "end_to_end": e2e}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out["per_op_class"].items():
        print(f"{k:24s} calls={v['calls']:5d} elem_mismatch={v['mismatch_rate']:.4f} call_mismatch={v['call_mismatch_rate']:.3f} max_abs={v['max_abs_diff']:.3g}")
    print(json.dumps(out["end_to_end"]))


if __name__ == "__main__":
    main()
