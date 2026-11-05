#!/usr/bin/env python3
"""Offline answer scoring for wp6 outputs (same extractor for both modes): boxed > 'answer is' > 'Answer:' line > first conclusion paragraph."""
import json, re, sys
from collections import Counter

def num(s):
    n = re.findall(r"-?\d[\d,]*\.?\d*", s or "")
    return n[-1].replace(",", "").rstrip(".") if n else None

def extract(t):
    m = re.search(r"\\boxed\{([^}]*)\}", t)
    if m: return num(m.group(1))
    m = re.search(r"answer is[:\s]*([^\n]*)", t, re.I)
    if m and num(m.group(1)): return num(m.group(1).split(".")[0] if re.search(r"\d", m.group(1).split(".")[0]) else m.group(1))
    m = re.search(r"\nAnswer:[ \t]*([^\n]*)", t)
    if m and num(m.group(1)): return num(m.group(1))
    m = re.search(r"(?:^|\n)(?:Therefore|Thus|Hence|So,)[^\n]*", t)
    if m: return num(m.group(0))
    return None

d = json.load(open(sys.argv[1])); P = d["per_prompt"]
rows = []
for p in P:
    a, b = extract(p["text_stock"]), extract(p["text_inv"])
    rows.append((p["id"], p["gold"], a, b))
n = len(rows); ok_s = [r[2] == r[1] and r[2] is not None for r in rows]; ok_i = [r[3] == r[1] and r[3] is not None for r in rows]
none_s = sum(r[2] is None for r in rows); none_i = sum(r[3] is None for r in rows)
both = [r for r in rows if r[2] is not None and r[3] is not None]
c = Counter((x, y) for x, y in zip(ok_s, ok_i))
out = {"prompts": n, "acc_stock": sum(ok_s) / n, "acc_invariant": sum(ok_i) / n, "no_answer_stock": none_s, "no_answer_inv": none_i,
       "same_answer_where_both_present": sum(r[2] == r[3] for r in both), "both_present": len(both),
       "stock_right_inv_wrong": c[(True, False)], "stock_wrong_inv_right": c[(False, True)], "both_right": c[(True, True)], "both_wrong": c[(False, False)],
       "identical_sequences": d["identical_sequences"]}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1].replace(".json", "_scored.json"), "w"), indent=1)
