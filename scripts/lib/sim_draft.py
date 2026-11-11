#!/usr/bin/env python3
"""Offline simulation of exact speculative decoding over recorded reference sequences.
speedup = N*c(1) / sum_blocks c(rows). c(M) = measured block cost (results/kernels/wp5_block_cost.json); M>=7 EXTRAPOLATED (assumption).
Drafters return a list of candidate chains; chains are merged into a prefix trie, rows = 1 + trie nodes."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

COST = {1: 15.0, 2: 15.6, 3: 17.7, 4: 17.8, 5: 23.6, 6: 23.7, 7: 29.5, 8: 29.6, 9: 35.4, 10: 35.5}  # 7+ extrapolated
def cost(m): return COST.get(m, COST[10] + 3 * (m - 10))

def trie_nodes(chains):
    s = set()
    for c in chains:
        for i in range(1, len(c) + 1): s.add(tuple(c[:i]))
    return len(s)

def simulate(docs, drafter, n=128, overhead=0.0):
    tot_c = tot_tok = blocks = steps = 0; tot_acc = 0
    for d in docs:
        prompt, ref = d["prompt"], d["ref"][:n]
        p = 0
        while p < len(ref) - 1:
            hist = prompt + ref[:p + 1]
            chains = [c for c in drafter(hist, d) if c]
            rem = ref[p + 1:]
            if not chains:
                tot_c += cost(1) + overhead; p += 1; steps += 1; continue
            rows = 1 + trie_nodes(chains)
            acc = 0
            for c in chains:
                a = 0
                while a < len(c) and a < len(rem) - 1 + 0 and c[a] == rem[a]: a += 1
                acc = max(acc, a)
            acc = min(acc, len(rem) - 1) if len(rem) else 0
            tot_c += cost(rows) + overhead; steps += 1; p += 1 + acc; tot_acc += acc; blocks += 1
        tot_tok += len(ref)
    return {"speedup": tot_tok * cost(1) / tot_c, "tok_per_block": tot_tok / max(1, steps), "blocks": blocks, "acc": tot_acc}

def ngram_matches(tokens, query, k):
    n = len(query); out = []
    for i in range(len(tokens) - n):
        if tokens[i:i + n] == query:
            t = tokens[i + n:i + n + k]
            if t: out.append(t)
    return out

def pld(k, ng=3):
    def f(hist, d):
        for n in range(min(ng, len(hist)), 1, -1):
            m = ngram_matches(hist[:-1], hist[-n:], k)   # exclude the trailing query itself
            if m: return [m[-1]]
        return []
    return f

def pld_tree(k, width, ng=3):
    """distinct continuations from different matches (most recent first), up to `width` chains"""
    def f(hist, d):
        for n in range(min(ng, len(hist)), 1, -1):
            m = ngram_matches(hist[:-1], hist[-n:], k)
            if m:
                seen, out = set(), []
                for c in reversed(m):
                    if tuple(c) not in seen: seen.add(tuple(c)); out.append(c)
                    if len(out) == width: break
                return out
        return []
    return f

def corpus_drafter(docs, k, width=1, ng=(4, 3, 2)):
    idx = defaultdict(list)
    for j, d in enumerate(docs):
        seq = d["prompt"] + d["ref"]
        for n in ng:
            for i in range(len(seq) - n):
                idx[tuple(seq[i:i + n])].append((j, tuple(seq[i + n:i + n + k])))
    def f(hist, d):
        j0 = d["_j"]
        for n in ng:
            if len(hist) < n: continue
            c = Counter(t for j, t in idx.get(tuple(hist[-n:]), []) if j != j0)
            if c: return [list(t) for t, _ in c.most_common(width)]
        return []
    return f

def combine(*fs):
    def f(hist, d):
        out = []
        for g in fs: out += g(hist, d)
        return out
    return f

if __name__ == "__main__":
    for path in sys.argv[1:]:
        docs = json.load(open(path))
        for j, d in enumerate(docs): d["_j"] = j
        print("==", Path(path).name, len(docs), "docs")
        for name, dr in [("PLD K=2 (current)", pld(2)), ("PLD K=3", pld(3)), ("PLD K=5", pld(5)),
                         ("PLD-tree K=3 w=2", pld_tree(3, 2)), ("PLD-tree K=3 w=3", pld_tree(3, 3)),
                         ("corpus K=3 w=1", corpus_drafter(docs, 3)), ("PLD+corpus K=3", combine(pld(3), corpus_drafter(docs, 3))),
                         ("PLD+corpus K=3 w=2", combine(pld(3), corpus_drafter(docs, 3, 2)))]:
            r = simulate(docs, dr); print(f"{name:22s} speedup {r['speedup']:.3f}  tok/block {r['tok_per_block']:.2f}")
