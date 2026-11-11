"""Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python3 scripts/sim_adaptive_block.py   (features cached by drafter_eval_sim.py --cache results/drafter/feats_3b_<set>.pt)
CPU-only policy simulation: fixed M=4 hybrid vs confidence-gated variable block size, Qwen2.5-3B refs (GSM8K, HumanEval).
Cost model: scripts/sim_draft.py COST table (measured block costs) + drafter chain cost CD every step (it must run every step to
produce next drafts) + host-sync overhead DELTA for adaptive policies (needs a per-step host read to pick the graph).
Drafter numerics: float32 on CPU (deployed: bf16) -> small difference, fine for a policy estimate."""
import json, glob, os, sys
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, "scripts")
import sim_draft as SD
from safetensors import safe_open
from zassd.kernels.drafter import Drafter

CD, K = 2.0, 3
ck = torch.load("results/drafter/d2_k5.pt", map_location="cpu")
vid = ck["state"]["vocab_ids"]; dr = Drafter(ck["d"] if "d" in ck else 2048, ck.get("ff", 4096), vid, ck["state"]["head"]); dr.load_state_dict(ck["state"]); dr.eval()
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*"))[0]
for f in sorted(glob.glob(snap + "/*.safetensors")):
    with safe_open(f, "pt") as sf:
        if "model.embed_tokens.weight" in sf.keys(): emb = sf.get_tensor("model.embed_tokens.weight").float()

def chains_for(doc, feats):
    P, R = doc["prompt"], doc["ref"]; n = len(R)
    idx = torch.arange(len(P) - 1, len(P) + n - 1)
    s = feats[idx].float(); cur = torch.tensor(R); toks, confs = [], []
    with torch.no_grad():
        for k in range(K):
            s = dr.step(s, emb[cur]); lg = dr.logits(s); pr = torch.softmax(lg, -1); c, a = pr.max(-1); cur = vid[a]
            toks.append(cur); confs.append(c)
    return torch.stack(toks, 1).tolist(), torch.stack(confs, 1).tolist()

def pld_tail(hist, ng=3, k=3):
    for n in range(min(ng, len(hist)), 1, -1):
        m = SD.ngram_matches(hist[:-1], hist[-n:], k)
        if m: return m[-1]
    return []

def run(docs, feats_list, policy, tau=0.0, delta=0.0, n=128, cpld=1.0):
    tot_c = tot_tok = steps = 0
    for d, feats in zip(docs, feats_list):
        prompt, ref = d["prompt"], d["ref"][:n]; T, C = d["_t"], d["_c"]; p = 0
        while p < len(ref) - 1:
            pl = pld_tail(prompt + ref[:p + 1])
            drafts = [pl[j] if j < len(pl) else T[p][j] for j in range(K)]
            conf = [cpld if j < len(pl) else C[p][j] for j in range(K)]
            rem = ref[p + 1:]
            full = 0
            while full < K and full < len(rem) - 1 and drafts[full] == rem[full]: full += 1   # accepted drafts if all K were sent
            if policy == "fixed": j = K
            elif policy == "oracle": j = 0 if full == 0 else K
            else:
                j = 0
                while j < K and conf[j] >= tau: j += 1
                if j == 2: j = 3                       # M=3 and M=4 cost the same
            acc = min(full, j)
            tot_c += SD.cost(1 + j) + CD + (delta if policy != "fixed" else 0.0)
            steps += 1; p += 1 + acc
        tot_tok += len(ref)
    return tot_tok * SD.cost(1) / tot_c, tot_tok / steps

for name in ("gsm8k", "humaneval"):
    docs = json.load(open(f"results/kernels/ref/qwen3b_{name}100.json")); feats = torch.load(f"results/drafter/feats_3b_{name}.pt")
    for d, f in zip(docs, feats): d["_t"], d["_c"] = chains_for(d, f)
    print(f"== {name} (3B, simulated; cost = block + {CD} ms drafter/step)")
    sp, tps = run(docs, feats, "fixed"); print(f"  fixed M=4 hybrid         speedup {sp:.3f}  tok/step {tps:.2f}")
    sp, tps = run(docs, feats, "oracle", delta=0.0); print(f"  oracle (M=1 iff all rejected, no sync cost) {sp:.3f}  tok/step {tps:.2f}   <- upper bound for any gating")
    for delta in (0.0, 0.3):
        for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            sp, tps = run(docs, feats, "adaptive", tau=tau, delta=delta); print(f"  adaptive tau={tau:.1f} delta={delta:.1f}ms  speedup {sp:.3f}  tok/step {tps:.2f}")
