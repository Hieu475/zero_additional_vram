"""Direction 1 — Hybrid router simulation (CPU-only, analytic, honest).

Compares 4 strategies on 4 workload archetypes (timings from roofline + cost model):
  pure-LS, pure-PLD (miss -> vanilla), naive-hybrid (PLD on hit, LS on every miss),
  routed (cost-aware: on weak hits compare E[TPS] PLD vs LS; on miss compare LS vs vanilla).

Key scientific insight: average speed over TIME, not over cycles.
Adding an LS cycle (42.7 tok/s) into a PLD average (57 tok/s) DRAGS THE AVERAGE DOWN
even when LS > vanilla. The router only wins by (a) skipping bad LS cycles
(pick vanilla under high entropy), (b) lowering K on weak hits, (c) picking config by entropy.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from zassd.modeling.roofline import SpeculativeRoofline, KNOWN_GPUS

OUT = Path("experiments/16_hybrid_roofline")
OUT.mkdir(parents=True, exist_ok=True)

roof = SpeculativeRoofline()
g = KNOWN_GPUS["RTX4050-laptop"]
TV = g.t_vanilla_ms
TPS_V = 1000.0 / TV  # 41.5 tok/s

T_VERIFY = lambda k, rho=1.0: roof.scale(roof.t_verify_k1 + (k - 1) * roof.verify_slope * rho, g.bandwidth_gbs)
T_LS = lambda k, c=0.75: roof.scale(roof.t_draft_k1 * (c / 0.75) * k, g.bandwidth_gbs)

# (name, h, a_pld, a_ls_base, frac_high_entropy, description)
WORKLOADS = [
    ("generic-chat", 0.12, 0.95, 0.86, 0.15, "generic chat prompt (fit from 1.39x)"),
    ("summarization", 0.35, 0.95, 0.80, 0.10, "CNN/DM: highly repetitive n-grams"),
    ("reasoning", 0.07, 0.85, 0.75, 0.40, "GSM8K: high entropy, many hard tokens"),
    ("code", 0.25, 0.90, 0.82, 0.20, "HumanEval: repeated templates + novel logic"),
]

WEAK_FRAC = 0.30  # 30% of PLD hits are weak bigram hits
K_PLD, K_LS = 4, 1
results = []
for name, h, a_pld, a_ls, f_high, desc in WORKLOADS:
    # -- pure LS (K=1, alpha base) --
    e_ls, t_ls = 1 + a_ls * K_LS, T_LS(K_LS) + T_VERIFY(K_LS) + roof.t_misc
    sp_ls = (e_ls / t_ls) * TV
    # -- pure PLD (hit -> K=4, miss -> vanilla) --
    e_pld = h * (1 + a_pld * K_PLD) + (1 - h) * 1.0
    t_pld = h * (0.08 + T_VERIFY(K_PLD) + roof.t_misc) + (1 - h) * TV
    sp_pld = (e_pld / t_pld) * TV
    # -- naive hybrid (hit -> PLD K=4 even on weak hits; miss -> LS) --
    a_ls_blended = (1 - f_high) * a_ls + f_high * 0.35  # naive does not condition on entropy
    t_nv_miss = t_ls
    e_nv_miss = 1 + a_ls_blended * K_LS
    e_pld_w_naive = 1 + (a_pld * 0.75) * K_PLD  # weak hit: discounted alpha (honest)
    e_nv = h * ((1 - WEAK_FRAC) * (1 + a_pld * K_PLD) + WEAK_FRAC * e_pld_w_naive) + (1 - h) * e_nv_miss
    t_nv = h * (0.08 + T_VERIFY(K_PLD) + roof.t_misc) + (1 - h) * t_nv_miss
    sp_nv = (e_nv / t_nv) * TV
    # weak hit: PLD K=1 with alpha*0.75 vs LS K=1 -> pick max TPS
    e_pld_w = 1 + (a_pld * 0.75) * K_PLD
    t_pld_w = 0.08 + T_VERIFY(K_PLD) + roof.t_misc
    tps_pld_w = e_pld_w / t_pld_w
    tps_ls = e_ls / t_ls
    use_pld_w = tps_pld_w >= tps_ls
    e_w = e_pld_w if use_pld_w else e_ls
    t_w = t_pld_w if use_pld_w else t_ls
    # miss: high-entropy fraction (alpha_ls=0.35 -> LS loses to vanilla -> pick vanilla)
    e_ls_bad, t_ls_bad = 1 + 0.35 * K_LS, t_ls
    miss_e = (1 - f_high) * e_ls + f_high * 1.0
    miss_t = (1 - f_high) * t_ls + f_high * TV
    e_rt = h * (1 - WEAK_FRAC) * (1 + a_pld * K_PLD) + h * WEAK_FRAC * e_w + (1 - h) * miss_e
    t_rt = h * (1 - WEAK_FRAC) * (0.08 + T_VERIFY(K_PLD) + roof.t_misc) + h * WEAK_FRAC * t_w + (1 - h) * miss_t
    sp_rt = (e_rt / t_rt) * TV
    row = {"workload": name, "desc": desc, "pure_ls": round(sp_ls, 3),
           "pure_pld": round(sp_pld, 3), "naive_hybrid": round(sp_nv, 3),
           "routed": round(sp_rt, 3),
           "routed_vs_naive": round(sp_rt - sp_nv, 3),
           "routed_vs_best_single": round(sp_rt - max(sp_ls, sp_pld), 3),
           "weak_hit_uses": "pld_K1" if use_pld_w else "layer_skip"}
    results.append(row)
    print(f"[{name:13s}] LS={sp_ls:.3f} PLD={sp_pld:.3f} naive={sp_nv:.3f} "
          f"ROUTED={sp_rt:.3f} (vs naive {row['routed_vs_naive']:+.3f}, "
          f"vs best-single {row['routed_vs_best_single']:+.3f})")

with open(OUT / "hybrid_router_simulation.json", "w") as f:
    json.dump(results, f, indent=2)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r["workload"] for r in results]
    x = list(range(len(labels)))
    w = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, key in enumerate(["pure_ls", "pure_pld", "naive_hybrid", "routed"]):
        ax.bar([xi + (i - 1.5) * w for xi in x], [r[key] for r in results], w, label=key)
    ax.axhline(1.0, color="red", ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Expected speedup vs vanilla")
    ax.set_title("Hybrid router projection (RTX 4050, analytic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "hybrid_router_projection.png", dpi=150)
    print("[FIG] saved hybrid_router_projection.png")
except Exception as e:
    print(f"[FIG] skip ({e})")
print(f"DONE -> {OUT/'hybrid_router_simulation.json'}")
