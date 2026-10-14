from scripts.summarize_paired_router_evaluation import choose_best_fixed, summarize_method


def row(cohort, role, method, tps, exact=True):
    return {"cohort": cohort, "selection_role": role,
            "measurement": {"decode_tps": tps, "sequence_exact_match": exact,
                            "total_time_s": 1.0, "verification_time_s": .5,
                            "paired_speedup_vs_vanilla": tps / 10,
                            "slowdown_vs_vanilla": tps < 10,
                            "peak_vram_mb": 2000, "acceptance_rate": .5,
                            "first_divergence_position": None}}


def test_best_fixed_uses_development_records_only():
    pairs = {
        "dev": {"direct_vanilla": row("diagnostic", "development", "direct", 10),
                "pld_k2": row("diagnostic", "development", "pld_k2", 11),
                "pld_k4": row("diagnostic", "development", "pld_k4", 12),
                "pld_k8": row("diagnostic", "development", "pld_k8", 9),
                "cka90_k2": row("diagnostic", "development", "cka", 8)},
        "heldout": {"direct_vanilla": row("heldout", "heldout_evaluation", "direct", 10),
                    "pld_k2": row("heldout", "heldout_evaluation", "pld_k2", 100),
                    "pld_k4": row("heldout", "heldout_evaluation", "pld_k4", 1),
                    "pld_k8": row("heldout", "heldout_evaluation", "pld_k8", 9),
                    "cka90_k2": row("heldout", "heldout_evaluation", "cka", 8)},
    }

    best, means = choose_best_fixed(pairs)

    assert best == "pld_k4"
    assert means["pld_k2"] == 11


def test_summary_reports_exactness_slowdown_and_oracle_regret():
    pairs = {"h1": {"direct_vanilla": row("heldout", "heldout_evaluation", "direct", 10),
                    "pld_k2": row("heldout", "heldout_evaluation", "p2", 12),
                    "pld_k4": row("heldout", "heldout_evaluation", "p4", 14),
                    "pld_k8": row("heldout", "heldout_evaluation", "p8", 13),
                    "cka90_k2": row("heldout", "heldout_evaluation", "cka", 8),
                    "candidate": row("heldout", "heldout_evaluation", "candidate", 11, exact=False)}}

    result = summarize_method(pairs, "candidate", "heldout", 1)

    assert result["sequence_exact_match_rate"] == 0
    assert result["slowdown_frequency"] == 0
    assert result["mean_regret_tps_vs_offline_best_fixed"] == 3
