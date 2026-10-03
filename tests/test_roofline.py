"""Tests for SpeculativeRoofline (Direction 2) — CPU-only."""
import sys
sys.path.insert(0, "src")

from zassd.modeling.roofline import SpeculativeRoofline, KNOWN_GPUS


def test_k2_matches_measured_4050():
    # Measured: CKA fixed K=2 on Qwen ~0.87-0.88x
    r = SpeculativeRoofline()
    p = r.predict(KNOWN_GPUS["RTX4050-laptop"], c=0.75, alpha=0.73, k=2)
    assert 0.75 < p.speedup < 1.05, f"speedup={p.speedup}"
    assert p.alpha_breakeven > 0.7  # needs high alpha to win at K=2


def test_pld_wins_everywhere():
    r = SpeculativeRoofline()
    for gname in ["RTX4050-laptop", "RTX4090", "A100-40GB"]:
        p = r.predict(KNOWN_GPUS[gname], c=0.01, alpha=0.9, k=3)
        assert p.wins, f"{gname} should win with PLD-like draft, got {p.speedup}"


def test_k1_sweet_spot_beats_k2_on_4050():
    r = SpeculativeRoofline()
    g = KNOWN_GPUS["RTX4050-laptop"]
    p1 = r.predict(g, c=0.83, alpha=0.86, k=1)
    p2 = r.predict(g, c=0.75, alpha=0.73, k=2)
    assert p1.speedup > p2.speedup


def test_wide_gpu_unlocks_larger_k():
    r = SpeculativeRoofline()
    p4050 = r.predict(KNOWN_GPUS["RTX4050-laptop"], c=0.75, alpha=0.73, k=4)
    pA100 = r.predict(KNOWN_GPUS["A100-40GB"], c=0.75, alpha=0.73, k=4)
    assert pA100.speedup > p4050.speedup
