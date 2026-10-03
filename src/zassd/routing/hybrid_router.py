"""Hybrid Zero-VRAM Draft Router (Direction 1).

Problem: the current codebase (speculative.py draft_mode="hybrid") only does naive
fallback: if PLD finds a candidate use PLD, otherwise use layer-skip.
That is not routing but opportunistic fallback — it does not predict beforehand
which source yields higher utility, ignoring entropy / cost / acceptance history.

Contributions of this file (targeting top-tier MLSys/EMNLP):
  1. Cost-aware routing: estimate E[TPS] for each source before running draft,
     pick argmax utility instead of "use PLD if available".
  2. PLD quality signal: distinguish long hits (n=3, full K) vs weak hits (n=2, K=1)
     — weak hits have low actual alpha and often lose to layer-skip.
  3. Entropy-conditioned prior: high H -> prefer deeper layer-skip (cka_90);
     low H -> prefer PLD (repetitive text, templates, code).
  4. Online EMA tracking of alpha_pld / alpha_ls separately for self-calibration.

Theory references:
  - PLD: Saxena (2023) Prompt Lookup Decoding; He et al. (2023) REST;
    Yan et al. (2025, NAACL) "Decoding Speculative Decoding" (draft/verify cost analysis).
  - Adaptive draft: Mamou et al. (2024) Dynamic Speculation Lookahead;
    BanditSpec (Hou et al. 2025, ICML) — treat draft selection as online learning.

All logic is pure-Python / CPU, testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from zassd.decoding.ngram import find_candidate_tokens


@dataclass
class RouterDecision:
    """Routing result for one inference cycle."""
    source: str  # "pld" | "layer_skip" | "vanilla"
    candidates: list[int] = field(default_factory=list)
    ngram_match_len: int = 0  # 3 = strong hit, 2 = weak hit, 0 = miss
    expected_tps_pld: float = 0.0
    expected_tps_ls: float = 0.0
    reason: str = ""


@dataclass
class RouterStats:
    """Accumulated stats for ablation in the paper."""
    total_cycles: int = 0
    pld_cycles: int = 0
    layer_skip_cycles: int = 0
    vanilla_fallbacks: int = 0
    pld_hits: int = 0
    pld_strong_hits: int = 0
    # EMA acceptance per source
    alpha_pld_ema: float = 0.85
    alpha_ls_ema: float = 0.70

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["pld_share"] = self.pld_cycles / max(1, self.total_cycles)
        return d


class HybridDraftRouter:
    """Cost-aware router between PLD (c~0) and layer-skip (c~0.5-0.8).

    Utility formula (simple, explainable — suitable for systems papers):
        E[N|source] = 1 + alpha_source * K_eff
        E[TPS|source] = E[N] / (T_draft(source) + T_verify(K_eff))
    Pick the source with the largest E[TPS], adjusted with an entropy prior.
    """

    # PLD draft is nearly free: just a CPU list-scan, measured via micro-benchmark
    PLD_DRAFT_MS = 0.08
    # Vanilla fallback when both sources look bad (e.g. PLD miss + extremely high entropy)
    ENTROPY_PANIC_THRESHOLD = 4.5

    def __init__(
        self,
        cost_model: Optional[Any] = None,
        ngram_size: int = 3,
        entropy_low: float = 0.5,
        entropy_high: float = 2.0,
        pld_k: int | None = None,
        ema_beta: float = 0.15,
        pld_prior_alpha: float = 0.85,
        ls_prior_alpha: float = 0.60,
        prefer_pld_margin: float = 1.25,
        enable_vanilla_skip: bool = True,
        enable_weak_hit_cap: bool = True,
        enable_horizon_tuning: bool = True,
    ) -> None:
        """Args (ablation switches, all True in the full router):
            prefer_pld_margin: 1.25 = asymmetric PLD preference; 1.0 = none.
            enable_vanilla_skip: False = never skip speculation (no panic
                vanilla, no LS-vs-vanilla opportunity-cost skip).
            enable_weak_hit_cap: False = weak bigram hits keep full horizon.
            enable_horizon_tuning: False = no 4->2 PLD horizon truncation.
        """
        self.cost_model = cost_model
        self.ngram_size = ngram_size
        self.entropy_low = entropy_low
        self.entropy_high = entropy_high
        self.ema_beta = ema_beta
        self.prefer_pld_margin = prefer_pld_margin
        self.enable_vanilla_skip = enable_vanilla_skip
        self.enable_weak_hit_cap = enable_weak_hit_cap
        self.enable_horizon_tuning = enable_horizon_tuning
        self.pld_k = pld_k
        self.stats = RouterStats(
            alpha_pld_ema=pld_prior_alpha, alpha_ls_ema=ls_prior_alpha
        )

    # -- internal estimators -------------------------------------------------
    def _verify_ms(self, k: int) -> float:
        if self.cost_model is not None and hasattr(self.cost_model, "predict_verify_ms"):
            try:
                return float(self.cost_model.predict_verify_ms(k))
            except Exception:
                pass
        # fallback: gamma_0=23.3 (Qwen) + 3.5ms/extra token (measured in cost_model)
        return 23.3 + max(0, k - 1) * 3.5

    def _ls_draft_ms(self, config_name: str, k: int) -> float:
        if self.cost_model is not None and hasattr(self.cost_model, "predict_draft_ms"):
            try:
                return float(self.cost_model.predict_draft_ms(config_name, k))
            except Exception:
                pass
        return 19.7 * k  # fallback Qwen cka_75 K=1

    def _ls_alpha(self, config_name: str, k: int, entropy: float) -> float:
        if self.cost_model is not None and hasattr(self.cost_model, "predict_acceptance_rate"):
            try:
                return float(
                    self.cost_model.predict_acceptance_rate(config_name, k, entropy=entropy)
                )
            except Exception:
                pass
        return self.stats.alpha_ls_ema

    # -- main API ---------------------------------------------------------
    def _vanilla_tps(self) -> float:
        try:
            return float(self.cost_model.baseline_tps)
        except Exception:
            return 41.5

    def probe_pld(
        self, full_history: list[int], curr_tok: int, step_k: int
    ) -> tuple[list[int], int]:
        """Probe PLD at no GPU cost. Returns (candidates, match_len)."""
        tokens = full_history + [curr_tok]
        cands = find_candidate_tokens(
            tokens, ngram_size=self.ngram_size, max_candidates=step_k
        )
        if not cands:
            return [], 0
        # determine strong vs weak match by re-checking whether n=3 hits
        strong = find_candidate_tokens(tokens, ngram_size=3, max_candidates=1)
        mlen = 3 if strong else 2
        return cands, mlen

    def select_source(
        self,
        full_history: list[int],
        curr_tok: int,
        step_k: int,
        entropy: float,
        config_name: str = "cka_75",
        pld_k: int | None = None,
    ) -> RouterDecision:
        """Pick the best draft source for the current cycle (runs on CPU before draft).

        pld_k: max candidates for PLD (free draft so it may exceed layer-skip's
            step_k; None = use step_k). The router compares E[TPS] fairly
            using each source's actual K_eff.
        """
        pld_k = step_k if pld_k is None else max(step_k, pld_k)
        cands, mlen = self.probe_pld(full_history, curr_tok, pld_k)
        if self.enable_weak_hit_cap and mlen <= 2 and len(cands) > step_k:
            # Weak (bigram) hit: weak evidence -> clamp horizon to step_k.
            # PLD draft is free but long verify is wasteful when alpha is low.
            cands = cands[:step_k]

        # Extremely uncertain text -> go straight to 1-token vanilla, avoid wasting a cycle
        if self.enable_vanilla_skip and entropy >= self.ENTROPY_PANIC_THRESHOLD and not cands:
            return RouterDecision(
                source="vanilla", candidates=[], ngram_match_len=0,
                reason=f"panic-entropy H={entropy:.2f}, skip speculation",
            )

        # --- PLD branch ---
        tps_pld = 0.0
        if cands:
            k_eff = len(cands)
            # weak (bigram) hits get discounted alpha; strong hits keep EMA; low entropy adds bonus
            alpha_pld = self.stats.alpha_pld_ema
            if mlen <= 2:
                alpha_pld *= 0.75
            if entropy < self.entropy_low:
                alpha_pld = min(0.98, alpha_pld + 0.05)
            elif entropy > self.entropy_high:
                alpha_pld *= 0.90
            t_cycle = self.PLD_DRAFT_MS + self._verify_ms(k_eff)
            tps_pld = (1.0 + alpha_pld * k_eff) / (t_cycle / 1000.0)
            # Pick the optimal horizon for PLD: long K only pays off when alpha is high.
            # Compare E[TPS] of full horizon vs truncated at 2 (cheaper verify).
            if self.enable_horizon_tuning and k_eff > 2:
                _k2 = 2
                _t2 = self.PLD_DRAFT_MS + self._verify_ms(_k2)
                _tps2 = (1.0 + alpha_pld * _k2) / (_t2 / 1000.0)
                if _tps2 > tps_pld:
                    cands = cands[:_k2]
                    k_eff = _k2
                    tps_pld = _tps2

        # --- layer-skip branch ---
        alpha_ls = self._ls_alpha(config_name, step_k, entropy)
        # correct with observed EMA (simple BanditSpec-style online learning)
        alpha_ls = 0.5 * alpha_ls + 0.5 * self.stats.alpha_ls_ema
        t_cycle_ls = self._ls_draft_ms(config_name, step_k) + self._verify_ms(step_k)
        tps_ls = (1.0 + alpha_ls * step_k) / (t_cycle_ls / 1000.0)

        # --- argmax decision with margin preferring PLD (0 VRAM + 0 draft energy) ---
        if cands and tps_pld * self.prefer_pld_margin >= tps_ls:
            reason = (
                f"PLD win: tps_pld={tps_pld:.1f} vs tps_ls={tps_ls:.1f} "
                f"(match={mlen}, K={len(cands)}, H={entropy:.2f})"
            )
            return RouterDecision(
                source="pld", candidates=cands, ngram_match_len=mlen,
                expected_tps_pld=tps_pld, expected_tps_ls=tps_ls, reason=reason,
            )
        if not self.enable_vanilla_skip and tps_ls < self._vanilla_tps():
            # Ablation (no feasibility gate): never skip; fall through to the
            # argmax between the available speculative sources instead.
            if cands and tps_pld >= tps_ls:
                return RouterDecision(
                    source="pld", candidates=cands, ngram_match_len=mlen,
                    expected_tps_pld=tps_pld, expected_tps_ls=tps_ls,
                    reason=f"no-gate argmax PLD (tps_pld={tps_pld:.1f})",
                )
            reason = f"no-gate fallback layer_skip (tps_ls={tps_ls:.1f}, H={entropy:.2f})"
            return RouterDecision(
                source="layer_skip", candidates=[], ngram_match_len=mlen,
                expected_tps_pld=tps_pld, expected_tps_ls=tps_ls, reason=reason,
            )
        if tps_ls < self._vanilla_tps():
            return RouterDecision(
                source="vanilla", candidates=[], ngram_match_len=0,
                expected_tps_pld=tps_pld, expected_tps_ls=tps_ls,
                reason=f"PLD miss + LS weak (tps_ls={tps_ls:.1f}) -> vanilla skip",
            )
        if not cands:
            reason = f"PLD miss -> layer_skip (tps_ls={tps_ls:.1f}, H={entropy:.2f})"
            return RouterDecision(
                source="layer_skip", candidates=[], ngram_match_len=0,
                expected_tps_pld=tps_pld, expected_tps_ls=tps_ls, reason=reason,
            )
        reason = (
            f"LS win: tps_ls={tps_ls:.1f} vs tps_pld={tps_pld:.1f} "
            f"(weak-match={mlen}, H={entropy:.2f})"
        )
        return RouterDecision(
            source="layer_skip", candidates=[], ngram_match_len=mlen,
            expected_tps_pld=tps_pld, expected_tps_ls=tps_ls, reason=reason,
        )

    def update(
        self, source: str, accepted: int, proposed: int, ngram_match_len: int = 0
    ) -> None:
        """Update EMA after each cycle (called from the generate loop)."""
        self.stats.total_cycles += 1
        if proposed <= 0:
            self.stats.vanilla_fallbacks += 1
            return
        obs_alpha = accepted / max(1, proposed)
        b = self.ema_beta
        if source == "pld":
            self.stats.pld_cycles += 1
            self.stats.alpha_pld_ema = (1 - b) * self.stats.alpha_pld_ema + b * obs_alpha
            self.stats.pld_hits += 1
            if ngram_match_len >= 3:
                self.stats.pld_strong_hits += 1
        elif source == "layer_skip":
            self.stats.layer_skip_cycles += 1
            self.stats.alpha_ls_ema = (1 - b) * self.stats.alpha_ls_ema + b * obs_alpha
