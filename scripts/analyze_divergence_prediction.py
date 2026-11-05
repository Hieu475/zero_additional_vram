#!/usr/bin/env python3
"""B: can the share of diverged sequences be predicted from (i) the per-row logit-gap noise measured once on GSM8K and (ii) the
decision margins observed on each data set?

Model: a block row that decides a committed token flips it when its top-1/top-2 gap shifts by less than -margin. With the empirical
noise sample s_k (gap shift of block vs sequential logits, measured by `ops`), the flip probability of a decision with margin m is
q(m) = mean_k [ s_k < -m ]. For a prompt with decisions m_1..m_D the divergence probability is 1 - prod(1 - q(m_j)).
Baseline predictor ignoring margins: flip rate p = (flipped rows / rows) from the GSM8K audit, divergence = 1 - (1-p)^D.
Compares predicted and observed shares per (config, data set) cell; no fitting except the single noise sample per config.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

P = Path("experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles")
CONFIGS = ["qwen05b", "qwen15b", "qwen3b", "olmo2_1b", "phi35", "torchao_qwen3b"]
DATASETS = ["gsm8k", "humaneval", "cnndm", "ext150"]


def load_noise(cfg):
    d = json.loads((P / f"b_noise_{cfg}_m3_gsm8k20_v1.json").read_text())
    pr = d["per_row"]
    shift = np.array(pr["gap_shift"]); gap = np.array(pr["gap_seq"])
    flips = int(np.sum(shift < -gap))  # rows where the gap changed sign
    return shift, flips / len(shift), d


def predict_prompt(margins, shift_sorted):
    if not margins:
        return 0.0
    m = np.asarray(margins)
    # q(m) = fraction of noise samples with shift < -m  (sorted search)
    q = np.searchsorted(shift_sorted, -m, side="left") / len(shift_sorted)
    return float(1 - np.exp(np.sum(np.log1p(-np.clip(q, 0, 1 - 1e-12)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=P / "b_divergence_prediction_v1.json")
    a = ap.parse_args()
    cells = []
    for cfg in CONFIGS:
        shift, p_flip, _ = load_noise(cfg)
        ss = np.sort(shift)
        for ds in DATASETS:
            f = P / f"b_margins_{cfg}_{ds}_naive_k2_v1.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            per = d["per_prompt"]
            observed = d["diverged"] / d["prompts"]
            pred_margin = float(np.mean([predict_prompt(p["decision_margins"], ss) for p in per]))
            n_dec = [len(p["decision_margins"]) for p in per]
            pred_flat = float(np.mean([1 - (1 - p_flip) ** n for n in n_dec]))
            cells.append({"config": cfg, "dataset": ds, "prompts": d["prompts"], "observed": observed,
                          "pred_margin_model": pred_margin, "pred_flat_rate": pred_flat,
                          "mean_decisions": float(np.mean(n_dec)), "audit_flip_rate": p_flip})
    obs = np.array([c["observed"] for c in cells]); pm = np.array([c["pred_margin_model"] for c in cells]); pf = np.array([c["pred_flat_rate"] for c in cells])

    def stats(pred):
        ss_res = float(np.sum((obs - pred) ** 2)); ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        slope = float(np.sum(obs * pred) / np.sum(pred * pred)) if np.sum(pred * pred) else float("nan")  # through the origin
        r = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 2 else float("nan")
        return {"R2_vs_identity": 1 - ss_res / ss_tot if ss_tot else float("nan"), "pearson_r": r,
                "slope_obs_on_pred": slope, "mean_abs_error_pp": float(100 * np.mean(np.abs(obs - pred)))}
    out = {"schema": "zassd-divergence-prediction-v1", "cells": cells, "n_cells": len(cells),
           "margin_model": stats(pm), "flat_rate_model": stats(pf)}
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{'config':16s}{'data':10s}{'obs':>7s}{'margin':>9s}{'flat':>8s}")
    for c in cells:
        print(f"{c['config']:16s}{c['dataset']:10s}{c['observed']:7.2f}{c['pred_margin_model']:9.2f}{c['pred_flat_rate']:8.2f}")
    print("margin model:", json.dumps(out["margin_model"]))
    print("flat-rate model:", json.dumps(out["flat_rate_model"]))


if __name__ == "__main__":
    main()
