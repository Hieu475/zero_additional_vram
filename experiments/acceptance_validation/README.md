# Acceptance survival validation

The roofline runner now writes cycle-level acceptance traces in
`acceptance_cycles/{calibration,heldout}.jsonl`. Each row records prompt/action,
configured K, context length, draft and accepted counts, emitted count, draft / verify /
cache latency, pre-cycle entropy, and pre-cycle top-1 logit margin. Cycle tracing is
enabled by this validation runner; other generation calls keep the default off.

Run a validation pass and analyze it with:

```bash
python scripts/run_heldout_roofline_validation.py \
  --samples-per-task 12 --max-new-tokens 32 \
  --output-dir experiments/acceptance_validation/run_<tag>
python scripts/collect_acceptance_cycles.py \
  --run-dir experiments/acceptance_validation/run_<tag>
python scripts/evaluate_acceptance_predictor.py \
  --run-dir experiments/acceptance_validation/run_<tag>
```

The evaluator hashes calibration prompt IDs into train/development groups, selects
among global, per-action, context, entropy, and combined empirical predictors on the
development group, refits on calibration, then evaluates on the held-out CNN/DM
prompts. It reports token-per-cycle MAE with prompt-cluster bootstrap intervals and
feasibility confusion matrices at speedup thresholds 1.00, 1.03, and 1.05. Online EMA
is evaluated causally and resets for each prompt/action/K sequence.

## RTX 4050 pilot

Run directory: `pilot_rtx4050_committed/`. Model and runtime metadata are in
`environment.json`; raw prompt-level outcomes and cycle traces are preserved in the
run directory. This pilot used 6 prompts per calibration task and 6 held-out CNN/DM
prompts (12 calibration prompts total), with 32 generated tokens and six layer-skip
actions. It is exploratory, not a final held-out claim.

| Predictor | Development TPC MAE | Held-out TPC MAE | 95% prompt bootstrap CI |
|---|---:|---:|---:|
| Global mean acceptance | 1.072 | 1.114 | [1.029, 1.216] |
| Global survival by position | 0.952 | 0.980 | [0.926, 1.049] |
| Per action (selected) | 0.758 | 0.833 | [0.763, 0.927] |
| Context conditioned | 0.952 | 0.980 | [0.926, 1.049] |
| Entropy conditioned | 0.954 | 0.976 | [0.930, 1.033] |
| Combined | 0.793 | 0.864 | [0.785, 0.976] |
| Online EMA | — | 0.925 | [0.862, 1.007] |

At the 1.00 threshold, measured-latency diagnostic feasibility had TP=0, FP=1,
TN=34, FN=1. At 1.03 it had TP=0, FP=0, TN=35, FN=1; at 1.05 it had TP=0, FP=0,
TN=36, FN=0. Thus the selected
predictor improved acceptance/TPC error over the global mean but did not identify
profitable cases in this small run. The feasibility calculation intentionally uses
measured cycle latency and prompt-matched vanilla throughput to isolate acceptance
prediction; it is not a deployable runtime prediction. The pilot has only six held-out
prompt clusters, and EOS / token-budget truncation can make realized emissions lower
than `1 + sum_i P(A >= i)`.

For survival probabilities themselves, the selected per-action predictor had MAE
0.376 and Brier score 0.199, compared with 0.481 / 0.291 for global mean acceptance.

The first attempt, `pilot_rtx4050/`, stopped early when the no-router vanilla fallback
called a router observer. That guard is fixed in the decoder; use the retry directory
for analysis.
