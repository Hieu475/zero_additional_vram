# Held-out speculative roofline validation

## Protocol

The action cost model in `src/zassd/profiling/action_cost_model.py` was frozen.
Its draft-cost, verify-cost, acceptance, and cache coefficients were not fitted
on these measurements. Vanilla decode throughput was measured on calibration
prompts from GSM8K and HumanEval, separately at each tested context length. The
test workload uses disjoint CNN/DailyMail prompt IDs.

The main run uses Qwen2.5-3B-Instruct NF4 with FP16 compute and SDPA on the RTX
4050 Laptop GPU. It tests CKA-60/67/90 with K=2/4, 32 generated tokens, and
context lengths 128, 512, 1,024, and 2,048. Six CNN/DailyMail prompts are tested
at each context length (144 action/context/prompt rows), paired with vanilla.
Calibration uses six prompts each from GSM8K and HumanEval at each length.
Measurements include draft, verify, cache, remaining decode time, acceptance,
tokens per cycle, throughput, VRAM, and before/after GPU telemetry.

Long contexts are formed by repeating source prompt token IDs, while shorter
contexts are truncated. This is a controlled synthetic context sweep, not a
natural long-document evaluation. The separate `run_rtx4050_frozen_b2bfb01`
uses original prompt lengths and 12 CNN/DailyMail held-out prompts; it provides
a natural-prompt check with 72 action/prompt pairs.

## Held-out results

| Error / decision metric | Held-out result |
|---|---:|
| Draft latency MAE / MAPE | 3.16 ms / 5.94% |
| Verify latency MAE / MAPE | 3.62 ms / 9.64% |
| Total cycle latency MAE / MAPE | 3.10 ms / 3.57% |
| Acceptance rate MAE / MAPE | 0.143 / 21.37% |
| Tokens per cycle MAE / MAPE | 0.374 / 12.57% |
| Decode throughput MAE / MAPE | 3.82 tokens/s / 12.14% |
| Mean paired speedup, prompt-cluster bootstrap 95% CI | 0.809× (0.788–0.830) |

The predictor classified every tested action as slower than vanilla. It was
correct on 131/144 action/context/prompt rows and missed all 13 measured faster
rows. This gives 90.97% raw accuracy but 50% balanced accuracy, zero recall for
profitable actions, and zero false positives. The raw accuracy is therefore a
class-imbalance result, not evidence that the model finds profitable cases.

| Context tokens | Mean paired speedup | Measured faster rows |
|---:|---:|---:|
| 128 | 0.721× | 1/36 |
| 512 | 0.825× | 5/36 |
| 1,024 | 0.843× | 3/36 |
| 2,048 | 0.840× | 4/36 |

Across the tested actions and contexts, layer-skip speculation was slower on
average. The model predicts cycle cost fairly closely, but its acceptance
prediction is too inaccurate to identify the few profitable cases. This
supports a negative result for the current predictor/action set; the six
held-out prompts per context are still an exploratory sample, not a final
systems claim.

## Artifacts

- `context_sweep_rtx4050_bb0fdb9/heldout/raw_measurements.jsonl`: per-action
  held-out raw observations, paired vanilla throughput, token IDs, and GPU
  telemetry.
- `context_sweep_rtx4050_bb0fdb9/calibration/`: per-context vanilla calibration
  observations and baseline rates.
- `context_sweep_rtx4050_bb0fdb9/prediction_errors.json`: overall and
  per-context errors, bootstrap intervals, and feasibility confusion matrices.
- `context_sweep_rtx4050_bb0fdb9/figures/`: predicted-versus-measured
  throughput and held-out feasibility map.
- `run_rtx4050_frozen_b2bfb01/`: natural prompt-length run and accompanying
  measurements.

## Reproduction

```bash
python scripts/run_roofline_context_sweep.py \
  --samples-per-task 6 --max-new-tokens 32 \
  --context-lengths 128 512 1024 2048 \
  --output-dir experiments/roofline_validation/reproduction_context_sweep

python scripts/analyze_roofline_prediction.py \
  --run-dir experiments/roofline_validation/reproduction_context_sweep
```

The static KV cache capacity can now be set above its historical 2,048-token
default through `kv_cache_max_capacity`; existing callers retain the 2,048
default. The measured run used commit `bb0fdb9`, following the separate cache
capacity change at `77608b1`.
