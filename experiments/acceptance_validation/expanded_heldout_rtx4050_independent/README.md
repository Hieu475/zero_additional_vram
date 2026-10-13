# Independent expanded acceptance validation

## Protocol

Collected on the RTX 4050 Laptop GPU using Qwen2.5-3B-Instruct at the pinned
revision in `config.json`, NF4 weights, FP16 compute, SDPA, and a 16-token
generation budget. For each prompt the runner measured vanilla and six fixed
layer-skip actions (CKA 60/67/90, K=2/4) in shuffled order. Prompt splits use a
stable hash of task and prompt ID. Before splitting, the runner excluded all
18 prompts present in the previous acceptance pilot (12 calibration and 6
CNN/DM held-out prompts). Dataset hashes, excluded IDs, software versions,
GPU telemetry, and the execution commit are recorded in `config.json` and
`environment.json`.

The available snapshot has 50 prompts per task. After excluding prior pilot
prompts, the realized split was 44 train, 17 development, 27 in-domain
held-out, and 44 cross-task CNN/DM held-out. This remains smaller than the
initially proposed 100/40/50 sizes. Confidence intervals resample prompts,
not speculative cycles.

## Acceptance prediction

Predictor selection used only complete-cycle TPC MAE on development prompts.
Per-action survival was selected (0.813 tokens/cycle MAE; combined
action/context/entropy scored 0.822). On held-out data:

| Held-out split | Prompts | Per-action TPC MAE (95% prompt CI) | Survival Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Same-task GSM8K/HumanEval | 27 | 0.852 (0.802–0.900) | 0.185 | 0.042 |
| Cross-task CNN/DM | 44 | 0.842 (0.811–0.875) | 0.188 | 0.080 |

Combined had slightly better Brier/ECE (0.182/0.045 in-domain and
0.186/0.073 cross-task); its TPC MAE was nearly identical (0.850 in-domain,
0.845 cross-task). Context-only,
entropy-only, and online EMA predictors were worse for TPC MAE. This supports
per-action survival as a compact held-out predictor, but does not show that
additional features or online adaptation improve it.

## Feasibility and performance

The frozen cost-model policy performed poorly on both held-out splits. At the
1.03 measured-speedup threshold it predicted 22/174 in-domain and 37/264
cross-task prompt-action pairs profitable. Precision was 4.5% and 5.4%,
recall was 14.3% and 22.2%, and false-positive rate was 13.5% and 13.7%,
respectively. At 1.05, precision remained 4.5%/5.4%, recall was 25.0%/25.0%,
and false-positive rate was 13.3%/13.7%. The predictor therefore does not
provide a reliable runtime feasibility decision yet.

Measured layer-skip decode throughput remained below vanilla for every
action. Mean paired speedup ranged from 0.621 to 0.926 in-domain and 0.638 to
0.910 cross-task. Exact sequence match against vanilla was 23/27 for each
action in-domain and ranged from 37/44 to 44/44 cross-task, depending on K and action;
these results do not support a lossless claim. The benchmark establishes
acceptance-model behavior, not a layer-skip speedup.

## Reproduction

```bash
python scripts/run_expanded_acceptance_validation.py \
  --output-dir experiments/acceptance_validation/expanded_heldout_rtx4050_independent \
  --max-new-tokens 16 --seed 4171

python scripts/evaluate_expanded_acceptance.py \
  --run-dir experiments/acceptance_validation/expanded_heldout_rtx4050_independent
```

The per-split `raw_measurements.jsonl` and `acceptance_cycles/*.jsonl` are the
source records. `evaluation.json` contains all candidate predictor results,
prompt-cluster intervals, reliability statistics, and decision confusion
matrices. The plot is in `figures/reliability.png`. The runner refuses to
overwrite a non-empty output directory.
