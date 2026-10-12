# Natural document context pilot

This is an exploratory RTX 4050 pilot that checks whether the current layer
skip cost model and measurements behave similarly on naturally occurring
document contexts. The source is the official [THUDM/LongBench repository](https://github.com/THUDM/LongBench).
The runner pins the Hugging Face archive revision and checks its SHA-256 before
selecting examples. It stores selected token IDs, source task/record IDs and
measurement rows; it does not copy the source documents into the repository.

The pilot has 12 distinct examples, three at each exact input length: 384,
768, 1536 and 3072 Qwen2.5 tokens. The examples come from NarrativeQA,
Qasper, GovReport, MultiNews and QMSum. Each input is a natural document prefix
with a fixed task/question suffix. It contains no repeated or synthetic text.
For every prompt, vanilla and six CKA/K actions ran in seeded random order.
There was one measured execution per prompt/action and 16 maximum generated
tokens. The roofline coefficients and baseline throughput were frozen from the
separate RTX 4050 roofline calibration; this pilot was not used to fit them.

## Results

| Action | Mean speedup vs paired vanilla | Prompt bootstrap 95% CI | Exact sequences |
| --- | ---: | ---: | ---: |
| CKA60, K=2 | 0.808x | 0.734–0.889 | 9/12 (75.0%) |
| CKA67, K=2 | 0.808x | 0.728–0.883 | 9/12 (75.0%) |
| CKA90, K=2 | 0.834x | 0.785–0.879 | 9/12 (75.0%) |
| CKA60, K=4 | 0.556x | 0.483–0.646 | 8/12 (66.7%) |
| CKA67, K=4 | 0.580x | 0.525–0.627 | 8/12 (66.7%) |
| CKA90, K=4 | 0.695x | 0.640–0.749 | 8/12 (66.7%) |

The frozen predictor classified every action as unprofitable. Three individual
K=2 observations were measured above 1.0x (two CKA60 and one CKA67), but none
of the paired prompt-bootstrap confidence intervals suggests an average win.
These single observations are not evidence of a stable profitable region.
Every K=4 action was slower than vanilla in this pilot. At 3072 input tokens,
only one of the three prompts had sequence identity for each action; all three
prompts at 1536 tokens matched exactly. The pilot cannot attribute these token
mismatches to NF4 arithmetic, batching, or cache state because it records
sequences but not per-step logits/KV digests.

Cycle decomposition is in the raw rows. The summary reports decode throughput
prediction error and measured draft/verify/cache time by action. The results
are not a replacement for the synthetic repeated-context sweep: they show a
natural-document workload in which layer skipping did not beat vanilla under
this configuration.

## Reproduction

```bash
python scripts/run_natural_context_validation.py \
  --output-dir experiments/natural_context_validation/my_run \
  --per-bucket 3 --max-new-tokens 16 --seed 2026

python scripts/analyze_natural_context_validation.py \
  --raw experiments/natural_context_validation/my_run/raw_measurements.jsonl \
  --output experiments/natural_context_validation/my_run/summary.json
```

The run directory includes `config.json`, `environment.json`,
`prepared_inputs.jsonl`, `raw_measurements.jsonl`, and `summary.json`. The pilot
is small and has one measured run per prompt/action; its bootstrap interval
captures prompt variation only, not session or thermal variation. Expand to at
least 20 examples per context bucket and include a paired PLD baseline before
using this result as a paper claim.
