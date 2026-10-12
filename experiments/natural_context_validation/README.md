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

## 3072-token NF4 exactness trace

`pilot_rtx4050/exactness_trace_3072.json` and
`exactness_summary_3072.json` trace the three longest prompts under the same
Qwen revision, NF4 weights, FP16 compute, SDPA backend, static cache and CKA90/K2
action. Vanilla was repeated once per prompt as a determinism control. All
three vanilla repetitions matched their references. Batched verification
matched vanilla on 1/3 prompts; sequential target verification matched 3/3.

The active KV digest first differed from vanilla after the first speculative
cycle for NarrativeQA, after cycle start 3 for Qasper, and after cycle start 2
for QMSum. Cache lengths still agreed at those points. The Qasper sequence first
diverged at generated token 6, where the vanilla top-1 margin was zero; QMSum
first diverged at token 15 with a 0.125 margin. Sequential verification had no
cache-prefix digest divergence in these three traces and reproduced the vanilla
sequences. This is prompt-specific evidence that batched verification changes
the numerical KV path even before an output token necessarily changes. The
trace does not isolate whether the cache drift comes from batch-shaped kernels
or another implementation detail, and three prompts do not establish a general
NF4 guarantee. Its diagnostic timings include cache hashing and must not be
used as a performance comparison.

## Expanded paired run with PLD

`full_n20_pld_rtx4050/` contains 80 natural-document prompts, 20 at each exact
input length (384, 768, 1536 and 3072 tokens). It has 640 raw rows: one paired
vanilla, PLD K=4 (`prompt+generated` candidates) and six layer-skip actions per
prompt, all using NF4, FP16 compute, SDPA and the static KV backend. Method order
was randomized within each prompt. The cost-model coefficients remained frozen.

PLD averaged **1.031x** over all 80 prompts (prompt-bootstrap 95% CI
**0.985–1.081**) with **91.2%** sequence identity. This is near break-even, not
an overall speedup claim. The short-context bucket (384-token inputs) is a
promising stratum: **1.129x** (95% CI **1.034–1.234**), but sequence identity
was 90%. At 3072 tokens PLD was **0.967x** (95% CI **0.911–1.025**) with 95%
sequence identity. The observed PLD lookup hit rate was 22.3% overall and 19.5%
at 384 tokens; it accepted 0.22 draft tokens per verification cycle overall.

All layer-skip actions were slower on average. CKA90/K2 was 0.847x
(95% CI 0.828–0.867) with 80% sequence identity; CKA90/K4 was 0.732x
(95% CI 0.707–0.757), also with 80% identity. Frozen cost-model predictions
marked every layer-skip action unprofitable. Individual single-run observations
above 1x occurred for K2 actions, but they did not change the paired mean
conclusion.

The run was one session, not a thermal-controlled study. Telemetry snapshots
ranged from 61–88°C and 1935–2550 MHz SM clock; buckets were traversed in order,
although method order was randomized within each prompt. The prompt bootstrap
does not account for session or thermal variation. Treat the short-context PLD
result as a candidate regime for a repeated, thermally controlled validation.
The 91.2% sequence identity also means performance must be reported with the
correctness trade-off; this does not support a lossless-output claim.

## Independent-order repeat

`repeat2_n20_pld_rtx4050/` repeats the same 80 prompt IDs and methods, keeps
sample seed 2027, and uses method-order seed 2028. It has a separate 640-row
raw file, environment record and summary. `session_comparison_rtx4050.json`
contains paired prompt-level session deltas and bootstrap intervals; reproduce
it with `scripts/analyze_natural_context_sessions.py`.

| Action | First session mean | Repeat mean | Repeat prompt bootstrap 95% CI | Exact match, both sessions |
| --- | ---: | ---: | ---: | ---: |
| PLD K=4 | 1.031x | 1.043x | 0.996–1.093 | 91.25% |
| CKA90 K=2 | 0.847x | 0.855x | 0.836–0.875 | 80.0% |
| CKA90 K=4 | 0.732x | 0.744x | 0.717–0.772 | 80.0% |

For PLD, the repeat-minus-first paired mean change was +0.012x (prompt
bootstrap 95% CI -0.002–0.027), and per-prompt speedups were strongly correlated
across sessions (r=0.957). At 384 input tokens PLD remained above break-even in
both individual sessions (1.129x and 1.120x); its exact-match rate was 90% in
both. This is a repeated candidate regime, not a validated lossless speedup.
At the other lengths, paired session changes were small and their confidence
intervals included zero.

The repeat does not establish thermal control: temperature reached 88°C in both
sessions, SM clock ranged about 1920–2550 MHz, and telemetry recorded GPU
utilization from 57% to 100% during the first session and 53% to 100% in the
repeat. Initial snapshots were cooler (53°C and 54°C), but system load varied.
The intervals above resample prompts, not independent sessions; there are only
two sessions on the same RTX 4050. Thus this strengthens prompt-level
repeatability but does not show thermal robustness or cross-device
generalization.

Reproduce the repeat with:

```bash
python scripts/run_natural_context_validation.py \
  --output-dir experiments/natural_context_validation/repeat2_n20_pld_rtx4050 \
  --per-bucket 20 --max-new-tokens 16 --seed 2027 --order-seed 2028

python scripts/analyze_natural_context_sessions.py \
  --session1 experiments/natural_context_validation/full_n20_pld_rtx4050/raw_measurements.jsonl \
  --session2 experiments/natural_context_validation/repeat2_n20_pld_rtx4050/raw_measurements.jsonl \
  --output experiments/natural_context_validation/session_comparison_rtx4050.json
```

Reproduction:

```bash
python scripts/run_natural_context_validation.py \
  --output-dir experiments/natural_context_validation/my_n20_run \
  --per-bucket 20 --max-new-tokens 16 --seed 2027

python scripts/analyze_natural_context_validation.py \
  --raw experiments/natural_context_validation/my_n20_run/raw_measurements.jsonl \
  --output experiments/natural_context_validation/my_n20_run/summary.json \
  --seed 2027

python scripts/plot_natural_context_validation.py \
  --summary experiments/natural_context_validation/my_n20_run/summary.json \
  --output experiments/natural_context_validation/my_n20_run/figures/natural_context_summary.pdf
```

Reproduce the diagnostic with:

```bash
python scripts/trace_natural_context_exactness.py \
  --inputs experiments/natural_context_validation/pilot_rtx4050/prepared_inputs.jsonl \
  --prompt-tokens 3072 --max-new-tokens 16 \
  --output experiments/natural_context_validation/my_run/exactness_trace_3072.json

python scripts/summarize_natural_context_exactness.py \
  --trace experiments/natural_context_validation/my_run/exactness_trace_3072.json \
  --output experiments/natural_context_validation/my_run/exactness_summary_3072.json
```
