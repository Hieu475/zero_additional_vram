# Independent short-context PLD validation

## Data and protocol

This experiment uses 60 new natural-document prompts from the pinned
LongBench snapshot at exactly 384 input tokens. Before sampling, it excluded
source IDs present in the earlier natural-context pilot and both previous
80-prompt sessions. Source IDs, tokenized input IDs, snapshot hash and
prepared-input hash are recorded in each run. No source document was repeated
to construct context length.

Each of the first three process sessions measured the same 60 prompts in a
different randomized method order. Methods were vanilla autoregressive,
vanilla through the speculative core at K=0, PLD forced-miss K=4, and
prompt-plus-generated PLD K=2/4/6/8. Each raw row stores prefill, decode and
end-to-end latency, token IDs, cycle statistics and GPU telemetry. The runs
were collected in the same operational period and are not a controlled
cold-versus-hot thermal experiment. Session temperatures ranged from 64–88°C;
SM clocks ranged from 1950–2550 MHz.

## Paired results

| Method | Mean decode speedup over 3 sessions | Session-level 95% t CI | Mean end-to-end speedup | Exact match |
| --- | ---: | ---: | ---: | ---: |
| Vanilla via speculative core K=0 | 0.969x | 0.936–1.002 | 0.974x | 60/60 |
| PLD forced-miss K=4 | 0.968x | 0.948–0.988 | 0.973x | 60/60 |
| PLD K=2 | 1.109x | 1.058–1.160 | 1.086x | 44/60 |
| PLD K=4 | 1.078x | 1.054–1.101 | 1.058x | 44/60 |
| PLD K=6 | 1.100x | 1.040–1.159 | 1.073x | 45/60 |
| PLD K=8 | 1.116x | 1.101–1.131 | 1.085x | 44/60 |

The intervals above use three session means as the independent units (t
interval with 2 degrees of freedom). They do not treat the 180 repeated prompt
measurements as 180 independent observations. The core and forced-miss
controls were exact but about 3% slower than direct vanilla on average,
exposing framework/lookup overhead. The K=4 lookup-cycle hit rate averaged
19.6%, with 0.212 accepted candidate tokens per verification cycle.

PLD K=4 cleared break-even in all three process sessions, but its exactness
result did not improve: the same 16 prompts mismatched vanilla in every
session, with the same first-divergence position. K=2 and K=8 had 16 repeated
mismatches; K=6 had 15. This is deterministic for these prompts and runtime,
but does not identify whether numerical kernels or cache behavior cause the
divergence.

## Sequential verification trade-off

The separate `sequential_verify_rtx4050_retry1/` run compared vanilla,
batched PLD K=4 and sequential PLD K=4 on the same 60 prompts. Sequential
verification matched vanilla on 60/60 prompts, at 0.965x decode speedup
(prompt-bootstrap 95% CI 0.955–0.975) and 0.972x end-to-end speedup (0.964–
0.980). Batched K=4 in that same process session measured 1.043x decode
speedup (1.002–1.096) with 44/60 exact outputs. Sequential verification is a
correctness recovery path here, but it removes the speedup.

The sequential path advances target KV one token at a time and stops at the
first rejected candidate, so a rejected token is never committed. Its tests
cover all-accepted candidates and forced rejection at positions zero and one.
This establishes token identity on the tested path; the experiment does not
claim bitwise-equal cache tensors across different batched implementations.

## Interpretation

There is a reproducible short-context PLD speedup under the current NF4/FP16/
SDPA setup, but it is paired with substantial sequence divergence. The
sequential path restores exact token IDs at a latency cost greater than the
observed gain. Report these as separate performance and correctness results;
they do not support a lossless acceleration claim. The three sessions are
repeated process runs, not independently scheduled thermal-control sessions.

## Reproduction

```bash
python scripts/run_independent_pld_short_context.py \
  --output-dir experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050 \
  --per-bucket 60 --max-new-tokens 32 --seed 20261013 --order-seed 20261014

python scripts/run_independent_pld_short_context.py \
  --output-dir experiments/natural_context_validation/independent_pld_short_context/session2_rtx4050 \
  --prepared-inputs experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl \
  --per-bucket 60 --max-new-tokens 32 --seed 20261013 --order-seed 20261015

python scripts/run_independent_pld_short_context.py \
  --output-dir experiments/natural_context_validation/independent_pld_short_context/session3_rtx4050 \
  --prepared-inputs experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl \
  --per-bucket 60 --max-new-tokens 32 --seed 20261013 --order-seed 20261016

python scripts/analyze_independent_pld_sessions.py \
  --session experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050 \
  --session experiments/natural_context_validation/independent_pld_short_context/session2_rtx4050 \
  --session experiments/natural_context_validation/independent_pld_short_context/session3_rtx4050 \
  --output experiments/natural_context_validation/independent_pld_short_context/session_analysis.json

python scripts/run_independent_pld_short_context.py \
  --output-dir experiments/natural_context_validation/independent_pld_short_context/sequential_verify_rtx4050_retry1 \
  --prepared-inputs experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl \
  --per-bucket 60 --max-new-tokens 32 --seed 20261013 --order-seed 20261017 --sequential-only
```

The first sequential-verification launch stopped after two prompts due to an
early-rejection indexing bug. Those partial rows are not used; the corrected
run is the `*_retry1` directory. The tests are in
`tests/test_prompt_lookup_sequential_verify.py`.
