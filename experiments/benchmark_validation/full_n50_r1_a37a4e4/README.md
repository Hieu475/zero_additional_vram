# RTX 4050 full paired benchmark (N=50/task)

## Protocol

- Model: Qwen/Qwen2.5-3B-Instruct, NF4, float16 compute, SDPA.
- Checkpoint revision, runtime versions, dataset SHA256 values, driver, GPU clock,
  power snapshots, temperature, and per-run token IDs are recorded in this folder.
- 50 prompt IDs each from GSM8K, HumanEval, and CNN/DM; 48 output tokens maximum.
- Two warmup prompts per task, one repeat, and three rotated order seeds (42, 43,
  44): 150 unique prompts, 450 paired prompt/order groups, 2,700 raw method rows.
- The first generated token comes from prefill logits. `decode_tokens` excludes it;
  `ttft_ms` is prefill time until prompt logits are available. Decode duration
  includes post-prefill token selection and later decoding. End-to-end TPS includes
  prefill.
- CIs resample prompts. Repeats and order seeds are averaged within each prompt.
  Controls are not used to adjust/correct the measured speedups.

## Paired results

| Method | End-to-end speedup vs vanilla (95% CI) | Decode TPS ratio (95% CI) | Exact sequence match |
|---|---:|---:|---:|
| Vanilla via speculative core | 0.970 (0.965–0.975) | 0.967 (0.962–0.973) | 100.0% |
| PLD forced miss | 0.972 (0.968–0.976) | 0.969 (0.964–0.974) | 100.0% |
| PLD | 1.063 (1.036–1.092) | 1.070 (1.039–1.103) | 83.3% |
| PLD gate | 1.053 (1.025–1.081) | 1.058 (1.027–1.091) | 83.3% |
| Routed | 1.050 (1.023–1.079) | 1.056 (1.025–1.088) | 83.3% |

Both negative controls are token-exact and show about 2.8–3.0% lower end-to-end
throughput than direct vanilla. Treat this as observed framework/path overhead;
the speculative speedups have not been corrected by adding it back.

PLD is the strongest tested method. Routed/PLD paired speed ratio is 0.989
(95% CI 0.985–0.993), so routing is measurably slower than pure PLD here. PLD
speedup is task-dependent: mean ratios are 1.084 on GSM8K, 1.115 on HumanEval,
and 0.989 on CNN/DM. The speed result is not lossless: one quarter of the 150
prompts have a PLD-family sequence mismatch, and the mismatches repeat across all
three order seeds. Do not present the speedup as exact decoding.

The GPU temperature snapshots ranged from 83–88 °C and SM clocks from 2,070 to
2,535 MHz. Order rotation reduces systematic ordering bias, but this is still a
single-device run with thermal variation.

## Numerical diagnosis

There are 25 unique mismatching prompt IDs (7 GSM8K, 5 HumanEval, 13 CNN/DM).
Vanilla repeated exactly on all of them. For all 25, the default trace finds an
active KV digest difference before the first mismatch, while canonical sequential
verification matches vanilla for all 48 generated tokens. `divergence_report.json`
records 75 method/prompt cases (three PLD-family methods per prompt) as numerical
divergence based on that controlled intervention. This is strong evidence for
batch/cache numerical path dependence in this NF4 setup; it does not establish
the same rate for other models, precisions, or attention backends.

The separate paired exactness/latency study is in
`experiments/numerical_validation/tradeoff_n25_r1/`. On the 25 mismatch prompts,
sequential verification was 100% exact at 0.968x vanilla throughput (95% CI
0.959–0.976). KV replay reached 68% exactness at 0.166x; a 0.05 margin guard
reached 36% at 0.725x. Sequential verification is the only tested recovery path
that restored sequence identity across all selected prompts.

## Artifacts and reproduction

- `raw_measurements.jsonl`: all 2,700 per-method observations and token IDs.
- `summary.json`, `statistics.json`: prompt-level paired distributions and CIs.
- `divergence_report.json`: joined first-mismatch/cache/sequential evidence.
- `*_divergences_default.json`, `*_divergences_sequential.json`: per-prompt traces.
- `figures/`: paired speedup, by-task speedup, control overhead, exactness PDFs.
- `config.yaml`, `environment.json`: run configuration, code/checkpoint revision,
  package/backend metadata, dataset hashes, and device data.

Re-run the benchmark with:

```bash
python scripts/run_strong_baseline_comparison.py \
  --samples-per-bench 50 --max-new-tokens 48 --warmup-prompts 2 \
  --repeats 1 --order-seeds 42 43 44 \
  --output-dir experiments/benchmark_validation/reproduction_n50_r1
```

The original benchmark was executed at repository commit `a37a4e4`; the trace,
summary, and figure tooling was added afterward. Its full SHA and model revision
are recorded in `environment.json`.
