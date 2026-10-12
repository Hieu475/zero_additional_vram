# Conservative reduced synchronization ablation

The branch `research/reduced-sync` adds a conservative execution variant to
`self_speculative_generate`. It skips entropy extraction when no controller,
router or acceptance trace needs that value, and removes phase-end CUDA waits
whose preceding greedy scalar reads already synchronize. It keeps token/EOS
decisions, verification, cache commit/crop and the verification timing boundary
unchanged. The reduced path is selected with `reduced_sync=True`; the default
remains the reference implementation.

## Correctness

`tests/test_reduced_sync_exactness.py` compares reference and reduced-sync raw
token IDs and active KV digests at K=0, 1, 2 and 4, tests EOS at the first token
and inside a speculative cycle, and checks one-token and short output budgets.
On the RTX 4050, the dedicated test file passed **8/8** in 10.9 seconds. Both
variants produced identical token IDs and per-cycle KV digests in the tested
cases. This establishes regression correctness for this model/runtime path; it
does not establish equivalence for every backend, precision or sampling mode.

## Uninstrumented paired pilot

`initial_rtx4050/` contains 12 prompts, two paired repeats per prompt, randomized
variant order, NF4 weights, FP16 compute, SDPA, CKA90/K2 and a 16-token output
limit. The raw rows carry the same tokenized-input hash for each pair. Bootstrap
resampling uses prompts as units and averages the two repeats within each
prompt.

| Measurement | Mean ratio, reduced/reference | Prompt-bootstrap 95% CI |
| --- | ---: | ---: |
| End-to-end speedup | 1.020x | 0.995–1.051 |
| Decode-only throughput speedup | 1.010x | 0.992–1.026 |

Sequence exact-match was 100% across all paired runs. The intervals include
1.0, so this pilot does not establish a reliable speedup. Mean time per prompt
was 0.484 s reference and 0.476 s reduced-sync. Draft time was 0.261/0.257 s,
verify time 0.173/0.175 s and cache/commit time 1.84/1.08 ms. These are small,
single-session estimates and need a larger repeated benchmark.

## Nsight Systems comparison

`nsight_rtx4050/` stores the raw `.nsys-rep` captures, per-run metrics and
`comparison.json`. Both captures used prompt 1, CKA90/K2, NF4/FP16, 24 output
tokens, the same code revision and CUDA/NVTX/OS runtime tracing. Nsight's host
CPU sampling was unavailable in this environment, so the comparison reports
CUDA synchronization API counts and durations rather than sampled CPU wait
stacks.

| CUDA API | Reference | Reduced-sync |
| --- | ---: | ---: |
| `cudaDeviceSynchronize` calls | 28 | 12 |
| `cudaStreamSynchronize` calls | 45 | 36 |
| Total synchronization API calls | 73 | 48 |
| Sum of synchronization API durations | 90.83 ms | 88.92 ms |

The one captured sequence matched exactly. The count fell by 34%, while summed
API duration fell by about 2.1 ms (2.1%). Captured end-to-end times were 0.815 s
and 0.797 s, respectively, but each variant has only one profiled run; those
times are affected by Nsight and are not evidence of a throughput improvement.
The data indicate that removing redundant synchronization calls lowers API
counts, while most remaining synchronization and host wait is still driven by
per-token scalar reads required by the current CPU-controlled draft/EOS path.
The device-resident decision path should only be explored if repeated,
uninstrumented measurements confirm these scalar reads dominate.

## Reproduction

```bash
python -m pytest tests/test_reduced_sync_exactness.py -q

python scripts/benchmark_reduced_sync.py \
  --output-dir experiments/reduced_sync/my_run \
  --limit 12 --repeats 2 --max-new-tokens 16 --k 2 --config cka_90

python scripts/analyze_reduced_sync.py \
  --raw experiments/reduced_sync/my_run/raw_measurements.jsonl \
  --output experiments/reduced_sync/my_run/analysis.json

nsys profile --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o experiments/reduced_sync/nsight_rtx4050/reference \
  python scripts/profile_layer_skip_nsight.py \
  --mode layer_skip --config cka_90 --k 2 --max-new-tokens 24 \
  --prompt-id 1 --sync-variant reference \
  --output-json experiments/reduced_sync/nsight_rtx4050/reference.json
```

Use `--sync-variant reduced_sync_conservative` for the second Nsight capture,
then summarize both with `scripts/analyze_reduced_sync_nsight.py`.
