# Benchmark validation artifacts

These artifacts are exploratory RTX 4050 measurements for the Qwen2.5-3B-Instruct
NF4 configuration. Each run stores its commit, dataset hashes, configuration,
raw rows, and summary. Do not treat the pilot as a final paper result.

## Pilot benchmark

`pilot_15_45a6a57/` contains 5 prompts from each of GSM8K, HumanEval, and CNN/DM,
48 generated tokens, one repeat, and three rotated method orders (45 paired rows
per method). Prompt-level bootstrap intervals include zero-one boundaries as
appropriate. Speedup intervals against vanilla include 1.0 for PLD, PLD gate,
and routed, so this pilot does not establish a throughput win.

The two negative controls were token-exact against direct vanilla. Their mean
throughput ratios were 0.9755 (vanilla through the speculative core) and 0.9743
(PLD forced miss). This measures roughly 2.5% path overhead for this setup. PLD,
PLD gate, and routed each had 93.3% row-level exact matches: all three order
seeds for one CNN/DM prompt diverged, which motivated the focused trace below.

## Focused token trace

`exactness_6_851c5d7/` stores token IDs for two prompts from each benchmark and
the first mismatch position. The mismatch was `cnndm_001`, at generated token
index 28, and was repeated across the three PLD-family methods. The direct and
speculative-core vanilla controls matched exactly.

`nf4_cnndm_001_851c5d7/` contains the default trace, sequential-verification
trace, and batch-equivalence audit for that prompt. The 398-token prefill logits
and KV were bitwise equal between the audit paths. Batched versus sequential
verification produced non-bitwise K/V from layer 1 and small logit differences
(maximum absolute difference 0.21–0.31 over K=1,2,4), while all tested greedy
argmax decisions still agreed within each horizon. In the default decode, the
first sequence mismatch occurred at index 28 where the vanilla reference logit
margin was 0.0. Re-running with canonical sequential verification matched the
vanilla sequence for all 48 tokens. This supports batch/cache numerical path
dependence as the cause for this prompt; broader precision and backend coverage
is still needed before making a general NF4 claim.

## Reproduction commands

Run from the repository root with an available NVIDIA GPU:

```bash
python scripts/run_strong_baseline_comparison.py \
  --samples-per-bench 5 --max-new-tokens 48 --warmup-prompts 2 \
  --repeats 1 --order-seeds 42 43 44 \
  --output-dir experiments/benchmark_validation/reproduction

python scripts/audit_nf4_batch_equivalence.py \
  --benchmark data/benchmarks/cnndm_eval.jsonl --ids cnndm_001 --samples 1 \
  --max-k 4 --output experiments/benchmark_validation/nf4_batch_audit.json

python scripts/trace_first_divergence.py \
  --benchmark data/benchmarks/cnndm_eval.jsonl --ids cnndm_001 --samples 1 \
  --method prompt_lookup --k 4 --max-new-tokens 48 --hash-kv \
  --output experiments/benchmark_validation/first_divergence.json

python scripts/trace_first_divergence.py \
  --benchmark data/benchmarks/cnndm_eval.jsonl --ids cnndm_001 --samples 1 \
  --method prompt_lookup --k 4 --max-new-tokens 48 --hash-kv \
  --canonical-verify --output experiments/benchmark_validation/sequential_verify.json
```
