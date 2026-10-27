# Protocol for the first exact quantized speculative-decoding feasibility study

This directory contains **diagnostic screening**, not a held-out paper result. The aim is to decide whether removing repeated full-prefix canonical replay could plausibly make an exact implementation fast enough to justify kernel work.

## Reference contract for the BNB pilot

- Model: `Qwen/Qwen2.5-3B-Instruct`, pinned revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Quantization: bitsandbytes NF4, double quantization enabled, FP16 compute, SDPA attention.
- Device in this pilot: RTX 4050 Laptop GPU. The manifest records versions and GPU name.
- Tokenization: one token-ID tensor per prompt, passed unchanged to every method.
- Greedy reference: one prompt prefill, then one target forward per generated token, static KV capacity 2048, temperature zero, tokenizer EOS stopping rule.
- Comparisons: token-ID equality for the complete output, including EOS if emitted. This is backend-relative equality, not FP16/FP32 equivalence.

The first rows are the first `--per-task` prompts from `gsm8k_eval.jsonl`, `humaneval_eval.jsonl`, and `cnndm_eval.jsonl`. These prompts have been used in earlier ZASSD work and are **not** held out. The runner records input file hashes and per-prompt token-ID hashes. It refuses to overwrite a nonempty output directory.

## Pinned B1 reference identity

The current B1 reference contract is [`b1_reference_contract_v3.json`](b1_reference_contract_v3.json), with ID `ca155f567ba080347f765335e8999b0256abafe87c88c0263ac38344b2b2ca27`. It pins Qwen2.5-3B NF4, the loaded quantized-state digest, tokenizer/template identity, GPU/runtime/code hashes, `StaticPreallocatedKVCache` capacity 2048, SDPA padded attention, and physical rows=3. This physical-row reference supports K≤2; the rows=5 and rows=9 schedules define separate references because shape can change the trajectory.

Contract v3 supersedes v2 because per-cycle decoder instrumentation changed a pinned source hash. It was derived from a fully identified K=2/rows=3 EOS-branch pilot after that instrumentation. Run `scripts/build_b1_reference_contract.py --contract ... --validate-manifest ...` to check a run manifest against it. The B1 runner accepts `--reference-contract` and fails before decode if the loaded model, tokenizer, quantized state, runtime, operator sources, schedule, or cache do not match. Runner self-hash is kept in run provenance but excluded from this compatibility check to avoid a self-referential hash. The runner accepted the pinned runtime on the K=1/2 development runs and rejected a rowwise attention mismatch before writing a manifest or decoding. Across three previously used task prompts and two process sessions, singleton tokens and active-KV digests matched; block matched B1 in all six runs. These finite checks do not prove general correctness; the ID is not a correctness proof.

## Per-cycle oracle pilot

[`b1_cycle_oracle_k1_k2_v5.json`](b1_cycle_oracle_k1_k2_v5.json) summarizes K=1/2 runs on the same three development prompts across two process sessions, with physical rows fixed at 3 and a 64-token budget. The oracle uses measured per-cycle draft, verify and commit wall times and only includes transitions whose emitted tokens match the common singleton trajectory and whose shared cache frontier passed the state audit. Its path completed for 3/3 prompts. Median oracle gain over the best fixed-K schedule under the same additive timing model was 1.0005×, so this small sample shows essentially no benefit from switching K each cycle. Median predicted speedup over B1 singleton was 1.211×, but this excludes K-selection/switching/checkpoint costs and is a finite hindsight estimate, not a deployable result.

## Long-context cycle and cache costs

[`b1_cycle_oracle_long_cnndm_v2.json`](b1_cycle_oracle_long_cnndm_v2.json) extends the same reference to a 397-token prompt and 256-token budget. It covers K=1/2 in two process sessions each and admits only state-audited transitions. Its idealized oracle gain over fixed K=2 is 1.0217×, with 64 K switches and an estimated 2.07 ms break-even cost per switch. The companion [`cache primitive benchmark`](b1_cache_checkpoint_cost_cnndm_v1.json) measures zero-copy fork, logical crop, checkpoint restore and clone+restore at active lengths 397/461/652. Cycle timing excludes trace-list append and state-observer hashing; the oracle adds a per-request residual calibrated from observed runs, which may not transfer exactly to a different cycle count. The artifacts are diagnostic: one long-context prompt, no online selector, and no production checkpoint implementation.

## Methods and accounting

`run_exact_quantized_feasibility.py` runs vanilla, PLD batched, PLD sequential, and the existing margin guard on the same prompt. Method order rotates by prompt. A short warmup occurs before timed rows. The guard threshold 1.5 is inherited from the previous exploratory study; it is not a mathematically certified threshold.

For each guard run, the script reports the fraction of its measured time spent rebuilding canonical KV and a **free-restoration ceiling**:

`T_ceiling = T_guard - T_measured_canonical_replay`.

The corresponding `tokens / T_ceiling` assumes that exact canonical state could be restored instantly while every other observed decision and cost remained unchanged. It is a cost-only counterfactual. It is not an executable algorithm, a guaranteed upper bound for every possible policy, or evidence of correctness. If the guard's observed output is inexact, this ceiling is not a ceiling for an exact output.

Both end-to-end latency and decode latency are stored. The initial feasibility pilot has a single process session; its ratios are diagnostic and do not support confidence claims. The later B1 fixed-row K sweep uses two independent process rounds on the same GPU, rotating method order by round and reversing K order. It covers six development prompts; the three added IDs were already present in earlier workspace experiments, so this is not fresh or held-out data. Before freezing a policy, expand to a broader development cohort and keep the held-out cohort unused.

The analyzer also reports a **whole-prompt hindsight oracle**: on each prompt, it picks the fastest *observed exact* method among vanilla, batched PLD, sequential PLD and margin guard. It is an upper bound on this fixed menu under perfect knowledge of future output equality. It is not a cycle-level oracle, a policy that can be deployed, or a proof that an online certificate can select those prompts cheaply.

## Required checks before a paper claim

1. Freeze a genuinely unseen prompt set and record its IDs and hashes **before** selecting any threshold or kernel configuration. A hash split of prompts already inspected is insufficient.
2. Validate that candidate source, cache capacity, prefill, EOS, attention masks, model revision and precision match within each comparison.
3. Run repeated sessions on at least two GPUs and three model families. Establish each backend's own vanilla reference.
4. Report first divergence, exact whole-sequence match, KV drift, replay fraction, peak VRAM and paired latency. Keep all prompt rows, including failures and OOMs.
5. State the operator-level assumptions that make a certificate valid. An empirical maximum error or 100% test match is not a universal error bound.

The conditional certificate code is in `src/zassd/certification/greedy.py`. It deliberately requires externally established state and kernel error bounds; no current quantized backend supplies them. The implementation must remain fail-closed until those bounds or a shape-invariant kernel contract are proved and tested.

## Cohort provenance audit

An attempted reservation of 50 IDs from `gsm8k_ext150.jsonl` is recorded in `frozen_gsm8k_ext50.json`. A subsequent read-only audit found **all 50 IDs** in both `experiments/16_hybrid_roofline/standard_suite_gsm8k_ext150.json` and `standard_suite_gsm8k_N200.json`. The attempt is therefore **invalid as a genuinely unseen held-out cohort**. The manifest is retained, with the failed provenance check recorded in `frozen_gsm8k_ext50_provenance_audit.json`. A new external or independently collected cohort must be frozen before any final evaluation.

`scripts/prepare_exactness_holdout.py` subsequently froze GSM8K `main` test indices 300–349 in `data/benchmarks/gsm8k_exactness_holdout.jsonl`. The [`manifest`](frozen_gsm8k_300_349.json) pins dataset revision `740312add88f781978c0658806c59bc2815b9866`, cohort file hash, IDs and prompt hashes. All 50 prompts were checked against that pinned revision; no ID or prompt-text hash overlapped existing local benchmark JSONL files, and none of their IDs appeared in old local result artifacts. **Do not run this cohort while making kernel or policy choices.** This local provenance check cannot rule out public dataset exposure during model pretraining or use in external experiments.
