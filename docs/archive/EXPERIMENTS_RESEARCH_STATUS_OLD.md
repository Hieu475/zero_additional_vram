# Research status: ZASSD

Updated 2026-10-13 on branch `research/numerical-causality`. Results below are tied to their stated model, prompt set, precision, and device; do not generalize beyond those scopes.

## P0 — numerical correctness and performance

### Numerical causality

The 16-prompt Qwen2.5-3B-Instruct NF4/FP16, SDPA/static-KV PLD K=4 cohort had KV and logit numerical differences in all 16 traces. Eight prompts changed token argmax within 32 output tokens; eight remained exact within that budget. Canonical KV substitution restored the vanilla argmax at the first changed output token in 2/8 divergent prompts. In 6/8 cases the flip remained under canonical KV with the same batched candidate execution. This supports cache state as a direct cause in two cases and shows it is not a complete explanation for the cohort. Benchmark reruns previously had 16/16 mismatches, so the 8/16 causal-trace reproduction must remain explicit.

Evidence: [`numerical_causality/README.md`](numerical_causality/README.md), [`first_divergence_summary.json`](numerical_causality/first_divergence_summary.json), [`causal_attribution.json`](numerical_causality/causal_attribution.json).

### Precision/backend screen

All 14 batched/sequential cells completed on Qwen2.5-0.5B-Instruct, RTX 4050, four prompts, K=4, and 16 output tokens. Numerical KV/logit drift appeared on all four prompts in every batched cell, across NF4, FP16, BF16, and FP32. No sequential cell drifted in this sample. Token divergence varied by configuration (0–2 of 4). This suggests interaction between batched execution and numeric format/backend/cache; it is a small-model mechanism screen, not direct Qwen2.5-3B validation.

Evidence: [`cross_precision_validation/README.md`](cross_precision_validation/README.md), [`matrix_summary.json`](cross_precision_validation/qwen25_05b_rtx4050/matrix_summary.json).

### Reduced-sync

Three uninstrumented sessions per configuration, 30 prompts per session, K=2 and 32 output tokens: all sequences matched exactly. Every within-session prompt-bootstrap interval for E2E and decode speedup contains 1.0. The measured screening does not establish throughput gain; retain as engineering cleanup/secondary ablation rather than a primary contribution.

Evidence: [`reduced_sync/repeated_screening_rtx4050/README.md`](reduced_sync/repeated_screening_rtx4050/README.md).

## P1 — generalization and policy

### Thermal PLD validation

Three cool-start and three heat-soaked sessions each used the same 12 384-token prompts, Qwen2.5-3B NF4/FP16, SDPA/static KV, and 32 output tokens. Mean-of-session decode speedups: PLD K=4 was 1.043x cool and 1.050x hot; K=8 was 1.075x cool and 1.093x hot. Each configuration matched vanilla on 7/12 prompts in every session. The same five prompts diverged in all six sessions, and vanilla histories were identical. Session-level intervals are based on only three runs; conditions were run on separate occasions and temperature was observed, not controlled per measurement. This supports repeatable performance characterization but not a causal claim that temperature has no effect or a lossless acceleration claim.

Evidence: [`thermal_validation/README.md`](thermal_validation/README.md), [`thermal_repeated_analysis.json`](thermal_validation/thermal_repeated_analysis.json).

### Router and feasibility

Existing GSM8K/HumanEval/CNN-DM policy-regret analysis reports routed policy efficiency of 0.919–0.929 versus a per-prompt best-fixed comparator; pure PLD had higher average TPS in those runs. The router therefore has not shown added value over pure PLD. The offline comparator uses full-prompt observations and must be described as diagnostic, not deployable. Existing feasibility diagnostics also show poor profitable-action recall under calibrated cost estimates. A fresh unified paired evaluation including exactness, action distribution, precision/recall, and regret is still outstanding.

Evidence: [`oracle_regret.json`](16_hybrid_roofline/oracle_regret.json), [`feasibility_validation/README.md`](feasibility_validation/README.md).

### Cross-hardware

No second-GPU validation has been completed. RTX 4050 is the only device represented in the new P0/P1 data above.

## P2 — paper

The paper has not yet been consolidated around the verified claims. Pending: freeze selected results, align tables/figures and limitations, and add an artifact manifest with commit/config/data/reproduction commands.

## Validation performed

The numerical trace consistency suite passed 4 tests. Python compilation and `git diff --check` passed for the research scripts at the previous validation point. The environment emitted a SciPy/NumPy compatibility warning; it did not fail the focused test run.
