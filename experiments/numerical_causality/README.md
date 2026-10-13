# Numerical causality: first NF4 cohort

This run traces the 16 prompt IDs selected from the independent short-context
PLD K=4 mismatch cohort. It pins Qwen2.5-3B-Instruct to revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, uses NF4 weights with FP16 compute,
SDPA, static KV, `prompt+generated` lookup, K=4, and the benchmark's static
capacity (`prompt_length + max_new_tokens + K + 16`). The trace budget is 32
new tokens. See `config.json` and `environment.json` for the run settings.

## Current result

- All 16 prompts showed a numerical KV difference and a numerical logit
  difference at some cycle.
- Eight prompts diverged in token IDs within the 32-token budget; the other
  eight were exact within this run's budget.
- At the first output divergence, canonical KV substitution restored the
  vanilla argmax in 2 of the 8 diverged prompts.
- In 6 prompts, the mismatch remained when the same candidate inputs were
  evaluated with canonical KV. This shows that KV history alone does not
  explain those flips; batched versus token-at-a-time execution remains a
  supported explanation.
- The two restored cases were one-token fallback cycles with the same token
  prefix as vanilla. This is direct evidence that cache history can cause a
  target argmax flip in those cases.

These results are an initial causal attribution on one model, precision,
attention backend, and GPU. The run does not establish that cache history is
the only mechanism, and the 8/16 reproduction rate differs from the earlier
independent benchmark's 16/16 mismatch count. Keep both outcomes in the paper;
do not present this cohort trace as a reproduction of all prior mismatches.

## Artifacts

- `traces/multicycle_nf4_sdpa_static.json`: full per-prompt/per-cycle trace,
  including token histories, cache lengths, first differing KV layer, KV/logit
  errors, margins, and argmax agreement.
- `interventions/kv_substitution.jsonl`: same-prefix canonical replay and
  KV-substitution outcomes for each traced cycle.
- `first_divergence_summary.json` and `causal_attribution.json`: cohort-level
  summary and per-prompt classification.
- `traces/*smoke.json`: intermediate debugging runs; use the full trace above
  for reported cohort results.

## Reproduction

```bash
python scripts/trace_multicycle_numerical_divergence.py \
  --ids qasper_119_ctx384 multi_news_189_ctx384 qasper_109_ctx384 \
    gov_report_98_ctx384 gov_report_134_ctx384 gov_report_80_ctx384 \
    qmsum_2_ctx384 narrativeqa_6_ctx384 multi_news_143_ctx384 \
    qmsum_4_ctx384 qmsum_196_ctx384 qmsum_54_ctx384 \
    narrativeqa_155_ctx384 qasper_163_ctx384 narrativeqa_139_ctx384 \
    narrativeqa_55_ctx384 \
  --samples 16 --max-new-tokens 32 \
  --output experiments/numerical_causality/traces/multicycle_nf4_sdpa_static.json
python scripts/analyze_numerical_causality.py \
  --input experiments/numerical_causality/traces/multicycle_nf4_sdpa_static.json \
  --output-dir experiments/numerical_causality
```

Focused consistency tests:

```bash
python -m pytest -q tests/test_numerical_trace_consistency.py
```
