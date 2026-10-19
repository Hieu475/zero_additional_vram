# Context Matrix n=20: Provenance and Completeness Audit

Scope: output-128 only; frozen 20-prompt cohort at contexts 128, 384, 1024, and 2048; BNB and HQQ; vanilla, PLD K=4, and CKA90/K2.

- Structurally complete session cells: **24/24**.
- Complete cells carrying `working_tree_dirty=true`: **14/24**; clean-provenance cells: **10**.
- Missing session files: **0**.
- A complete cell has 20 prompt triplets (60 records), matching prompt hashes, expected backend/context/capacity, successful warmups, and no recorded failures.

## Cell status

| Context | Backend | Session 1 | Session 2 | Session 3 |
|---:|---|---:|---:|---:|
| 128 | BNB | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows) |
| 128 | HQQ | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows) |
| 384 | BNB | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows) |
| 384 | HQQ | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows) |
| 1024 | BNB | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows) |
| 1024 | HQQ | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows) | 20/20 prompts (60 rows) |
| 2048 | BNB | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows) |
| 2048 | HQQ | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows) | 20/20 prompts (60 rows) |

## Provenance findings

14 complete cells include records with `working_tree_dirty=true`; 10 complete cells have consistently clean working-tree and source-tree flags, prompt hashes, and one execution commit. The HQQ/context-384 session-2 cell contains an earlier dirty partial run plus a clean append, so it is structurally complete but remains exploratory. Each recorded commit is checked against current `src/zassd` and the runner snapshot is hashed in `audit.json`. Historical dirty-file lists/source hashes were not saved, so reverted uncommitted edits cannot be ruled out for dirty records.

Records preserve model/tokenizer revision, backend quantization metadata, attention backend, prompt token count/hash, output budget, KV capacity, warmup status, randomized method order/seed, torch/CUDA/GPU model/capability, latency, TPS, token IDs, exactness, and peak allocated/reserved VRAM.

Missing per-run provenance includes GPU driver, Transformers/bitsandbytes/GemLite versions, temperature, EOS token/stop reason, GPU temperature/SM clock/power, and source-tree diff/hash. Temperature 0.0 is visible in the runner source at the recorded revision, but is not embedded in records.

## Metric and interpretation caveats

The runner computes decode TPS as `(generated_tokens - 1) / decode_time`, excluding the first output token. Preserve and label that definition in comparisons. Within-backend token exactness is recorded; cross-backend token equality is not a correctness criterion because BNB and HQQ use different quantized weights.

A prompt triplet is complete only when vanilla, PLD, and CKA rows all exist. Partial prompt triplets are preserved and must not count as complete observations. The output-128 matrix does not meet the stated three-session protocol, and no output-32/256 matrix is included.

## Recommended rerun order

1. Preserve raw records unchanged as exploratory artifacts. All 24 session cells are now structurally complete and no session files are missing; the HQQ/context-384 session-2 cell mixes historical dirty records with a clean append and should remain exploratory.
2. Add missing runtime provenance to the runner and start from a clean, committed source tree.
3. Keep clean-provenance cells as the primary subset. Rerun mixed/dirty cells only if a clean three-session confirmatory result is required; do not treat structural completeness as provenance cleanliness.
4. After three clean sessions exist per backend/context, compute paired summaries and document/session-aware uncertainty.
5. Decide whether output-32 or output-256 is needed after the primary output-128 matrix is valid.

Machine-readable details: [`audit.json`](audit.json).
