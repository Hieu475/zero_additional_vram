# Context Matrix n=20: Provenance and Completeness Audit

Scope: output-128 only; frozen 20-prompt cohort at contexts 128, 384, 1024, and 2048; BNB and HQQ; vanilla, PLD K=4, and CKA90/K2.

- Structurally complete session cells: **13/24**.
- Complete cells carrying `working_tree_dirty=true`: **13/13**; clean-provenance cells: **0**.
- Missing session files: **10**.
- A complete cell has 20 prompt triplets (60 records), matching prompt hashes, expected backend/context/capacity, successful warmups, and no recorded failures.

## Cell status

| Context | Backend | Session 1 | Session 2 | Session 3 |
|---:|---|---:|---:|---:|
| 128 | BNB | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | missing |
| 128 | HQQ | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | missing |
| 384 | BNB | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | missing |
| 384 | HQQ | 20/20 prompts (60 rows); dirty tree | 4/20 prompts (12 rows); dirty tree | missing |
| 1024 | BNB | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | missing |
| 1024 | HQQ | 20/20 prompts (60 rows); dirty tree | missing | missing |
| 2048 | BNB | 20/20 prompts (60 rows); dirty tree | 20/20 prompts (60 rows); dirty tree | missing |
| 2048 | HQQ | 20/20 prompts (60 rows); dirty tree | missing | missing |

## Provenance findings

All present session records report `working_tree_dirty=true`. Each recorded commit is present, the exact runner file at each commit is retrievable and hashed in `audit.json`, and `src/zassd` matches current HEAD across the captured revisions. This reconstructs the committed execution source, but historical dirty-file lists/source hashes were not saved; uncommitted edits at run time that were later reverted cannot be ruled out. Treat the data as reproducible against the recorded commit, with that limitation stated.

Records preserve model/tokenizer revision, backend quantization metadata, attention backend, prompt token count/hash, output budget, KV capacity, warmup status, randomized method order/seed, torch/CUDA/GPU model/capability, latency, TPS, token IDs, exactness, and peak allocated/reserved VRAM.

Missing per-run provenance includes GPU driver, Transformers/bitsandbytes/GemLite versions, temperature, EOS token/stop reason, GPU temperature/SM clock/power, and source-tree diff/hash. Temperature 0.0 is visible in the runner source at the recorded revision, but is not embedded in records.

## Metric and interpretation caveats

The runner computes decode TPS as `(generated_tokens - 1) / decode_time`, excluding the first output token. Preserve and label that definition in comparisons. Within-backend token exactness is recorded; cross-backend token equality is not a correctness criterion because BNB and HQQ use different quantized weights.

A prompt triplet is complete only when vanilla, PLD, and CKA rows all exist. Partial prompt triplets are preserved and must not count as complete observations. The output-128 matrix does not meet the stated three-session protocol, and no output-32/256 matrix is included.

## Recommended rerun order

1. Preserve existing raw records unchanged as exploratory artifacts; 13 cells are structurally complete, one HQQ/context-384 cell is partial (4/20 prompts), and 10 session files are absent.
2. Add missing runtime provenance to the runner and start from a clean, committed source tree.
3. Keep structurally complete cells when their recorded source commit matches current HEAD and no source-file diff is found; rerun only the partial/missing cells under an enhanced provenance runner. Rerun previously complete cells only if source equivalence fails or a source modification is discovered.
4. After three clean sessions exist per backend/context, compute paired summaries and document/session-aware uncertainty.
5. Decide whether output-32 or output-256 is needed after the primary output-128 matrix is valid.

Machine-readable details: [`audit.json`](audit.json).
