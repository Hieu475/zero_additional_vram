# Context Matrix n=20: Provenance and Completeness Audit

Scope: output-128 only; frozen 20-prompt cohort at contexts 128, 384, 1024, and 2048; BNB and HQQ; vanilla, PLD K=4, and CKA90/K2.

- Structurally complete session cells: **24/24**.
- Complete cells carrying `working_tree_dirty=true`: **0/24**; clean-provenance cells: **24**.
- Missing session files: **0**.
- A complete cell has 20 prompt triplets (60 records), matching prompt hashes, expected backend/context/capacity, successful warmups, no measured failures, and clean working/source-tree provenance. Rerun cells are read from `clean_reruns/`; original files remain unchanged.

## Cell status

| Context | Backend | Session 1 | Session 2 | Session 3 |
|---:|---|---:|---:|---:|
| 128 | BNB | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows) |
| 128 | HQQ | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows) |
| 384 | BNB | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows) |
| 384 | HQQ | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows) |
| 1024 | BNB | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows) |
| 1024 | HQQ | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows) | 20/20 prompts (60 rows) |
| 2048 | BNB | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows) |
| 2048 | HQQ | 20/20 prompts (60 rows); clean rerun | 20/20 prompts (60 rows) | 20/20 prompts (60 rows) |

## Provenance findings

24 selected cells have clean working-tree/source-tree flags, prompt hashes, and one execution commit per cell. 14 cells read the clean reruns under `clean_reruns/`; original records remain intact. Each recorded commit is checked against current `src/zassd` and runner snapshots are hashed in `audit.json`.

Rerun records include model/tokenizer revision, quantization metadata, prompt hash, KV capacity, warmup status, method order/seed, PyTorch/CUDA/GPU, driver and package versions, telemetry snapshots, latency/TPS, token IDs, EOS/stopping status, exactness, and allocated/reserved VRAM.

Historical dirty records lack some provenance fields and are not selected where a clean rerun exists. Replacement cells were run in BNB-then-HQQ blocks, not interleaved, so absolute cross-backend throughput is descriptive. Temperature/power measurements are point samples, not continuous traces.

## Metric and interpretation caveats

The runner computes decode TPS as `(generated_tokens - 1) / decode_time`, excluding the first output token. Preserve and label that definition in comparisons. Within-backend token exactness is recorded; cross-backend token equality is not a correctness criterion because BNB and HQQ use different quantized weights.

A prompt triplet is complete only when vanilla, PLD, and CKA rows all exist. All 24 backend/context matrices now have three clean sessions; no output-32/256 matrix is included.

## Recommended rerun order

1. Keep original dirty records unchanged and analyze clean replacements listed in `clean_reruns/manifest.json`.
2. The output-128 three-session matrix now has clean provenance. Continue with numerical audit or a separately specified output-length study.
3. Interpret intervals cautiously because three session clusters per condition provide limited session-level uncertainty.

Machine-readable details: [`audit.json`](audit.json).
