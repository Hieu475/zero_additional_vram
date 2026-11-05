# Working draft: Auditing shape invariance of 4-bit LLM inference, and restoring exact speculative verification

Status: first working draft (2026-11-05), written for the author to edit. Every number below comes from a result file under
`experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/` (called `PROFILES/`);
`[TODO]` marks anything not yet supported by evidence. Claims are deliberately narrower than the repository's earlier notes.

## Abstract

Speculative decoding is lossless in exact arithmetic, but a verifier that processes several tokens in one forward pass can return
different logits than single-token decoding, because GPU kernels choose reduction strategies by input shape. We study this for
weight-only 4-bit models on one consumer GPU. We audit, operator by operator, whether a block of M rows is bitwise equal to the same
operator applied one row at a time on the same inputs, and we measure how often a naive prompt-lookup decoder (PLD) leaves the
sequential greedy trajectory. In our setup, attention, the LM head and occasionally RMSNorm differ between block and per-row execution for every
model and backend tested; the 4-bit projections differ depending on library version and block size. Naive blocks produced a different
128-token continuation in 22–92% of prompts (K=2). Running only the operator classes the audit flags one row at a time during
verification produced no diverged sequence in any run of the corrected code, on three datasets (including 120 prompts never used for any
design decision), two 4-bit backends and five models from three families. We also report negative results: one backend (HQQ+GemLite) was not
run-to-run deterministic and was excluded, and speculative decoding added only 1.009x on top of an exact CUDA-graph baseline.

## 1. Introduction

[TODO: motivation paragraph; cite the speculative decoding papers and the batch-invariance line of work.] The question we ask is narrow:
for a quantized model and greedy decoding, which operators make a multi-token verification pass differ bitwise from sequential decoding,
how much does that matter for the generated text, and can the answer be turned into a fix that restores equality?

Contributions, in decreasing order of how much evidence supports them:
1. A measurement procedure and tool (`scripts/shape_audit.py`: `ops`, `plan`, `trajectory`, `summarize`) that attributes block/per-row
   differences to operator classes and emits the list of classes to run row by row.
2. Measurements across block sizes (M=2..16), context lengths (to 3072 tokens), five models from three families, two valid 4-bit backends and two
   bitsandbytes versions.
3. End-to-end measurements of how often naive block verification leaves the sequential greedy trajectory, with confidence intervals.
4. Evidence that the audit-derived list restores exact trajectories on all tested configurations.

We do not claim the phenomenon is new: kernel reduction strategies depending on shape is the premise of batch-invariant inference, and
an Apple-silicon study reports that a batch of five tokens versus one shifts quantized logits enough to flip the argmax.

## 2. Setting and definition of "exact"

Reference decode: sequential greedy decoding with batch size 1 and one token per forward pass, on the same loaded model and a
preallocated KV cache. A method is exact on a prompt if its generated tokens (and committed KV) equal the reference bit for bit.
Block verification feeds the pending token and K drafted tokens (M=K+1 rows) in one forward pass; drafts come from n-gram lookup in the
prompt and generated history (PLD, n-gram size 3). We use K=2 unless stated. Greedy decoding, batch size 1, 128 new tokens.

## 3. Method

**Operator audit (`ops`).** For teacher-forced blocks of M tokens on the exact sequential state, we run the unmodified model once on the
block, capture each operator's real inputs and outputs, and re-apply the same operator to each row separately; we count elements that
differ bitwise. Operators: quantized projections, RMSNorm, LM head, and each scaled-dot-product-attention call. For grouped-query models
the per-row attention call uses the compact K/V with `enable_gqa=True`, as the single-token step does (see Section 6, a bug we hit).
We also compare block logits with sequentially computed logits (rows that are not bitwise equal, argmax flips).

**Plan (`plan`).** Operator classes with at least one mismatching element become the row-wise list.

**Trajectory test (`trajectory --rowwise`).** Naive PLD (block verified by one ordinary forward pass) is compared against the reference
over many prompts; with `--rowwise`, the listed classes are evaluated one row at a time during the verification pass only.
Divergence is "any token differs within 128 tokens".

## 4. Experimental setup

One RTX 4050 Laptop GPU (6 GB), PyTorch 2.11 (CUDA 12.8), bitsandbytes 0.50.0 unless stated (0.49.2 in a separate environment), torchao
0.18.0 (Int4WeightOnly, tile-packed format, bf16 activations), Transformers 5.14.1. Models: Qwen2.5 0.5B/1.5B/3B Instruct, OLMo-2-1B,
Phi-3.5-mini. Datasets: GSM8K (first 50 prompts of the project's evaluation file; the audits used the first 20), HumanEval (50),
CNN/DailyMail (50 plus two disjoint 60-prompt groups from a separate file with 0/150 overlap), and 12 long-context prompts of
511–3072 tokens (concatenated CNN/DailyMail articles). Raw prompts, no chat template. Confidence intervals are Wilson 95%.

## 5. Results

### 5.1 Which operators differ (20 GSM8K prompts, 48 steps, 960 rows)

| Config | Projections (M=3) | Projections (M=4) | LM head | SDPA | Argmax flips (M=3) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B, bnb 0.50.0 | 0 elements | 13.0% | 0.19% | 18.6% | 4/960 |
| Qwen2.5-3B, bnb 0.49.2 | 34.2% | 34.2% | 0.19% | 18.6% | 2/960 |
| Qwen2.5-1.5B, bnb 0.50.0 | 0 | 13.9% | 0.11% | 18.3% | 7/960 |
| Qwen2.5-0.5B, bnb 0.50.0 | 0 | 14.3% | 0 | 3.6% | 6/960 |
| OLMo-2-1B, bnb 0.50.0 | 0 | 15.4% | 0.05% | 17.4% | 2/960 |
| Phi-3.5-mini, bnb 0.50.0 | 0 | 14.8% | 0.08% | 19.2% | 3/960 |
| Qwen2.5-3B, torchao INT4 | 0 | 0 | 0.16% | 17.4% | 9/960 |

RMSNorm differs rarely (for example 87 of 141.6M elements for Qwen2.5-3B). Block logits were not bitwise equal to sequential logits
on essentially all rows (960/960 for Qwen2.5-3B); the argmax flipped on 0.2–1.0% of rows. For Qwen2.5-3B (bnb 0.50.0) block sizes 2 to 16 give
the same SDPA/LM-head rates, and context lengths 512–3072 change the SDPA rate by under a percentage point. HQQ+GemLite is excluded (Section 6).
Source: `C_SHAPE_AUDIT_SUMMARY.md`, `op_shape_sensitivity_gsm8k20_v1.json`, `c2_*`, `c3_*`, `c4_*`, `c8_*`, `c10_*`.

### 5.2 How often naive blocks leave the greedy trajectory (K=2, 128 tokens; diverged / prompts)

| Config | GSM8K | HumanEval | CNN/DM (eval) | CNN/DM (ext150, 60) |
|---|---:|---:|---:|---:|
| Qwen2.5-3B bnb | 18/50 | 19/50 | 33/50 | 47/60, 49/60 |
| Qwen2.5-1.5B bnb | 25/50 | 13/50 | 36/50 | 39/60 |
| Qwen2.5-0.5B bnb | 33/50 | 26/50 | 38/50 | 41/60 |
| OLMo-2-1B bnb | 17/50 | 17/50 | 27/50 | 29/60 |
| Phi-3.5-mini bnb | 23/50 | 31/50 | 41/50 | 55/60 |
| Qwen2.5-3B torchao INT4 | 19/50 | n/a | 38/50 | 41/60 |

The long-context set (511–3072 tokens) gives 11/12 (Qwen2.5-3B) and 4/12 (OLMo-2-1B).

### 5.3 Row-wise evaluation of the audited classes

On every configuration in the table above, evaluating the classes named by `plan` one row at a time during verification produced
**0 diverged sequences** (50 or 60 prompts each; Wilson upper bound 6–7%; 12 prompts for long context, upper bound 24%).
On GSM8K the same list (LM head, RMSNorm, SDPA) was applied to all five bnb models; on HumanEval and CNN/DM each model used the list that
`plan` produced from its own GSM8K audit (Qwen2.5-0.5B: SDPA only; Qwen2.5-1.5B: LM head and SDPA; others: LM head, RMSNorm, SDPA).
Control experiments: dropping RMSNorm from the Qwen2.5-3B list left 3/50 sequences diverged; with K=3 (M=4) on bnb 0.50.0 the same list
left 18/50 diverged until the projections were added (then 0/50). On bnb 0.49.2 the list including projections gave 0/50 at K=2 and K=3.
The list for torchao needed no projection class at M=3 or M=4, as the audit predicted.
CNN/DM prompts in `cnndm_ext150` (120 prompts, none used for any choice of list or code path) are the independent check.
Source: `fix_*`, `heldout_*`, `ext150_*`, `ext150b_*`, `longctx_*`, `c8_*`, `c10_*`.

### Figures (generated by `scripts/make_paper_figures.py` into `paper/figures_c/`, each with a CSV of the plotted values)

**Figure 1** (`fig1_trajectory_divergence`): share of prompts whose 128-token continuation differs from sequential greedy decoding,
naive block verification (orange) versus the audit-derived operator list evaluated row by row (blue), K=2, with 95% Wilson intervals
and `diverged/prompts` labels, for four prompt sets. Caveats to put in the caption: (i) the torchao row is empty for HumanEval (not run);
(ii) in the last panel the torchao naive result uses 60 prompts while the audit-list result pools two disjoint 60-prompt groups (120),
and Qwen2.5-3B (bnb) pools both groups for both methods; (iii) on GSM8K the list is the same three classes for every model, on the other sets it
is the model's own `plan` output; (iv) the CNN/DM (50) panel is the data on which the GQA bug was found, so only the last panel is an
independent test of the corrected code.

**Figure 2** (`fig2_projection_vs_block_size`): share of 4-bit projection output elements that differ between a block of M rows and
row-by-row execution (Qwen2.5-3B, 20 GSM8K prompts). bitsandbytes 0.50.0 is exact up to M=3 and differs from M=4 (13.0%);
bitsandbytes 0.49.2 differs from M=3 (34.2%); torchao INT4 is exact at both M=3 and M=4 (not measured beyond M=4). The aqua series is
below 3:1 contrast on the page background, so all series are directly labelled and the CSV is the table view.

## 6. Negative results, exclusions and a bug

* **HQQ+GemLite is not run-to-run deterministic.** The M=1 reference decode of the same 50 prompts differs between two identical runs
  in 5/50 sequences (bitsandbytes: 0/50), and one GemLite projection called five times with the same input gave different outputs in
  64% (M=1) and 37% (M=3) of elements (`run_to_run_noise_v1.json`). We do not know why. Earlier shape-sensitivity numbers for this
  backend are withdrawn; it is excluded from all results.
* **A bug in our first row-wise attention.** The single-token step calls SDPA with `enable_gqa=True` on the compact K/V, while the block pass
  receives K/V expanded by `repeat_kv`. Our first per-row call used the expanded K/V and was not bitwise equal on 164–605-token
  prompts (Qwen models still diverged on 10–36 of 50). Using the compact K/V fixed this. Short contexts did not reveal the problem.
  The operator-audit rates were essentially unchanged by the fix (SDPA 18.64% before and after on GSM8K).
* **Speed.** A CUDA-graph decode loop that is exact against a single-token StaticCache reference reached 1.40x the eager speed
  (58.1 vs 41.6 tokens/s, 30 prompts); PLD with K=2 on top of it reached 1.009x of the graph baseline, below our pre-set 1.05x stopping rule.
  68% of steps found no n-gram draft and accepted blocks averaged 0.86 drafts. We therefore do not claim a speedup from exact speculative decoding here.

## 7. Limitations

One consumer GPU and one library stack (kernel choice also depends on GPU architecture and library version; bitsandbytes 0.50.1/0.50.2
change dispatch heuristics [TODO: untested]); models up to 3B; K ≤ 3; greedy only; batch size 1; each cell run once. "No diverged
sequence" is an observation on the sampled prompts, not a proof of exactness. Row-wise lists were chosen on 20 GSM8K prompts and then
applied elsewhere; the GQA fix was written after observing the failure on CNN/DailyMail, so that data set is not an independent test of
the fix (the 120 `cnndm_ext150` prompts are). Only two valid 4-bit backends, both INT4 weight-only. No GPTQ/AWQ, no larger models,
no other GPU [TODO]. The 12-prompt long-context result is indicative only.

## 8. Related work (to be completed)

* Batch-invariant inference (Thinking Machines Lab; SGLang, vLLM deterministic modes): premise that shape-dependent reductions
  break invariance; bf16, no weight quantization.
* LLM-42 (arXiv 2601.17768): decode–verify–rollback with a fixed-shape verifier; states it does not integrate with speculative
  decoding; no weight-quantized model in the body text; kernel invariance assumed, noting that atomic reductions are not covered.
* Yuan et al. (arXiv 2506.09501): batch size, GPU count and precision change greedy outputs; BF16/FP16/FP32 only.
* "Lossless but Not Free" (arXiv 2607.17283, single-author preprint): Apple silicon; reports that on a quantized Metal backend a batch of
  five versus one shifts logits enough to flip argmax, and states bit-exact greedy equivalence there is unattainable for implementations
  whose baseline and verifier use different batch shapes. Our fix evaluates the sensitive operators at the baseline's shape.
* "The Integer Alibi" (arXiv 2608.13756): INT8 W8A8, two kernels in one engine; 4-bit and block shape left to future work.
[TODO: speculative decoding papers; GPTQ/AWQ/NF4/HQQ/Marlin papers; verify all citations against the sources before submission.]

## 9. Conclusion

For the tested 4-bit configurations, block verification differs from single-token decoding in a small, identifiable set of operator
classes; the set depends on model, backend and library version; naive blocks changed the text for 22–92% of prompts; and running
exactly the audited classes one row at a time removed every observed divergence. We expect the procedure rather than the specific list to
transfer, and have not tested that claim on other GPUs.
