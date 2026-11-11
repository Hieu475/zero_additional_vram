# Shape-invariant 4-bit kernels for exact speculative decoding

Research code. Single question: can speculative decoding on a 4-bit LLM be made
**bitwise equal** to sequential greedy decoding *without* giving up the
memory-bandwidth saving that makes speculative decoding worthwhile?

Background (measured, RTX 4050 6 GB, bitsandbytes NF4 / torchao INT4): verifying a block of
K+1 tokens changes results bit-for-bit in some operators, so naive speculative decoding
leaves the sequential greedy trajectory (26-82% of sequences with prompt-lookup; 34-66% with a
real draft model). Running the sensitive operators row by row removes the divergence (0 of 50-60
sequences in every cell) but reads the weights once per row. The planned contribution is a set of
**M-invariant kernels** (LM head, attention, RMSNorm, NF4 GEMM) that read weights once and keep
each row's reduction order fixed.

**Read first:** `docs/PLAN_SHAPE_INVARIANT_KERNELS.md` (the single plan, work packages, pass/fail gates).

## Layout

```
docs/PLAN_SHAPE_INVARIANT_KERNELS.md   the research plan (follow this)
docs/direction_c/                      shape-audit results write-up, literature notes, paper outline/draft
docs/archive/                          earlier plans/status (B1/B2 fixed-row work, peer review)
src/zassd/                             backends (hf_bnb_nf4, torchao_int4_wo, ...), KV cache, certification
                                       (fixed-row adapters, NF4 kernels), n-gram drafting, utils
scripts/shape_audit.py                 audit tool: ops / plan / trajectory / summarize
scripts/audit_*.py, a1-a3_*.py         per-operator audit, PLD and draft-model divergence, CUDA-graph decode
scripts/run_*.sh                       batch runners for the audits (results below)
src/zassd/kernels/drafter.py           learned drafter (feature-conditioned, trimmed-vocab head)
scripts/drafter_*.py, wp9_*.py, wp10_*.py   drafter data/training/eval; exact CUDA-graph loop with drafter (wp9) and hybrid PLD+drafter (wp10)
scripts/summarize_drafter_runs.py      aggregates results/drafter_runs/*.json (mean/sd over interleaved reps)
kernels_reference/                     old Triton/CUDA kernel prototypes, kept as a starting point
experiments/exact_quantized_feasibility/.../native_bnb_profiles/   all audit results + C_SHAPE_AUDIT_SUMMARY.md
data/                                  benchmark prompt files
paper/                                 figures and LaTeX
tests/                                 unit tests (12)
```

## Quickstart

```bash
pip install -e .
bash reproduce.sh            # unit tests
PYTHONPATH=src:scripts python scripts/shape_audit.py summarize   # regenerate the summary table
```

## Status of results (see docs/direction_c/RESEARCH_DIRECTIONS_A_C.md)

- Operator audit and row-wise fix: done on bnb 0.49.2/0.50.0 and torchao; HQQ+GemLite excluded (non-deterministic).
- Draft-model speculative decoding (Qwen2.5-3B NF4 + 0.5B draft): naive 17-33/50 diverged, row-wise 0/50.
- Divergence prediction model: negative (R2 -3.4 against the pre-set R2 >= 0.8; correlation 0.87).
- Exact speculative decoding on a CUDA-graph baseline: 1.009x (negative; motivates the kernel work).
- Kernel work (M-invariant LM head, attention, RMSNorm, NF4 GEMM): done on Qwen2.5-3B/1.5B/0.5B, OLMo-2-1B, Phi-3.5-mini, Llama-3.2-1B/3B and Mistral-7B; exact PLD loop on CUDA graphs is 1.14-1.32x faster than the exact M=1 baseline (1200 prompt-model pairs, 0 mismatches) and 1.26-1.39x faster than the earlier row-wise exact loop; neural draft model is a negative result. See docs/PLAN_SHAPE_INVARIANT_KERNELS.md sections 13-17 and docs/PAPER_OUTLINE_KERNELS.md.
- Learned drafter and hybrid (docs/DRAFTER_2X_PLAN.md, sections 9-12): an EAGLE-style drafter (feature-conditioned, trimmed 16k-token head) trained by self-distillation, run inside the same CUDA graph, verified with the M-invariant kernels. Exact in all 18 (model, set, method) cells: tokens and KV bitwise equal to the M=1 baseline on every prompt, eager reference on every 10th. Speedup over the exact M=1 baseline, mean over 2-3 interleaved reps (Qwen2.5-3B / 1.5B, K=3, 100 prompts per set):

  | set | PLD | learned drafter | hybrid (in-graph PLD + drafter) |
  |---|---|---|---|
  | GSM8K | 1.22x / 1.23x | **1.70x / 1.58x** | 1.66x / 1.57x |
  | HumanEval | 1.39x / 1.53x | 1.37x / 1.34x | **1.58x / 1.70x** |
  | CNN/DM | **1.25x / 1.55x** | 1.11x / 1.07x | 1.24x / 1.47x |

  The drafter wins on math, PLD wins or ties on summarization, the hybrid wins on code. No cell reaches 2x; the calibrated cost model puts the ceiling near 1.9x for this drafter on this GPU (drafter chain costs 2.0 ms/step, bandwidth-bound). A confidence-gated variable block size is a negative result (simulation, section 12). One GPU, two Qwen sizes; the drafter must be trained per model (about 1-2 h data + 20 min training).

Earlier history is recoverable from git tag `pre-restructure-20261105`.
