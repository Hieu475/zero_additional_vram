# Shape-invariant 4-bit kernels for exact speculative decoding

Research code. Single question: can speculative decoding on a 4-bit LLM be made
**bitwise equal** to sequential greedy decoding *without* giving up the
memory-bandwidth saving that makes speculative decoding worthwhile?

Background (measured, RTX 4050 6 GB, bitsandbytes NF4 / torchao INT4): verifying a block of
K+1 tokens changes results bit-for-bit in some operators, so naive speculative decoding
leaves the sequential greedy trajectory (22-82% of sequences with prompt-lookup; 34-66% with a
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
- Kernel work (M-invariant LM head, attention, RMSNorm, NF4 GEMM): done on Qwen2.5-3B/1.5B/0.5B and OLMo-2-1B; exact PLD loop on CUDA graphs is 1.14-1.32x faster than the exact M=1 baseline (700 runs, 0 mismatches); neural draft model is a negative result. See docs/PLAN_SHAPE_INVARIANT_KERNELS.md sections 13-17 and docs/PAPER_OUTLINE_KERNELS.md.

Earlier history is recoverable from git tag `pre-restructure-20261105`.
