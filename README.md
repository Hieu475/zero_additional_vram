# ZASSD-B1: Exact Speculative Decoding under Shape-Dependent 4-bit Kernels

Research code for one question: **can speculative decoding on a 4-bit quantized
LLM emit exactly the same tokens as sequential greedy decoding of a precisely
defined reference?**

Batched verification of `K+1` candidate tokens uses different kernels, tile
shapes and reduction orders than single-token decoding. With bitsandbytes NF4
this changes logits and KV values, so Prompt-Lookup speculation matched the
vanilla output on only 83.3% of sequences (N=150) in our earlier protocol, and
21/60 in a later backend comparison. This repository studies a
**fixed-row (B1) schedule**: every decode-time linear/RMSNorm/attention call is
padded to a pinned physical row count, so singleton and block verification run
the same arithmetic, and draft tokens never enter target KV.

> Status: research prototype, one GPU (RTX 4050 Laptop 6 GB), mostly
> Qwen2.5-3B-Instruct NF4. The strongest result is the **B2 loop** (equal to the
> unmodified bitsandbytes greedy output, ~1.1-1.25x faster than eager BNB); B1 is
> the earlier, faster-but-different-reference path. Nothing here is a formal proof;
> see `experiments/exact_quantized_feasibility/B2_OPERATOR_ARGUMENT.md` for what is
> argued, measured, or open, and the limits below.

## What "exact" means here

| Reference | Meaning | Status |
|---|---|---|
| **B1** (fixed-row singleton) | Block verification equals fixed-row singleton decoding, token for token and KV prefix for KV prefix | Matched in all recorded runs (finite empirical evidence) |
| **B2** (original BNB greedy) | Equals the unmodified bitsandbytes greedy output | Tokens and per-step KV-prefix hashes equal to BNB greedy on all recorded runs (about 90 development prompt runs on 3B, up to 384 tokens, 17 EOS-ending; 4 prompts each on Qwen2.5-0.5B/1.5B), and tokens equal on a 50-prompt held-out GSM8K cohort (300-349). Finite evidence, not a proof. |

The B1 reference changes the arithmetic of the model. Speedups against original
BNB latency are therefore not evidence of BNB trajectory equivalence.

The B2 activationvec path (`_fused_nf4_m3_kernel` + `bnb_rowwise` attention) processes the
M=3 block with the same per-row arithmetic as the BNB singleton, so block and greedy outputs
are bitwise identical. Speedup over BNB then depends entirely on acceptance rate.

## Main development results (Qwen2.5-3B NF4, RTX 4050)

Full details and caveats: `experiments/exact_quantized_feasibility/FINDINGS.md`.

### B1 results — `triton_nf4_fixed_three_rows_group_scale_packed`, rowwise attention, K=2, rows=3

| Cohort | Runs | B1 match | Speedup vs B1 singleton | Speedup vs BNB latency | BNB tokens match |
|---|---:|---:|---:|---:|---:|
| 12 prompts, 256 tokens | 24 | 24/24 | 1.516x | 1.228x | 6/24 |
| 8 prompts, 512 tokens | 16 | 16/16 | 1.612x | 1.300x | 2/16 |

Intervals are prompt-cluster bootstraps over selected development prompts, not
workload-level estimates. A frozen held-out GSM8K cohort (indices 300-349,
`data/benchmarks/gsm8k_exactness_holdout.jsonl`) has not been used yet.

### B2 results — activationvec kernel + `bnb_rowwise` attention + native-BNB fallback + lazy LM head, K=2, rows=3

Empty-proposal cycles use the unmodified BNB M=1 step; the block's LM head is projected row by
row only up to the first rejected draft. Speed is relative to eager HF bitsandbytes greedy.

| Cohort (128 tokens, 3 rounds, prompt-cluster bootstrap) | Prompts | Equal to BNB | Speedup vs eager BNB (95% CI) |
|---|---:|---:|---|
| GSM8K dev | 30 | 30/30 | 1.166x [1.145, 1.188] |
| HumanEval dev | 30 | 30/30 | 1.249x [1.208, 1.296] |
| GSM8K held-out 300-349 (eager-LM-head variant, run once) | 50 | 50/50 | 1.117x [1.100, 1.135] |
| GSM8K held-out 400-449 (this variant, run once, tokens and per-step KV) | 50 | 50/50 | 1.154x [1.136, 1.172] |

B1 on the same held-out cohort is equal to BNB on only 24/50 prompts at 1.108x. B2 and B1 are not
distinguishable in speed there (B2/B1 = 1.008, CI [0.995, 1.022]); the difference is exactness.

**Baseline caveat.** Eager HF is CPU-launch-bound (~84% of a decode step is CPU enqueue). A
`torch.compile` static-cache BNB decode is 1.29-1.45x faster than eager and **faster than B2 (B2 is
0.82x of it on dev, 0.91x on held-out 400-449)**, but it is not an exact baseline (tokens equal eager on
9/12 dev and 25/50 held-out prompts). B2's claim is exactness to
eager BNB plus a modest speedup over eager, not a speedup over compiled decoding.

The same loop is also equal to BNB (tokens and per-step KV) on Qwen2.5-0.5B and 1.5B development prompts, at
1.24x and 1.16x over eager.

Details and per-experiment caveats: `RESEARCH_STATUS_20261102.md` and `B2_OPERATOR_ARGUMENT.md`.

## Provenance of old runs

Experiment manifests pin `source_sha256` of the scripts and kernels that produced them.
The B1-focused cleanup deleted or edited many of those files, so pins in older manifests
no longer match the working tree. To re-validate or rerun a historical run, check out the
tag `evidence-pre-cleanup` (commit `06cf0d7`), where the pins match.

## Limitations

- One GPU, one main model/revision, greedy decoding, batch size 1, NF4 (bitsandbytes) only.
- Exactness is backed by recorded token/KV-hash equality, not by a proof; the operator argument
  in `B2_OPERATOR_ARGUMENT.md` leaves the kernel's equality with bitsandbytes' reduction order, PyTorch
  dispatch independence and cross-toolchain stability open;
  `src/zassd/certification/greedy.py` provides a conditional margin certificate with no validated kernel bounds.
- Triton `tl.sum` reduction order is compiler-defined, so bitwise behavior depends on the Triton/ptxas version.
- Previously used development prompts; the sample is small.

## Repository layout

```
src/zassd/
  decoding/b1_exact.py          B1 singleton and fixed-row block generators
  decoding/b1_triton_contract.py  contract-bound entry points
  certification/                fixed-row adapters (qwen2/llama/phi3/olmo2), NF4 Triton kernels,
                                capability gates, reference contract, conditional certificate
  cache/                        static KV cache, rollback
  backends/                     model backend loading (hf_bnb_nf4, ...)
  baselines/, controllers/,
  profiling/, models/           PLD / layer-skip baselines and helpers used by the pilots
scripts/                        runners (run_b1_*), audits (audit_*), summaries (summarize_*); _common.py = shared helpers
tests/                          cache invariants, B1 exactness, backend and numerical-trace tests
experiments/
  exact_quantized_feasibility/  FINDINGS.md, PROTOCOL.md, PROOF_OBLIGATIONS.md, operator contract, raw runs
  backend_validation/, numerical_causality/, cross_precision_validation/, ...  supporting evidence
RESEARCH_EXACT_QUANTIZED_SPECULATIVE_DECODING.md   research plan
RESEARCH_EXECUTION_PLAN_EXACT_QUANTIZED.md          execution gates
```

## Quickstart

```bash
pip install -e .
bash reproduce.sh --quick        # unit tests only
bash reproduce.sh                # + one short GPU B1 pilot
python scripts/run_b1_block_pilot.py --help
```

A `Dockerfile` pins PyTorch 2.11 / CUDA 12.8. Tested versions: Triton 3.6.0,
transformers 5.14.1, bitsandbytes 0.50.0.

## License

MIT.
