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
> Qwen2.5-3B-Instruct NF4 on development prompts. Nothing here is a formal
> proof or a held-out result. See the limits below.

## What "exact" means here

| Reference | Meaning | Status |
|---|---|---|
| **B1** (fixed-row singleton) | Block verification equals fixed-row singleton decoding, token for token and KV prefix for KV prefix | Matched in all recorded runs (finite empirical evidence) |
| **B2** (original BNB greedy) | Equals the unmodified bitsandbytes greedy output | **Token-trajectory achieved** on 23 closed-loop development runs via activationvec kernel; bitwise logits+KV verified on 2 prompts (1 cycle each). Speedup requires sufficient acceptance. |

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

### B2 results — `_fused_nf4_m3_kernel` (activationvec) + `bnb_rowwise` attention, K=2, rows=3

Token trajectory matches original BNB greedy in all 23 closed-loop development runs.
Speedup above BNB only when acceptance rate is sufficient.

| Prompt / budget | Sessions | B2 match | Acceptance | Speedup vs BNB |
|---|---:|---|---:|---:|
| HumanEval/112, 128 tok | 2 | 4/4 | 65.2% | 1.242–1.284x |
| HumanEval/116, 64 tok | 2 | 6/6 | 87.5% | 1.265–1.275x |
| HumanEval/119, 64 tok | 2 | 6/6 | 66.7% | 1.162–1.206x |
| HumanEval/120, 64 tok | 2 | 6/6 | 65.5% | 0.953–0.990x |
| Low-acceptance prompts (0–39%) | — | all match | <40% | 0.78–0.89x |

With native-BNB M=1 fallback on empty-proposal cycles, token trajectories still match original BNB greedy
(all rounds), and development-prompt geomean speedup vs BNB is 1.06x (CNN/DM, 2 prompts), 1.09x (GSM8K dev, 6),
1.15x (HumanEval dev, 6); without fallback it is 0.91-0.97x. A reject/rollback audit (5 scenarios, 1 prompt)
matched BNB singleton KV prefixes and next logits bitwise. Details: `RESEARCH_STATUS_20261102.md` section 10.

Bitwise logits and KV verified identical to BNB singleton on 2 prompts (1 cycle, all-accepted
path only). Closed-loop runs check token-ID equality only. See scope limits in FINDINGS.md.

## Provenance of old runs

Experiment manifests pin `source_sha256` of the scripts and kernels that produced them.
The B1-focused cleanup deleted or edited many of those files, so pins in older manifests
no longer match the working tree. To re-validate or rerun a historical run, check out the
tag `evidence-pre-cleanup` (commit `06cf0d7`), where the pins match.

## Limitations

- One GPU, one main model/revision, greedy decoding, batch size 1, NF4 (bitsandbytes) only.
- Exactness is backed by recorded token/KV-hash equality, not by an operator-level proof;
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
