# Qwen2 fixed-row B1 operator contract

**Status: experimental and incomplete.** This document pins the intended
execution contract for the current Qwen2 fixed-row prototype and lists the
unproven implementation assumptions. It is not a certificate that BNB NF4 or
the current PyTorch/Transformers kernels are bitwise shape invariant.

## Claim and reference

B1 compares the block decoder with greedy decoding under the **same** pinned
fixed-row operator schedule. It does not claim token equality with the original
BNB greedy path (B2). The measured Qwen2.5-3B configuration is:

| Setting | Value |
|---|---|
| Model checkpoint | `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Quantization | bitsandbytes NF4, double quantization, FP16 compute |
| Model attention implementation | SDPA |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Runtime | Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8, Transformers 5.14.1, bitsandbytes 0.50.0 |
| Cache | `StaticPreallocatedKVCache`, capacity 2048, batch 1 |
| B1 schedule measured | `attention_strategy="padded"`; K/R = 2/3, 4/5 or 8/9 |
| Decode budget measured | 256 output tokens |

Pilot manifests now include driver version, GPU compute capability/UUID,
selected PyTorch SDPA and determinism flags, a model-config hash, tokenizer
vocabulary/special-token/template identity, and hashes of the operator,
decoder, candidate lookup, cache, backend loader and runner source. The
`contract_id` hashes those values with the model revision, quantization
setup, physical row schedule, cache capacity, budget, and input file hashes.
The latest targeted EOS-draft run records `quantized_state_sha256`
`c6212d37bd02338c60536b1ca674b3e0c76fbcd810ba5df8e8e28b0e16f4f6cf`, computed
from named model parameters/buffers and recursively exposed bitsandbytes
quantization-state metadata/tensors. It strengthens loaded-state identity but
does not prove operator behavior or hash hidden runtime state. If `nvidia-smi`
is unavailable, driver data is null. This is still not a certified contract ID.

## Supported input domain

The intended domain is one unpadded token sequence, batch size one, Qwen2
modules in evaluation mode, no sliding-window layers, no requested attention
weights, and a static cache with enough capacity for prompt plus output budget.
The same checkpoint, quantized state, tokenizer-produced token IDs, prefill,
device, software stack, attention strategy, and physical row count must be used
by the singleton reference and block decoder. `physical_rows` must be at least
`K+1` and must remain fixed for the entire comparison.

The public B1 generators do not accept an attention mask argument. The
context manager now requires `config.model_type == "qwen2"` and
`config._attn_implementation == "sdpa"`, and wraps the model's `forward` to
reject external attention masks, explicit position IDs,
custom cache positions, and embedding inputs. The attention modules also
reject batch sizes other than one, non-static caches, training mode,
sliding-window attention, and requested attention weights. Transformers may
still create its ordinary internal causal mask for a multi-token forward; the
patched decode attention ignores that internal tensor and builds its own
causal visibility from the active prefix. Arbitrary padding, packed-sequence,
prefix, or user masks remain outside the contract. A caller-provided token-ID
tensor can itself contain padding IDs, which the generator cannot reliably
distinguish from intentional tokens; unpadded token IDs remain a caller
obligation.

## Decode operator schedule

Let `R=physical_rows`, `m` be the number of real decode rows in the current
forward, and `p` the active KV prefix length before those rows.

1. **Linear and RMSNorm calls.** For a rank-3 tensor with batch one and
   `m <= R`, the wrapper pads the sequence-row axis with zero inputs to R and
   makes it contiguous. The default `fixed_rows` strategy calls the original
   `Linear4bit` or `nn.Linear`; `Qwen2RMSNorm` is also called on the padded
   tensor. The opt-in `triton_nf4` strategy replaces only NF4 `Linear4bit`
   calls with the experimental direct-packed-weight kernel described below;
   it launches one independent program per real token row and does not compute
   padded rows.
   Each real token row is independent in the mathematical module. Rank, batch,
   and empty-row violations fail closed. A prefill longer than R uses the
   ordinary module path because prefill is shared by the singleton and block
   methods; a decode call whose rows exceed R raises instead of falling
   through to the original module.
2. **Q/K/V and positions.** Qwen2Attention projects Q, K and V through those
   wrapped linears, reshapes to heads, applies the model-supplied RoPE cosine
   and sine tensors, then writes the new K/V rows to the cache. The query and
   key position tensors must be identical for equal logical token prefixes.
3. **KV head expansion.** `repeat_kv` expands grouped-query K/V heads before
   SDPA. The grouping factor and tensor strides must match the pinned model.
4. **Padded attention.** The implementation pads query, key and value sequence
   axes to R. It constructs an additive mask of shape
   `[1, 1, R, p+R]`; for row i, keys `0..p+i` are visible and all later keys
   receive negative infinity. It invokes SDPA once with dropout zero,
   `scale=self.scaling`, and `is_causal=False`, then retains the first m
   outputs. Singleton and block calls at the same logical prefix therefore
   present the same Q row, K/V tensor shape and mask row to SDPA, provided the
   pre-SDPA tensors are already equal.
5. **Layer and vocabulary path.** The attention output projection, residual
   additions, post-attention norm, Qwen2 MLP projections/activation, final
   norm, and language-model head run through the same wrappers and ordinary
   Qwen2 code. Every remaining operation must preserve independence across
   token rows and the same per-row arithmetic for the claim to hold.
6. **Greedy selection.** For each target row, `argmax` uses PyTorch's index
   ordering for ties. The block verifier compares each target prediction to
   the PLD candidate and commits only the matching prefix. EOS candidates are
   not consumed. On rejection, the cache logical length is cropped to the
   prefix plus pending and accepted tokens; on EOS or budget completion, the
   terminal emitted token is removed from KV as in the singleton reference.

The alternative `rowwise` strategy instead invokes SDPA separately per real
row, truncating K/V to length `p+i+1` and passing no mask. It is a distinct
reference schedule and must have a distinct contract identity. A result under
`rowwise` does not certify `padded`, or vice versa.

## Experimental Triton NF4 linear strategy

`quantized_linear_strategy="triton_nf4"` is an opt-in prototype implemented
in [`nf4_fixed_rows.py`](../../src/zassd/certification/nf4_fixed_rows.py).
It reads BNB's packed NF4 codes and quantization metadata directly, including
the nested 8-bit scale state, and computes one token row/output-channel tile
per Triton program. The reduction runs only across the fixed input-feature
dimension with FP32 accumulation and an explicit FP16 compute boundary. It
does not materialize a full FP16 copy of the model weights. Unsupported dtypes,
layouts, or quantization-state forms raise instead of falling back to BNB.

This kernel defines a new B1 reference schedule; it is not intended to match
BNB's GEMM bitwise. Both fixed-row singleton and block runs must use the same
strategy, Triton/compiler build, device, and physical row count. A synthetic
nested-NF4 microcase matched BNB dequantization with zero observed output
difference and bitwise-equal repeated rows across singleton and three-row
calls. Two warmed 32-token Qwen2.5-3B development runs at K=2/R=3 and K=4/R=5
matched their B1 singleton tokens, with all 32 committed KV-prefix digests
matching B1. The K=2 run also matched original BNB tokens; K=4/R=5 differed
from original BNB, consistent with its distinct fixed-row reference. These are
finite empirical observations. A compiler audit now covers layer-0 q_proj,
gate_proj, and down_proj on SM 8.9. For each specialization, PTX and cubin
hashes were identical when recompiling with physical rows R=1 versus R=5;
only the launch-grid row count changed. SASS had separate FADD/FMUL
instructions, no FFMA or HMMA, and included warp-shuffle/shared-barrier
instructions for reduction. This confirms the inspected binary is row-count
independent for these dimensions and this pinned toolchain; it does not prove
all layer shapes or future compiler builds follow the same schedule. The
[machine-code audit](triton_nf4_qwen3b_cnndm002_k4_rows5_v1/kernel_code_audit/summary.json)
and [R=1 comparison](triton_nf4_qwen3b_cnndm002_k4_rows5_v1/kernel_code_audit_rows1/summary.json)
include PTX, cubin, SASS, and hashes. Latency was 0.863x B1 singleton and
0.409x original BNB at K=2, and 0.734x B1 singleton and 0.352x BNB at K=4.
It remains a slow correctness prototype, not a performance win or certified
capability. See the
[`K=2/R=3`](triton_nf4_qwen3b_cnndm002_k2_rows3_v1/) and
[`K=4/R=5`](triton_nf4_qwen3b_cnndm002_k4_rows5_v1/) artifacts.

A synthetic boundary audit used nested NF4 with `(K,N)` around the 16/32/64/
128/256/512 boundaries and odd/even output widths, plus three non-nested-scale
and FP16-bias variants. For all 18 tested shapes, the same row was bitwise
equal at M=1…5 and repeated equal rows stayed equal. BNB-dequantized FP16
linear matched bitwise in 17/18 shapes; the remaining nested/no-bias shape
`(K=130,N=17)` differed by at most `1.22e-4`. This finite result does
not extend the contract to arbitrary dimensions and confirms that the
prototype's B1 schedule is not universally bitwise-identical to BNB math.
See [`boundary_stress.json`](triton_nf4_qwen3b_cnndm002_k4_rows5_v1/boundary_stress.json)
and the reproducible harness
[`stress_nf4_fixed_rows_boundaries.py`](../../scripts/stress_nf4_fixed_rows_boundaries.py).

A paired 2048×2048 q_proj microbenchmark (50 invocations per sample, five
rounds) measured median Triton/BNB CUDA-time ratios of 2.46x, 9.19x, and 8.00x
at M=1, 3, and 5. At M=5 the medians were 0.345 ms (Triton) and 0.0431 ms
(BNB). This shows the current correctness-oriented linear kernel is the main
performance obstacle even before full-model attention/cache work. See
[`qproj_microbenchmark.json`](triton_nf4_qwen3b_cnndm002_k4_rows5_v1/qproj_microbenchmark.json)
and [`profile_nf4_linear_qwen.py`](../../scripts/profile_nf4_linear_qwen.py).

## Experimental group-scale variant

`quantized_linear_strategy="triton_nf4_group_scale"` is a separate, opt-in
Qwen2 B1 reference. It decodes each NF4 scale once per scale group and
broadcasts it over the corresponding weight tile. The currently registered
schedule pins `block_n=8`, `block_k=128`, and the group-scale kernel source in
the runtime identity. It accepts only K and tile sizes aligned to the BNB NF4
scale block; unsupported alignment or state fails closed.

On q_proj (2048×2048), its output remained bitwise equal across M=1/3/5, and
the K4/R5 group-scale probes matched logits and active KV for M=1…5 on two
development prefixes. It changes the q_proj output relative to the original
`triton_nf4` N=16/K=128 schedule, so it has a separate
[reference contract](triton_nf4_group_scale_qwen3b_cnndm002_k4_rows5_v1/reference_contract.json).
The two 32-token K4/R5 pilots matched the group-scale singleton at all 32
committed state prefixes. Both prompts also happened to match original BNB
token IDs in these runs, but that is finite empirical agreement, not B2
support.

The runner's paired speedup values (baseline time divided by block time) were
0.773x vs singleton and 0.531x vs BNB on `cnndm_002`, then 0.771x and 0.519x
on `cnndm_001`. Equivalently, block latency was 1.294x/1.883x the singleton/
BNB latency on `cnndm_002`, then 1.297x/1.927x on `cnndm_001`. On the same FP16
`cnndm_002` setup, its 32-token block run took 1.623 s versus 2.508 s for the
initial Triton kernel. It still did not beat B1 singleton or original BNB.
Nsight q_proj duration decreased from 347.5 us to 197.5 us; registers fell
from 70 to 40/thread and achieved occupancy rose from 55% to 95%. The
full-model speed gate therefore remains open. See the group-scale pilot and
Nsight report in the linked artifact directory above.

### K=256 grouped-scale schedule

The opt-in strategy `triton_nf4_group_scale_k256` fixes `block_n=8` and
`block_k=256`; it is a distinct operator schedule and reference contract from
the K=128 variant. Its contract ID is
`10b2da991db49afae753515a4681dc7c5d22262080980647212f370a8fb12c23` (see
[`reference_contract.json`](triton_nf4_group_scale_k256_qwen3b_cnndm002_k4_rows5_v1/reference_contract.json)).
The M=1…5 operator probes on both `cnndm_002` and `cnndm_001` matched the
selected K256 singleton logits and KV bitwise; at M=2 on `cnndm_001`, the
diagnosed module and attention tensors also had no differences. The two K4/R5,
32-token pilots on those prompts each matched all 32 committed KV prefixes.
Both pilots also matched original BNB token IDs, which remains finite
empirical agreement rather than B2 support. See the
[`cnndm_001` operator probe](triton_nf4_group_scale_k256_qwen3b_cnndm001_k4_rows5_v1/operator_probe_cnndm001_m1_m5.json).

The initial pilots had block latency ratios of 1.304× (`cnndm_002`) and
1.346× (`cnndm_001`) vs singleton, and 1.823×/1.836× vs BNB. Three additional
paired process sessions per prompt retained token equality vs B1 and BNB and
matched all 32 committed KV prefixes per run. Across six observations, the
geometric-mean latency ratios were 1.300× vs singleton (range 1.288–1.307×;
descriptive paired log-t 95% interval 1.290–1.309×) and 1.815× vs BNB (range
1.762–1.845×; interval 1.782–1.848×). The interval is diagnostic only because
it uses six runs on two fixed prompts. N8/K256 improved the q_proj
microbenchmark over N8/K128 at M=5, but not end-to-end latency. Verification
used 25–28 target calls and evaluated 48–50 target rows per 32 output tokens,
while singleton used 31 calls. Draft acceptance was 6/25 and 3/20 on the two
prompts. This contract is experimental and does not establish universal
operator/state equivalence or an end-to-end speedup. See the [paired
summary](triton_nf4_group_scale_k256_paired_multisession_summary_v1.json).

Nsight Compute measured the K256 M=5 q_proj launch at 177.38 μs, with 40
registers/thread, 95.02% achieved occupancy, no spills, 62% excessive global
sectors, and 88.26% L1/TEX throughput. K128 measured 197.50 μs with the same
register count and similar occupancy. Global access efficiency is the next
kernel-level optimization target; these are profiler timings, not end-to-end
latency. See the [K256 profile](triton_nf4_group_scale_k256_qwen3b_cnndm002_k4_rows5_v1/qproj_ncu_full.txt).

A synthetic K256 boundary audit covered 15 combinations of input/output
dimensions, nested/non-nested scales, and bias. The first row was bitwise
identical between M=1 and M=2…5 in all 75 checks, and repeated rows were equal
in all 15 cases. For K=256,N=17 with non-nested scales and no bias, the result
differed from BNB dequantize+`F.linear` by up to 0.0078125; 14 other cases were
bitwise equal to that construction. This audit supports shape invariance for
the tested K256 kernel cases, not BNB arithmetic equivalence. See the
[synthetic stress result](triton_nf4_group_scale_k256_qwen3b_synthetic_boundary_stress_v1.json).

### Packed-byte K256 schedule

The strategy `triton_nf4_group_scale_packed_k256` is registered as an
experimental Qwen2 schedule with its own runtime identity and contract
(`cf0d7b2094c6eb4a794e07e551c9fd5f44e536d5092a9dc99ce4ae64b9673fdc`). It
loads each NF4 packed byte once, decodes the high and low nibbles, then
interleaves them in K order for reduction. The M=1…5 operator probes on both
development prompts matched the packed singleton logits and KV bitwise; both
32-token pilots matched all 32 committed KV prefixes.

Across three paired sessions per prompt, token IDs matched B1 and original
BNB in all six runs. Block latency was 1.197× singleton (range
1.172–1.223×; diagnostic 95% paired log-t interval 1.173–1.220×) and 1.497×
BNB (range 1.477–1.537×; interval 1.472–1.522×). This improves over grouped
K256 but remains slower than both references. On q_proj, the kernel reduced
M=5 NCU duration from 177.38 to 124.61 μs and excessive global sectors from
62% to 15%, while increasing registers/thread from 40 to 56 and reducing
achieved occupancy from 95% to 71%. The confidence interval is only
diagnostic, based on six sessions across two fixed prompts. See the
[paired summary](triton_nf4_group_scale_packed_k256_paired_multisession_summary_v1.json),
[q_proj sweep](triton_nf4_group_scale_packed_k256_qwen3b_cnndm002_k4_rows5_v1/qproj_packed_byte_sweep_v2.json),
and [NCU report](triton_nf4_group_scale_k256_qwen3b_cnndm002_k4_rows5_v1/qproj_packed_ncu_full.txt).

### Packed-byte K256 at K=2/R=3

This uses a separate singleton contract from the K4/R5 schedule
(`4a190f8ae92791462997c64f06fa2857e9abf5e6f96d2ed7fdeac433c212088f`).
Operator probes M=1…3 on `cnndm_001` and `cnndm_002` matched logits/KV
bitwise. Three paired sessions per prompt matched B1 and BNB tokens and all 32
committed KV prefixes in each run. Block latency was 1.090× the K2 singleton
(range 1.033–1.129×; diagnostic paired log-t 95% interval 1.053–1.129×) and
1.352× BNB (range 1.290–1.391×; interval 1.312–1.392×). This narrows the gap
but still loses to both baselines in all six runs. The interval is
descriptive only for two fixed prompts. See the [K2 paired summary](triton_nf4_group_scale_packed_k256_k2_paired_multisession_summary_v1.json).

The pilot runner accepts an explicit `--ngram-size` for draft-proposer
experiments while keeping the target reference fixed. A one-run-per-prompt
screen at sizes 2 and 4 matched tokens/KV and had the same accepted draft
counts as size 3, but the repeats are too few to choose a new proposer size.
See the [screen artifact](triton_nf4_group_scale_packed_k256_k2_ngram_size_screen_v1.json).

### Packed-byte K256 at K=1/R=2

K=1/R=2 has a separate contract from K=2/R=3
(`2dd35565577e8c0e2bc4f6b1dd6c4ef1e257dba15bcd488a8cfea0ecabb7ca67`).
The M=1…2 probes on `cnndm_001` and `cnndm_002` matched logits and KV
bitwise. Three independent process sessions per prompt matched B1 and BNB
token IDs in all six runs and shared all 32 committed KV prefixes. The
[contract repeat audit](triton_nf4_group_scale_packed_k256_k1_reference_repeat_audit_v1.json)
passes for both prompt groups.

The [paired summary](triton_nf4_group_scale_packed_k256_k1_paired_multisession_summary_v1.json)
shows geometric-mean block latency of 1.018× K1 singleton (range
0.944–1.065×; diagnostic paired log-t 95% interval 0.974–1.064×) and 1.247×
original BNB (range 1.143–1.336×; interval 1.176–1.324×). Since the singleton
interval crosses 1.0, this sample cannot establish a speedup or slowdown. It
is the nearest measured setting to break-even, but it is still about 25%
slower than BNB. The mean candidate work was 35.5 target rows for 32 output
tokens, with 0.4 draft acceptance. These measurements motivate reducing K1
verification overhead before another K sweep; they are finite diagnostics on
two development prompts and one GPU.

A one-run-per-prompt n-gram screen at n=2 and n=4 also matched B1/BNB tokens
and all 32 KV prefixes. Acceptance stayed at 4/9 for `cnndm_001` and 2/6 for
`cnndm_002`, the same counts as n=3, so this screen does not justify changing
the proposer. See the [screen artifact](triton_nf4_group_scale_packed_k256_k1_ngram_screen_v1.json).

A paired PyTorch CUDA trace on `cnndm_001` counted 6,804 packed NF4 kernel
launches for block (27 target forwards) and 7,812 for singleton (31 forwards).
Block nevertheless processed 36 target rows versus 31 and accumulated 585.3
ms packed-kernel duration versus 518.3 ms in the trace. The profiler trace is
perturbed and represents one prompt; its useful finding is that extra
rowwise quantized-linear work offsets fewer launches. See the
[profile comparison](triton_nf4_group_scale_packed_k256_qwen3b_cnndm001_k1_rows2_v1/profile_comparison_summary.json).

### Packed-byte K256 with two-row CTA reuse

The experimental strategy `triton_nf4_group_scale_packed_rows2_k256` uses the
original packed K256 kernel for M=1 and shares one packed-weight decode across
two rows for M=2. The row reductions are kept separate. On q_proj, M=2 took
0.658× the packed K256 time and was bitwise equal to it; the two-row prototype
itself took 1.191× the packed kernel at M=1, so dispatch keeps the original
kernel for singleton input. Whole-model probes on
`cnndm_001` and `cnndm_002` were bitwise equal in logits and KV for M=1/2.

The strategy's distinct contract is
`40df82dc76c21bd085eff4836559fe22348a3d5797d5c9193489ebe258c46660`.
Three paired process sessions per prompt matched B1 and BNB tokens and all 32
committed KV prefixes in all six runs. Block latency was 0.941× singleton
(diagnostic paired-log 95% interval 0.908–0.975×) and 1.145× original BNB
(interval 1.097–1.196×). It improves K1 singleton latency in this sample but
does not yet beat BNB or reach the 1.15× speedup design target. Acceptance
remained 18/45. See the [paired summary](triton_nf4_group_scale_packed_k256_rows2_k1_paired_multisession_summary_v1.json)
and [repeat audit](triton_nf4_group_scale_packed_k256_rows2_reference_repeat_audit_v1.json).

The third development prompt `cnndm_000` also passed M=1/2 operator probes
and three 32-token pilots. Across all nine runs on `cnndm_000/001/002`, block
latency was 0.894× singleton (diagnostic paired-log 95% interval
0.838–0.953×) and 1.093× original BNB (interval 1.028–1.163×). All nine runs
matched both token baselines and 32/32 committed KV prefixes. The overall
acceptance rate was 42/81 (51.9%), with the added prompt at 24/36. The
three-prompt summary and contract audit are
[here](triton_nf4_group_scale_packed_k256_rows2_k1_three_prompt_summary_v1.json)
and [here](triton_nf4_group_scale_packed_k256_rows2_reference_repeat_three_prompt_v1.json).

### Packed-byte K256, N=16 two-row schedule

The N=16 tile has a distinct reference schedule from the N=8 two-row
contract. On q_proj, it changed the N=8 output slightly (M=2 max absolute
difference about `1.9e-6`), but each row was bitwise identical to its
sequential singleton N=16 result. Whole-model probes on all three development
prompts matched logits and KV at M=1/2. Its contract ID is
`ac30b8a2be89e329102ad7ae6a29e383ca8f2d43224fdf759bdc079e95b50d32`.

A 15-case synthetic stress covered K=64…1024, output tails around N=16/32,
nested and non-nested NF4 scales, and FP16 bias. Both rows at M=2 matched
their own singleton bitwise in all cases; repeated-row outputs also matched.
Every singleton case happened to match BNB dequantization followed by FP16
`F.linear` bitwise. This finite aligned-shape sweep does not establish
universal equivalence. See the [stress artifact](triton_nf4_group_scale_packed_rows2_n16_k256_synthetic_stress_v1.json)
and [harness](../../scripts/stress_nf4_packed_rows2_n16.py).

Nine 32-token runs matched both B1 and BNB tokens and all 32 committed KV
prefixes. Block latency was 0.882× singleton (diagnostic paired-log 95%
interval 0.831–0.936×) and 1.082× original BNB (interval 1.015–1.154×).
Acceptance was 42/81. Against matched N=8 prompt/session slots, full-model
block latency was 0.989×; thus the N16 tile's q_proj gain did not create a
material end-to-end gain over N8 in this sample. One profiler trace found the
packed kernel duration decreased from 585.3 to 382.3 ms, but the profiler
total is perturbed and is not a speed result. See the [three-prompt summary](triton_nf4_group_scale_packed_rows2_n16_k1_three_prompt_summary_v1.json),
[N8 comparison](triton_nf4_group_scale_packed_rows2_n16_vs_n8_three_prompt_comparison_v1.json),
[profiler report](triton_nf4_group_scale_packed_rows2_n16_qwen3b_cnndm001_profile_comparison_v1.json),
and [repeat audit](triton_nf4_group_scale_packed_rows2_n16_reference_repeat_three_prompt_v1.json).

A one-launch Nsight Compute basic profile measured 27.07 μs for q_proj and
131.04 μs for down_proj. They used 64 and 72 registers/thread, respectively;
register allocation limited theoretical occupancy to 66.7% and 58.3%, while
achieved occupancy was 48.1% and 46.1%. Both compute and DRAM throughput were
material in these launches. A separate local-memory counter run observed zero
local-memory sectors in both launches, with no runtime spill traffic observed
for these shapes; this does not cover other shapes or compiler configurations.
The report is a kernel diagnostic, not a latency benchmark; see the [summary
and raw reports](ncu_nf4_rows2_n16_v1/summary.json).

A diagnostic K-tile reduction to K=128 lowered register use (q_proj 64→48,
down_proj 72→56) and raised theoretical occupancy, but M=2 event medians were
1.230× and 1.356× the K=256 schedule, respectively. The achieved occupancy
gain was only 2–3 percentage points. Although finite M=1/2 operator checks
matched each schedule's own singleton rows, K=128 was not integrated; it has a
different arithmetic schedule and lacks whole-model validation. See the
[screen summary](triton_nf4_packed_rows2_n16_k128_exploration_v1/summary.json)
and [operator benchmark](triton_nf4_packed_rows2_n16_k128_exploration_v1/operator_benchmark.json).

An isolated N=16/K=256 one-warp schedule has a separate reference contract
(`49c5f6a606a9ef8c32ec8d39b9d6890971834762ac4bbee58ec0b8d60475afbd`). Nine
32-token development runs matched B1 and BNB tokens, and all 288 shared KV
prefix states matched. Its geometric-mean speedup was 1.142× over its own
singleton path and 0.916× relative to original BNB. Six matched slots against
the four-warp schedule showed 0.923× block latency and 0.942× singleton
latency. These are finite results, and the changed warp count defines a
different compiled schedule. NCU measured 166–168 registers/thread, 25%
theoretical and about 12% achieved occupancy, with zero local-memory sectors
for the two profiled projections. Keep it opt-in; it misses the 1.15× target
and remains slower than BNB on average. See the [pilot summary](triton_nf4_packed_rows2_n16_warp_sweep_v1/pilot_summary.json),
[NCU/operator summary](triton_nf4_packed_rows2_n16_warp_sweep_v1/summary.json),
and [reference contract](triton_nf4_packed_rows2_n16_warp1_k256_qwen3b_cnndm000_k1_rows2_session0_v1/reference_contract.json).

A single paired K=2/R=3 seed on `cnndm_000` also matched both token
references and all 32 shared KV prefixes. Its acceptance was 11/18 (61.1%),
below the K=1/R=2 seed's 8/12 (66.7%); speedup was 1.194× versus B1 and
0.940× versus BNB. This does not support expanding K=2, and it has a separate
physical-row contract
(`31bb7c874fe13b57a72af11f7cdbecc1b30b227ecb360828d98b31de23cdae46`). See
the [paired seed comparison](triton_nf4_packed_rows2_n16_warp_sweep_v1/k1_vs_k2_cnndm000_seed.json)
and [K=2 contract](triton_nf4_packed_rows2_n16_warp1_k256_qwen3b_cnndm000_k2_rows3_session0_v1/reference_contract.json).

On a diagnostic `cnndm_001` profile, warp=1 lowered packed-kernel self time
4.3% and verify time 6.7% at the same 6,804 packed launches; the profiler
total fell 6.1%. These trace deltas are not latency benchmarks. The N=8/16/32
warp=1 tile screen retained N=16 because N=8 did not consistently help across
q_proj/down_proj and N=32 was much slower. See the [profile comparison](triton_nf4_packed_rows2_n16_warp_sweep_v1/profile_comparison_cnndm001.json)
and [tile summary](triton_nf4_packed_rows2_n16_warp_sweep_v1/output_tile_summary.json).

A `num_stages=1/2/3` operator sweep found no M=2 latency change beyond about
1%. NCU showed the same 168/166 registers per thread for stage 1 and stage 3
on q_proj/down_proj, with only marginal occupancy changes and zero
local-memory sectors. It did not lower register pressure or justify another
whole-model reference. See the [stage summary](triton_nf4_packed_rows2_n16_warp_sweep_v1/pipeline_stage_summary.json)
and [operator sweep](triton_nf4_packed_rows2_n16_warp_sweep_v1/pipeline_stage_benchmark.json).

A separate K-tile screen for the one-warp N=16 schedule tested K=64/128/256.
All passed M=1/2 singleton equality. At M=2, K=128 was 1.010× K=256 on
q_proj and 1.052× on down_proj; K=64 was 1.006× and 1.136×, respectively.
Keep K=256. This operator-only result does not support a new contract or
whole-model run. See the [summary](triton_nf4_packed_rows2_n16_warp1_k256_k_tile_sweep_v1/summary.json)
and [raw benchmark](triton_nf4_packed_rows2_n16_warp1_k256_k_tile_sweep_v1/operator_benchmark.json).

The experimental `triton_nf4_split8_downproj_warp1_k256` schedule uses a
fixed 8-way K partition and deterministic second-pass partial reduction only
for Qwen2.5-3B's 11008→2048 down projections; other supported projection
shapes use the N=16/warp1 schedule. All 36 down projections passed finite
M=1/2 singleton equality. The 9 whole-model development pilots matched B1 and
original BNB token IDs, plus 288/288 shared KV states, but achieved only
1.126× B1 and 0.888× BNB geometric-mean latency ratios. A direct same-process
paired comparison across 3 prompts × 3 repeats measured split8 at 1.038× the
warp1-only block latency (prompt-level diagnostic interval 1.018–1.058), with equal output
tokens in every pair. Retain it as an opt-in correctness prototype; reject it
as the current performance schedule. See the
[summary](triton_nf4_packed_rows2_splitk_v1/summary.json), [paired A/B](triton_nf4_packed_rows2_splitk_v1/paired_warp1_vs_split8.json),
[layer audit](triton_nf4_packed_rows2_splitk_v1/down_layers_shape_audit.json),
and [pilot artifacts](triton_nf4_packed_rows2_splitk_v1/).
Reusing the partial buffer changed down_proj M=2 latency by less than 0.2%;
two/four warps were 5.2%/15.6% slower than one warp. Stop tuning this branch
and treat a different reduction mapping as required for further work. See the
[workspace benchmark](triton_nf4_packed_rows2_splitk_v1/operator_workspace_benchmark.json)
and [warp sweep](triton_nf4_packed_rows2_splitk_v1/operator_warp_sweep.json).

A separate BNB fixed-row K2/R3 budget screen covered 3 CNN/DailyMail prompts ×
3 sessions at 64 and 256 output tokens. All 18 runs matched their B1
singleton and 2,880/2,880 shared KV prefixes. At 64 tokens, BNB tokens matched
9/9 but latency was 0.908× BNB; at 256 tokens, BNB tokens matched 0/9 while
B1 speedup was 1.397× and BNB-relative geomean was 1.024× (prompt-level
diagnostic interval 0.951–1.102). This supports a long-generation B1 use case,
not B2 equivalence or a settled BNB speedup. See the [budget summary](qwen3b_bnb_fixedrows_k2_r3_cnndm3_256_v1/summary.json)
and [raw sessions](qwen3b_bnb_fixedrows_k2_r3_cnndm3_256_v1/).

A forced replay of the saved BNB token paths (identical across all three source
sessions) found bitwise-equal prompt-prefill logits and KV hashes, then a first
decode-step divergence: after consuming generated token 0, fixed-row logits
differed by max absolute 0.25–0.28125 and active KV hashes no longer matched.
The greedy argmax remained the same until generated-token indices 91, 93, and
99 for the three prompts. This pinpoints early numerical/state drift before
token drift. The first operator trace below localized the padded schedule's
onset to layer-0 SDPA. [Replay trace](qwen3b_bnb_fixedrows_k2_r3_cnndm3_256_v1/bnb_logit_kv_replay_v1.json),
[script](../../scripts/audit_b1_bnb_replay_drift.py).

That trace was followed by a module/SDPA audit. The first real-row mismatch in
the padded schedule was Qwen2 layer-0 SDPA: singleton BNB used native KV heads,
one query row, no mask and `enable_gqa=True`; padded verification used three
query rows, repeated KV heads and a padded causal mask. The real-row SDPA output
differed by 0.00024–0.00098. The experimental `bnb_rowwise` schedule now calls
the pinned singleton SDPA layout once per real query row, and evaluates RMSNorm
and `lm_head` at M=1 per row. A first version exposed a second mismatch at
`layers.33.post_attention_layernorm` after 62 tokens; singleton RMSNorm removed
it in the next replay.

The resulting K2/R3 pilot matched original BNB and its own singleton tokens in
6/6 runs over three CNN/DailyMail development prompts × two process sessions
at 256 tokens. All 1,536 shared B1 KV prefixes matched; a separate forced BNB
singleton replay recorded no logit or active-KV mismatch through 256 tokens
for all three prompts. Paired geometric-mean speedup was 1.227× versus B1
(prompt bootstrap interval 1.161–1.305) and 0.944× versus original BNB
(0.886–1.010). The BNB-relative result includes 1 and remains inconclusive.
This is finite runtime-scoped evidence, not a universal operator proof. See the
[multisession summary](qwen3b_bnb_rowwise_k2_r3_cnndm3_256_multisession_summary_v1.json),
[reference contract](qwen3b_bnb_rowwise_k2_r3_cnndm3_256_v1/reference_contract.json),
[module trace](qwen3b_bnb_fixedrows_k2_r3_cnndm3_256_v1/first_decode_module_trace_v4.json),
and [full BNB replay](qwen3b_bnb_rowwise_k2_r3_cnndm3_256_v1/bnb_singleton_state_replay_v1.json).

A K4/R5 screen showed the row-count limit of the current B2 schedule. With
physical_rows=5, original BNB and the fixed-row singleton already differed at
the first decode step; the first differing module was layer-0 `q_proj`. The
K4/R5 block still matched its own B1 singleton in 3/3 runs and all 768 shared
prefixes, but matched BNB 0/3 and reached only 0.818× BNB-relative latency.
Keep K4/R5 out of the B2 capability until each quantized-linear row can follow
the same arithmetic as original BNB M=1. [Screen summary](qwen3b_bnb_rowwise_k4_r5_cnndm3_256_v1/summary_screen_v1.json),
[first operator trace](qwen3b_bnb_rowwise_k4_r5_cnndm3_256_v1/first_decode_operator_trace_v1.json).

A pairwise-nibble prototype also passed finite singleton-row checks but had no
stable latency benefit. Its initial M=2 ratios were 1.017× q_proj and 1.008×
down_proj for split reductions, and 1.016× and 1.051× for fused reductions.
A follow-up 1/2/4-warp sweep regressed on down_proj in every case; the lone
q_proj M=2/warp=1 ratio of 0.972× was not consistent with the earlier run.
NCU measured 246 registers/thread for the split variant, above the 168/166
baseline, and outputs differed by up to 0.00195. No new reference contract or
whole-model run was justified. See the [pairwise summary](triton_nf4_packed_rows2_pairwise_dataflow_v1/summary.json)
and [warp sweep](triton_nf4_packed_rows2_pairwise_dataflow_v1/operator_benchmark_warp_sweep.json).

## Additional long-prefix probes

The first long prompt (`cnndm_000`, 397 tokens) was checked under three separate contracts: [`rows=3`](b1_reference_contract_v3.json), [`rows=5`](b1_reference_contract_rows5.json), and [`rows=9`](b1_reference_contract_rows9.json). The fixed-row probes report bitwise-equal logits and active KV for M=1..3, M=1..5, and M=1..9, respectively. At M=2, captured module and attention tensors had no differences. See the [rows=3 probe](b1_operator_probe_long_cnndm_rows3_v1.json), [rows=5 probe](b1_operator_probe_long_cnndm_rows5_v1.json), and [rows=9 probe](b1_operator_probe_long_cnndm_rows9_v1.json).

A second long development prompt (`cnndm_001`, 398 tokens) was checked under the same distinct contracts. Again, logits, active KV, and argmaxes matched bitwise for M=1..3/5/9, with no captured module or attention tensor differences at M=2. See the [rows=3 probe](b1_operator_probe_cnndm001_rows3_v1.json), [rows=5 probe](b1_operator_probe_cnndm001_rows5_v1.json), and [rows=9 probe](b1_operator_probe_cnndm001_rows9_v1.json). These are finite same-prefix observations across two development inputs; they do not close the universal proof obligations below. The three row counts use distinct reference IDs.

## State equality and rollback

The target state relevant to the next transition includes every layer's
active K/V bytes, logical cache lengths, next pending token, position
convention, and any model/backend state that affects a later call. The
diagnostic digest hashes active K/V bytes and logical lengths only; it does not
hash unused static-cache suffix bytes or all model/backend state. The updated
runner can hash every committed output-prefix KV slice without changing cache
lengths. On `cnndm_002`, this matched B1 at all 256 recorded prefixes for K=2,
4, and 8 under their separate row-count contracts. This closes the sampled
frontier gap for active KV on that prompt, but it is not a complete state
certificate.

Static-cache crop only reduces logical lengths. Rejected suffix bytes can
remain in allocated storage; the contract relies on subsequent writes
overwriting them before they can become visible to attention. This must be
verified for every rejection position and terminal branch.

## Experimental Triton batched-attention schedule

The opt-in `attention_strategy="triton_batched"` is a distinct, empirical B1
candidate. It uses the prototype Triton online-softmax kernel with one program
per real query/head. The kernel derives the allowed causal prefix for row `i`
as `K_LEN - Q_LEN + i + 1`, so a block call handles all real query rows in
one launch while each row follows the same key-tile arithmetic schedule as a
singleton. The Qwen adapter batches Q/K/V projection and calls this attention
kernel once for the whole block. It pads singleton RMSNorm and `lm_head`
inputs to the configured physical row count, applies each operator at that
fixed M, then crops unused rows. Block calls use the same M when all rows are
real. Unlike the `padded` strategy, attention itself does not add dummy query
rows or an additive mask; it assumes the documented batch-one, unpadded prompt
and active static KV prefix.

After the fixed-M RMSNorm/LM-head update, this prototype matched B1 singleton
tokens and every active-KV prefix on `cnndm_000` at 64 tokens and `cnndm_001`
at 128 and 256 tokens. Across three paired timing rounds on each of the first
three prompt/budget pairs, the descriptive geometric-mean ratios were 1.280x
block/B1 (diagnostic paired-log-t 95% interval 1.219–1.344) and 0.957x
block/original-BNB (0.908–1.008). BNB token IDs matched at 64 tokens on
`cnndm_000`, but first diverged at generated index 28 on `cnndm_001` and 99
on `cnndm_002`. The 256-token state audit has one timing round. These finite
results do not close the proof obligations below or establish a general
speedup. See the
[64-token repeats](triton_b1_parallel_attention_norm_fixed_cnndm000_k2_r3_64_s16_v1.json),
[128-token repeats](triton_b1_parallel_attention_norm_fixed_cnndm001_k2_r3_128_s17_v1.json),
[cnndm_002 repeats](triton_b1_parallel_attention_norm_fixed_cnndm002_k2_r3_128_s22_v1.json),
[256-token audit](triton_b1_parallel_attention_norm_fixed_cnndm001_k2_r3_256_s18_v1.json),
and the [kernel prototype](../../scripts/prototype_fixed_row_sdpa_triton.py).

A later 64-token run also matched the scalar cache sequence length and all
per-layer active lengths at every shared commit frontier, in addition to the
K/V prefix hashes. This is one finite metadata check, not coverage of all
reject/EOS/budget branches: [frontier audit](triton_b1_parallel_attention_fullstate_cnndm000_k2_r3_64_s20_v1.json).

The same attention/RMSNorm/LM-head schedule was separately paired with
`quantized_linear_strategy="triton_nf4_group_scale_packed_rows2_n16_warp1_k256"`.
This changes the linear reference arithmetic and therefore has its own
schedule identity. It matched B1 tokens, KV prefixes, and logical lengths on
three development prompts. Across nine timing pairs it measured 1.199x
block/B1 (diagnostic 95% interval 1.141–1.260) and 1.018x block/original-BNB
(0.959–1.080). The BNB interval crosses parity; original-BNB token paths
diverged at index 45/75 on the two longer prompts. See the [three paired
prompt artifacts](triton_b1_parallel_attention_norm_fixed_warp1n16_cnndm000_k2_r3_64_s23_v1.json),
[cnndm_001](triton_b1_parallel_attention_norm_fixed_warp1n16_cnndm001_k2_r3_128_s24_v1.json),
and [cnndm_002](triton_b1_parallel_attention_norm_fixed_warp1n16_cnndm002_k2_r3_128_s25_v1.json).

An independent process session repeated the 128-token `cnndm_002` case and
again matched all B1 tokens, 128 active KV prefixes, and shared-frontier
logical lengths. Its paired rounds measured 1.067x block/B1 and 0.884x
block/original-BNB; BNB first diverged at generated index 75. Both `cnndm_002`
sessions accepted 19/63 draft tokens (30.2%), versus 41/65 (63.1%) on
`cnndm_001`; their block verify times were 3.47–3.67 s versus 2.83–3.00 s.
This suggests low acceptance and greater verification work contribute to the
prompt-level regression, but does not isolate causality. This fourth session
is separate from the nine-pair aggregate above and confirms that the longer
`cnndm_002` workload remains slower than BNB under this alternate schedule. See
[session 26](triton_b1_parallel_attention_norm_fixed_warp1n16_cnndm002_k2_r3_128_s26_v1.json).

### K=1/R=2 with the N=16/one-warp linear schedule

Changing to `physical_rows=2`, K=1 defines a separate fixed-row B1 schedule.
Across two process sessions on each of three development prompts, the block
matched its own singleton tokens, KV prefixes (576/576 total), and shared
logical-length frontiers. The descriptive geomean over 18 timing pairs was
1.229x block/B1 and 1.014x block/original-BNB. BNB-relative results were
above parity on `cnndm_000` and `cnndm_001`, but below parity in both
`cnndm_002` sessions. The run matched BNB tokens only on `cnndm_000`; the
longer prompts diverged at indices 45 and 75. This finite screen is a
promising B1 candidate, not BNB certification or a general performance claim.
See the six closed-loop artifacts listed in
[FINDINGS.md](FINDINGS.md#k1r2-screen-with-the-n16one-warp-schedule).

The schedule was extended to 256 generated tokens with two sessions on each
of the three prompts. All 1,536 B1 token/KV prefixes matched. Across 18 paired
rounds, the geomeans were 1.291x block/B1 and 1.067x block/original-BNB; both
sessions on every prompt were above BNB parity. A 512-token `cnndm_002`
extension also matched all 512 prefixes in two sessions and measured 1.201x
B1 / 1.009x BNB across six timing pairs. Original-BNB token paths still
diverged on all three prompts (indices 153/45/75). These finite development
results make K=1/R=2 the leading performance candidate, while the separate
reference contract and runtime branch obligations remain open. See
[the 256-token artifacts](FINDINGS.md#k1r2-screen-with-the-n16one-warp-schedule)
and the 512-token runs linked there.

A 32-token R=2 operator trace measured 461.2 ms packed NF4 self CUDA time for
singleton and 371.6 ms for block; the block made 6,300 such launches versus
7,812. M=2 vocabulary projection was 107.6/86.6 ms and fixed-row attention
20.3/18.1 ms. These are profiler event sums with the same event-clear warning
described in the findings, not benchmark latency. The R=2 profile confirms
that NF4 linear remains the main target for further optimization.

The current composite reference identity is
[`b1_triton_batched_k1r2_n16_reference_contract_v3.json`](b1_triton_batched_k1r2_n16_reference_contract_v3.json),
ID `0295c5c92f6e6e6269b2ab3607635db1ab0fd33192351552e9f2232d23fbfc92`.
It pins the loaded weight-state digest, runtime/source hashes, R=2 operator
schedule, and eight full-state runs. Natural traces covered reject-at-0,
accept, no-candidate, and two budget completion paths across 2,026 cycles.
They did not exercise EOS candidate or EOS terminal behavior. The v3 contract
is experimental. A source-bound v4 revision also pins the contract-aware API
and EOS audit runner.

The standalone validator
[`validate_triton_b1_reference_contract.py`](../../scripts/validate_triton_b1_reference_contract.py)
checks the contract digest, saved runtime/model/tokenizer identity, operator
schedule, and all pinned source files. Its first report is
[`triton_b1_contract_validation_v1.json`](triton_b1_contract_validation_v1.json):
the v3 contract ID recomputes correctly and all 12 pinned source files match.
The fail-closed launcher
[`run_triton_b1_contract_audit.py`](../../scripts/run_triton_b1_contract_audit.py)
recomputes the model-state and runtime identity after loading the pinned model,
checks all contract source hashes, and starts decoding only after a pass. Its
first live run is
[`triton_b1_contract_gated_cnndm000_64_s2_v1.json`](triton_b1_contract_gated_cnndm000_64_s2_v1.json),
with a passing preflight report in
[`triton_b1_contract_live_preflight_v1.json`](triton_b1_contract_live_preflight_v1.json).
All 64 generated tokens, KV prefixes, and logical lengths matched the R=2
singleton. One timing pair measured 1.301x vs B1 and 1.097x vs original BNB;
this is a smoke measurement, not a repeatable performance estimate. The
launcher is fail-closed for this audit path. The general-purpose fixed-row
decoder remains available for experiments without a contract, while the new
R=2 contract-bound entry points require a live-verified handle.

The [v4 contract](b1_triton_batched_k1r2_n16_reference_contract_v4.json), ID
`eb6692d1cae1e3144ba99dc7f5b0eae9e4067c2c9efa2a4f142178b02fcd8b33`, additionally
pins the live validator, [contract-bound decoder API](../../src/zassd/decoding/b1_triton_contract.py),
and EOS audit runner. The validator
[`triton_b1_reference_contract.py`](../../src/zassd/certification/triton_b1_reference_contract.py)
hashes the loaded NF4 model state and checks the pinned runtime and sources;
it returns a model-instance-bound handle. The API exposes fixed singleton and
block entry points and does not accept alternate K, row count, attention, or
linear strategies. The [v4 API audit](triton_b1_r2_contract_api_eos_audit_v2.json)
passed both EOS branches with exact B1 tokens, KV prefixes, and logical lengths.

Controlled R=2 EOS audits now cover both outcomes. The accepted-EOS prompt
exercised an EOS draft accepted at candidate index 0 and EOS terminal; tokens,
both shared KV prefixes, and logical lengths matched B1. A second development
prompt exercised an EOS draft rejected at index 0, followed by a later EOS
terminal; tokens, all three shared KV prefixes, and logical lengths also
matched B1. See the [accepted-EOS report](triton_b1_r2_eos_accept_audit_v1.json)
and [rejected-EOS report](triton_b1_r2_eos_reject_audit_v1.json). This is
finite evidence, not a proof.

A four-prompt stress cohort then exercised two accepted-EOS and two rejected-
EOS cases. Every prompt matched B1 tokens, KV, and logical lengths; the
[combined report](triton_b1_r2_eos_branch_stress_v1.json) records four EOS
draft events. This remains a small, constructed development cohort and is not
an all-input guarantee.

The [v5 contract-bound performance runner](../../scripts/benchmark_triton_b1_contract_api.py)
was then measured on the three development CNN/DailyMail prompts, two paired
rounds each. At 128 tokens, all B1 token/KV prefixes matched (91/91, 97/97,
113/113); geomean block speedup was 1.276x vs singleton (descriptive paired
log-t 95% interval 1.159–1.405) and 1.030x vs original BNB (0.915–1.160).
At 256 tokens, all B1 token/KV prefixes again matched (183/183, 205/205,
205/205); geomean speedups were 1.301x vs singleton (1.063–1.592) and
1.137x vs BNB (1.033–1.252). The 256-token acceptance counts were 74/102,
51/78, and 51/81. Only two of six 128-token pairs matched BNB tokens; none
of the six 256-token pairs did, with first BNB divergences at generated
indices 153, 45, and 75. These intervals are descriptive for three fixed
prompts on one GPU, with repeated timing rounds; they are not independent
prompt-level confidence intervals or quality evidence. See the
[128-token report](triton_b1_r2_contract_api_perf_cnndm3_128_v1.json),
[256-token report](triton_b1_r2_contract_api_perf_cnndm3_256_v1.json), and
[v5 contract](b1_triton_batched_k1r2_n16_reference_contract_v5.json).

To check session repeatability on new prompts, the same 256-token runner was
started in two independent processes on `cnndm_003`, `cnndm_004`, and
`cnndm_005`. All six runs matched B1 tokens, KV, and logical lengths (1,206
shared prefixes total); the two process geomeans were 1.286x/1.255x vs B1 and
1.040x/1.051x vs BNB. The six-pair combined geomeans were 1.271x vs B1
(descriptive log-t 95% interval 1.245–1.297) and 1.045x vs BNB
(1.005–1.088). Draft acceptance was 334/508 overall. Every block sequence
differed from BNB; one prompt-session pair was slightly slower than BNB
(0.989x), so the apparent BNB gain remains prompt-dependent and small. The
[session summary](triton_b1_r2_contract_api_perf_cnndm003_005_256_summary_v1.json)
links and hashes both process reports. Treat the confidence interval as
descriptive: there are only three prompts, repeated in two processes on one
GPU.

The same two-process protocol was run at 512 tokens on the same added cohort.
All six runs remained B1 exact across 2,416 shared prefixes, with 656/1,038
draft acceptance. Combined block speedup was 1.246x vs B1 (descriptive
log-t 95% interval 1.200–1.294) and 1.012x vs BNB (0.960–1.068); the two
process BNB geomeans were 1.010x and 1.014x. No run matched BNB tokens, and
one BNB-relative timing pair fell to 0.927x. Thus the B1 gain persisted at
512, but BNB parity is unresolved. See the [512-token summary]
(triton_b1_r2_contract_api_perf_cnndm003_005_512_summary_v1.json),
[session 0](triton_b1_r2_contract_api_perf_cnndm003_005_512_session0_v1.json),
and [session 1](triton_b1_r2_contract_api_perf_cnndm003_005_512_session1_v1.json).

The repository's current EOS branch fixtures target the legacy singleton
decoder, while the generic Qwen2 capability registry does not bind this
composite contract. Thus the natural closed-loop coverage is useful finite
evidence but does not authorize fail-closed certified-mode selection.

The empirical scope also excludes B2 state equality. The Triton singleton
state differed from native BNB in the earlier one-transition audit, and a
separate 128-token run first diverged from BNB token IDs at generated index
28. Any future speed claim must use a new pinned contract for this schedule
and report both B1 state identity and BNB token/state comparisons separately.

## Proof obligations

To promote the schedule from experimental to certified, establish all of the
following for the declared domain:

1. **Linear kernels:** For the default strategy, each `Linear4bit` and native
   `nn.Linear` must preserve equal real input rows embedded in the fixed R-row
   tensor, with BNB dispatch, strides, accumulation order, rounding, bias, and
   output dtype pinned. For `triton_nf4`, validate packed-code order, nested
   scale reconstruction, dequantization rounding, reduction order, bias, and
   output dtype against an independently constructed reference. Treat the
   Triton/compiler/device tuple as part of that separate reference contract.
2. **RMSNorm and rowwise operators:** Prove real-row invariance under padding;
   account for the Transformers kernel decorator, reduction implementation,
   residuals, activation, final norm, and LM head.
3. **RoPE and attention:** Prove equal positions and Q/K/V bytes at every
   layer. For padded attention, prove that masked padded keys and fixed SDPA
   shapes yield the same real-row output as the singleton reference. Include
   backend dispatch, mask dtype/strides, and cache-length-dependent K shape.
4. **Prefill:** Pin the prefill path and prove it initializes the same first
   pending token and cache state in both B1 methods. The patch can affect
   short prompts whose sequence length is at most R because their prefill
   linear/norm calls also satisfy the padding wrapper's shape condition.
5. **Cache and decoder branches:** Prove active KV equality after each accepted
   row and correct crop/write behavior for every reject location, EOS, and
   output-budget boundary. Compare all layers, not only a sampled prefix.
6. **Runtime guard:** Keep the checks for model type, external masks and
   positions, batch size, training mode, sliding-window configuration, row
   count, and cache type fail-closed across supported call forms. Linear and
   RMSNorm wrappers reject unsupported decode tensor ranks, batches, and row
   counts; only a shared long prefill may use the ordinary unpadded module
   path. Add a
   reliable way to exclude padded token-ID inputs, or keep that restriction
   explicit at the public API boundary. Pin and check runtime versions before
   using the experimental operator patch.
7. **Reproducibility:** Hash model/quantization state, tokenizer/config,
   operator source, package/kernel versions, device and runtime flags into a
   contract ID. Any change requires a new validation and contract ID.

No current result discharges these universal obligations. Existing probes
cover finite M values and prompts. After adding model-input guards, a targeted
M=1..3 probe again matched logits and KV bitwise on the pinned 3B prefix. The
six-prompt K sweep matched B1 tokens in 12/12 prompt-round runs per K and
matched active KV digests at shared cycle frontiers. The additional 256-prefix
audit on `cnndm_002` matched active KV at every recorded prefix for K=2/4/8.
These are finite empirical results. See [the findings](FINDINGS.md) and [the
broader proof obligations](PROOF_OBLIGATIONS.md).
