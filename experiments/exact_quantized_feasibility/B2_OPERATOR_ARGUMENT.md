# B2 operator-by-operator argument (shape-invariant contract)

**Status:** a structured argument, not a machine-checked proof. Each claim is
tagged **[S]** (follows from the code by construction), **[E]** (supported only
by finite recorded experiments) or **[O]** (open). Nothing here covers other
GPUs, Triton/ptxas versions, bitsandbytes versions or models beyond the ones
listed under *Evidence*.

This instantiates the "alternative zero-error route" of `PROOF_OBLIGATIONS.md`:
every row of a verification block must return bitwise the same output and
resulting cache state as the pinned one-token transition (the original
bitsandbytes M=1 greedy step, called **B2** in the README).

## Setting and notation

State `S_t` = all KV entries `[0, L_t)` of every layer, logical length `L_t`, and
the pending token. The reference transition `R` runs `model(x)` with unmodified
bitsandbytes, `x` of shape `(1,1)`. The B2 block `F` runs `model` on `x_0..x_{R-1}`
(`R <= 3`) under the schedule in `src/zassd/certification/qwen2_padded.py`
(`attention_strategy="bnb_rowwise"`) with the 252 transformer `Linear4bit`
projections replaced by `_fused_nf4_m3_kernel` when the batch has exactly three
rows (`scripts/audit_bnb_nf4_activationvec_block_integration.py`).

**Row-equivalence property (RE).** For a block starting from `S_t` identical to the
reference state, row `i` of every intermediate tensor in `F` equals, bit for bit,
the corresponding tensor of `R` applied to `x_i` from the state `S_{t+i}` reached by
consuming `x_0..x_{i-1}` one at a time, and `F` writes to the cache at position
`L_t+i` the same K/V bytes as `R` would.

## Operator table

| Operator in a decoder step | Argument | Tag |
|---|---|---|
| Token embedding lookup | A gather of row `x_i` from the table; independent of the other rows. | [S] |
| RMSNorm (all norms) | The `bnb_rowwise` schedule evaluates each row with a separate call on a `(1,1,H)` tensor, i.e. the same call the reference makes. | [S] |
| q/k/v/o/gate/up/down projections (252 `Linear4bit`) | See *NF4 kernel*. Row independence is structural; equality with the bitsandbytes M=1 kernel is empirical. | [S]+[E] |
| Bias add (q/k/v) | Inside the Triton kernel (`HAS_BIAS`), added to the per-row fp32 sum, same instruction for every row. | [S]+[E] |
| Rotary embedding | Elementwise multiply/add on per-row `cos/sin` taken at the row's absolute position. IEEE elementwise operations do not depend on how many rows are in the tensor. Vectorized vs scalar PyTorch elementwise kernels were not separately audited. | [S]/[O] |
| KV cache write | `StaticPreallocatedLayer.update` copies the new K/V rows into `[L, L+R)`. Bytes are copied, not recomputed. | [S] |
| Attention | One `F.scaled_dot_product_attention` call per row with `q` of shape `(1,H,1,d)`, K/V truncated to exactly `L_t+i+1` positions, no mask, `enable_gqa=True`; `q/k/v` are `.contiguous()` copies, so shapes and layout match the reference's single-row call (`row_attention`, `bnb_rowwise`). Same dispatch and reduction order follow if the PyTorch SDPA dispatch is a function of shapes/dtypes/strides only. | [S] given that proviso; [E] otherwise |
| Softmax, scale | Inside the SDPA call above. | (as above) |
| Residual adds, SiLU*mul (MLP activation) | Elementwise; same remark as rotary. | [S]/[O] |
| Final norm | As RMSNorm. | [S] |
| LM head | Evaluated row by row with a separate `(1,1,H)` call into the same module the reference uses (`bnb_rowwise` branch of `fixed_linear`). The lazy variant calls it only for rows up to the first rejected draft; skipped rows write no state, so omitting them cannot change `S`. | [S] |
| Argmax | Torch `argmax` on a `(1,1,V)` tensor per row; ties follow the same rule as the reference. | [S] |

## NF4 kernel (`_fused_nf4_m3_kernel`)

Read directly from the source:

* Three accumulators `acc0, acc1, acc2` (one per row) are updated by textually
  identical statements from different activation pointers (`x_ptr`, `x_ptr+K`,
  `x_ptr+2K`). No statement reads another row's accumulator or activations. [S]
* The dequantized weight value (`weight0/weight1`) is computed once per
  (output, lane, pair) and used by all three rows, with the same rounding path
  (`fma` for the nested scale, fp16 rounding of the scale and of the weight). [S]
* Products use `mul.rn.f16x2` (inline PTX, round-to-nearest), converted to fp32
  and accumulated sequentially per lane over groups and pairs; the 32 lanes are
  then combined by a fixed five-step tree (`tl.gather` with shifts 16, 8, 4, 2, 1).
  The order does not depend on the row. The launch uses `enable_fp_fusion=False`,
  so the compiler may not contract multiply-add. [S] (conditional on the compiler
  honoring the written program order and `enable_fp_fusion=False`; **[O]**: not
  checked at PTX/SASS level for each Triton version)
* Consequence (**RE-linear**): for any rows `x_0, x_1, x_2` the kernel returns for
  row `i` exactly `f(x_i, W)`, where `f` does not depend on `i` or on the other
  rows. [S]
* **Equality with the reference.** Read at source level (bitsandbytes 0.50.0,
  `csrc/kernels.cu`, `kgemm_4bit_inference_naive<half,128,16>`): each lane accumulates its
  32 fp16 products **one at a time** (`local_C += (float)(local_A[k] * local_B[k])`), then
  `cub::WarpReduce::Sum`. The Triton kernel instead adds each adjacent pair of products first
  (`acc + (p0 + p1)`). Under strict IEEE semantics these association orders differ.
  The shipped binary does **not** behave like the source-level sequential order:
  `scripts/emulate_nf4_gemv_orders.py` (`nf4_gemv_order_emulation_v1.json`) emulates both
  orders in fp32 on random heavy-tailed fp16 activations (layers 0, 9, 18, 27, 35, all seven
  projections, 28,672,000 outputs). The real BNB M=1 kernel and the real Triton kernel both
  match the **pairwise** emulation in 28,672,000 / 28,672,000 outputs and both differ from the
  **sequential** emulation in 398 outputs (the two emulations differ in 398 fp16 outputs and
  in 394,422 fp32 lane sums). The test therefore can distinguish the two orders and it
  selects the pairwise one for both implementations. With zero BNB-vs-Triton mismatches in
  this family of inputs, the one-sided 95% upper bound on the per-output mismatch rate is
  about 1.0e-7 (rule of three), in addition to the 252-module / 9,289,728-element audit on real
  activations (`FINDINGS.md`) and the closed-loop checks below. [E]

  Not established: *why* the binary is pairwise (compiler transformation of the source,
  a different build flag, or a different source than the tag read), whether this holds for
  other GPU architectures or bitsandbytes builds, and the tail-group behaviour for every `K`
  (covered here by `K = 2048, 11008` only; the 0.5B/1.5B closed-loop audits exercise
  `K = 896, 1536, 4864, 8960`). Because the source reading disagrees with the binary, any
  future bitsandbytes upgrade must rerun the emulation script before the B2 claim is trusted.

## From RE to token equality

Let `I(t)`: the committed cache equals the reference cache `[0, L_t)` bitwise and the
pending token equals the reference's next greedy token.

1. **Block rows.** Assuming RE, the logits of row `i` equal the reference logits
   after consuming `x_0..x_i`; the cache bytes at `L_t+i` equal the reference's. [from RE]
2. **Acceptance.** Let `j` be the first position where `argmax(row j) != draft_j`
   (or none). Rows `0..j-1` agreed with the drafts, so the emitted tokens
   `x_0, d_0..d_{j-1}` equal the reference greedy sequence, and the pending
   token `argmax(row j)` is the reference's next token. [S]
3. **Rollback.** `crop(L_t + j + 1)` lowers the logical length. `update` writes at
   `seq_len` and returns only `[0, seq_len)`, so entries beyond the logical
   length are never read before being overwritten. The retained entries
   `[0, L_t+j+1)` are the reference's. Hence `I(t+j+1)`. [S]
4. **Fallback cycle.** A cycle with no draft runs the unmodified model with the
   schedule disabled, i.e. the reference transition itself. [S] (the swap of
   per-module `forward` between plain and patched, `SwitchableSchedule`, is
   covered by a CPU unit test and by the per-step KV audits.)
5. **EOS and budget.** Handled by the same rule as the canonical loop; the
   canonical loop's final-cache-length convention is not asserted by the audits
   (they compare non-terminal prefixes only). [E]

Induction on cycles gives: **if RE holds for every operator, every supported
`R <= 3` and every prefix length reached, the B2 loop emits exactly the
reference token sequence.** RE is the only non-structural hypothesis; its
operator-level status is the table above.

## Evidence (all development-only unless stated)

| Claim | Evidence | Scope |
|---|---|---|
| Operator outputs equal bitwise | `bnb_simt_full_operator_audit_summary.json` (252 modules, 9.29M elements) | 3 prompts |
| One block equals three singleton steps (logits, KV) | `activationvec_singleton_identity_summary_v1.json` | 2 prompts |
| Rejection, rollback and next step | `activationvec_reject_path_*`, `lazy_reject_path_*` (5 scenarios) | 3 distinct prompts on 3B, 1 each on 0.5B, 1.5B |
| Tokens and KV prefix hashes at every non-terminal step of closed-loop generation | `activationvec_fallback_kv_closed_loop_*`, `lazy_lm_head_*`, `eos_kv_long_v1.json` | about 90 prompt runs on 3B (up to 384 tokens, 17 EOS-ending), 4 on each of 0.5B, 1.5B |
| Tokens vs original BNB greedy | `b2_fallback_vs_b1_*`, `heldout_gsm8k_300_349_*` (50/50) | see files |

## What this does not establish

* Equality of the kernel with the bitsandbytes reduction order on all inputs: only measured (28.7M random outputs, 9.29M real), and the CUDA source read literally predicts a different order than the binary shows [O].
* Independence of the PyTorch SDPA/elementwise dispatch from the number of rows
  in neighbouring calls and from allocator state [O].
* Stability across Triton, ptxas, CUDA, PyTorch, bitsandbytes versions and GPUs
  (one RTX 4050 Laptop GPU, Triton 3.6.0, PyTorch 2.11.0, bitsandbytes 0.50.0) [O].
* Anything about sampling, batch size > 1, other quantization schemes, or
  `R > 3`.

A violation of RE at any operator would show up as a divergent token or KV hash
in the audits above; the audits are the safety net for the **[E]** and **[O]**
rows, not a replacement for closing them.
