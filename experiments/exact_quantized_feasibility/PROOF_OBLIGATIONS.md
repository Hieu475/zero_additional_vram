# Conditional whole-sequence theorem and the missing kernel contract

**Status:** mathematical reduction, not a proof that the current BNB, HQQ or
llama.cpp implementations satisfy its hypotheses. In particular, the measured
BNB differences make a zero-error bound false for the current batched path.

## Exact reference

Fix model weights and quantization state, backend and version, GPU architecture,
attention implementation, prefill procedure, cache type and capacity, position
and mask conventions, greedy tie rule, and EOS rule. Let the reference transition
`R(S_t, x_t) = (z_t, S_{t+1})` consume one token using the pinned one-token
path. `S_t` contains **every** item that can influence a later forward call,
including all KV entries, logical lengths, positions, masks and backend state.
The prompt prefill is a fixed initialization of `S_0` and the first pending
token. "Exact" means equality of generated token IDs with this reference, not
equality with an ideal real-arithmetic or FP32 model.

For the fast verifier, `F_M` consumes at most `M <= K+1` candidate tokens.
After committing a prefix of those tokens, its state must have a defined
projection `P_i(F_M(...))` for the state after candidate `i`; cropping allocated
KV storage alone is insufficient unless the resulting logical state satisfies
the bound below.

## Sufficient contract

At a shared token prefix, let `E_t` bound the distance between the fast and
reference states in a specified norm. For every possible state and candidate
prefix **within a stated operating domain**, the implementation must provide
upward-rounded constants `a_i,b_i,c_i,d_i >= 0` such that:

1. `||z_i^F - z_i^R||_infinity <= a_i E_i + b_i` for the corresponding next-token
   logits, where both executions have consumed the same candidate token.
2. `||P_i(S^F) - S_{i+1}^R|| <= c_i E_i + d_i` after committing that token.
3. The projection and inequalities cover every accepted prefix length, batch
   size, rejection point, cache crop, and EOS/budget branch that can occur.

The constants may depend on public dimensions, fixed weights, verified input
intervals and kernel schedule. They cannot be selected from a measured maximum
over an evaluation set. A bound for quantized linear layers alone is not enough:
attention, residuals, normalizations, position transforms, cache writes and
downstream effects must be included. If the bound is unavailable or the state
leaves its stated domain, the fast path must fail closed to a canonical state.

For a computed fast logit vector `z_i^F`, let `m_i` be its top-one minus
top-two gap, respecting the reference tie rule. If
`m_i > 2(a_i E_i + b_i)`, the fast argmax equals the reference argmax. After
committing the certified token, set `E_{i+1} = c_i E_i + d_i`. The update must
be performed even when the current top-one token agrees empirically with a
reference sample; otherwise latent KV drift is silently discarded.

## Whole-sequence induction

Assume the first pending token comes from the reference prefill and `E_0=0`.
At the start of each cycle, maintain: (I) emitted tokens equal the reference
prefix, (II) the pending token is the reference's next greedy token, and (III)
the stored state error is at most `E_t`. For each certified verifier row, the
margin inequality proves the next token, and the transition inequality
preserves (III). At the first uncertified row, discard every subsequent fast
prediction; reconstruct **the entire reference state** from a certified
checkpoint, then consume the relevant tokens with the reference transition
before selecting any next pending token. This restores `E=0` and all three
invariants. Thus induction proves token equality through EOS or the output
budget. Replay performed *after* committing an uncertified pending token does
not satisfy this induction.

This theorem is conditional on bounds that cover the actual finite-precision
kernel implementations. The helper in `src/zassd/certification/greedy.py`
implements only the strict margin comparison and abstract state recurrence.
There is currently no production caller supplying validated BNB/HQQ bounds.

## Alternative zero-error route

Instead of finite nonzero bounds, establish a shape-invariant operator
contract: every row of a verification block, with the same logical prefix and
candidate history, returns bitwise-identical logits and resulting relevant
state to the pinned one-token transition. This requires the dequantization,
reduction order, attention and cache-update schedule to match for all supported
`M` and prefix lengths. Then `b_i=d_i=0` and `E_i=0` by induction. A finite
collection of empirical equality tests is still only validation of an
implementation, not the proof of this universal contract.

## Prototype: fixed-row Qwen2 schedule

The implementation-specific contract, including shapes and known enforcement
gaps, is documented in
[`QWEN2_FIXED_ROW_OPERATOR_CONTRACT.md`](QWEN2_FIXED_ROW_OPERATOR_CONTRACT.md).

`src/zassd/certification/qwen2_padded.py` now implements a research prototype
for batch size one and unpadded Qwen2 prompts. It pins physical M for eligible
linear/RMSNorm calls (default 9) and pads unused rows. It provides two distinct
attention schedules: `rowwise` truncates K/V to each query's causal prefix and
uses unmasked SDPA per row; `padded` uses fixed Q/K/V row shapes and an
explicit additive causal mask in one SDPA call. The reference and block paths
must use the same M, attention strategy, model/backend/kernel versions,
prefill and static-cache contract. The block path commits only a verified
prefix and crops speculative suffix KV. The context manager now rejects
non-Qwen2 configs, external masks/positions, non-batch-one attention calls,
non-static decode caches, training, sliding-window layers, and returned
attention weights. Padding encoded directly in a caller-supplied token tensor
cannot be inferred reliably and remains a public-input restriction.

The finite-width proof still requires these obligations:

1. Show each wrapped linear kernel uses the same dispatch, strides, reduction
   tree and rounding for every logical row after padding; establish that rows
   are independent inside the fixed physical M call. This must cover BNB
   `Linear4bit` and every native `Linear` in the model, including LM head.
2. Show padded `Qwen2RMSNorm` leaves each real row bitwise unchanged and that
   all unwrapped operations between rowwise operators act independently on
   rows. Audit RoPE positions and final normalization.
3. For `rowwise`, show each block row receives exactly the same Q/K/V bytes,
   contiguous layout, logical K length, scale, dtype and no-mask SDPA call as
   its singleton counterpart. For `padded`, show the fixed Q/K/V shapes and
   additive mask reproduce the singleton row result despite masked padded keys.
   Both strategies rely on batch=1 with no padding; arbitrary user masks,
   packed sequences and sliding-window attention are outside the domain.
4. Show cache update writes the same per-token K/V bits and cache crop preserves
   the accepted prefix at every reject, EOS and budget boundary. Prefill must
   initialize identical state under both paths. Finite traces cover rejects
   at all draft positions for K=2/4/8, pending-token EOS, output-budget
   completion, and one accepted EOS-draft candidate. They do not exhaust cache
   crops or all reject/EOS/budget combinations; these remain proof obligations.
5. Pin the physical row count in the reference ID. Changing from rows=5 to
   rows=9 can change the quantized reference trajectory; K may vary only up to
   the pinned maximum without changing that reference.

Initial Qwen2.5-0.5B and 3B NF4 probes show bitwise logits/KV for M=1..9 at
one prefix each. Two-round 256-token K=2/4/8 pilots on six prompts show B1 token
matches 12/12 and active-cache hashes equal at shared cycle frontiers for each
K. These results validate the prototype on sampled executions; they do not
discharge the universal operator proof above. BNB greedy differs on many
prompts, so this establishes neither the B2 claim nor a proof for the complete
supported domain.

## Current evidence and gate

The runner now records a versioned `contract_id`, runtime/backend flags,
tokenizer/config/source hashes and GPU driver identity where available. The
packed quantized-state hash remains absent. The three-model [operator audit](FINDINGS.md#single-prefix-linear-shape-probe)
records equal primary inputs but different BNB NF4 `q_proj` outputs when the
batch length changes. A shape-invariant replacement must cover at least that
operator and the downstream operators/state. The first engineering gate is a
small, fully specified kernel schedule with either a formal arithmetic error
bound or identical per-row execution; benchmark expansion before this gate
would only collect more examples of empirical agreement or disagreement. The
fixed-row prototype clears the initial implementation gate on its sampled
Qwen2 scopes; next work is the operator proof, state stress across more
contexts, and kernel optimization before frozen evaluation.
