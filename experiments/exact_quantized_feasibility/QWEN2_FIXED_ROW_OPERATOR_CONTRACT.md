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
   `m <= R`, the wrapper pads the sequence-row axis with zero inputs to R,
   makes the input contiguous, calls the original `Linear4bit`, `nn.Linear`,
   or `Qwen2RMSNorm` module, then slices back to m output rows. Each real token
   row is independent in the mathematical module. It remains to prove that
   the actual kernels preserve each real row bitwise and that padding leaves
   kernel dispatch/reduction behavior fixed. Inputs outside the wrapper's
   shape conditions fall through to the original module rather than failing
   closed.
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

## Proof obligations

To promote the schedule from experimental to certified, establish all of the
following for the declared domain:

1. **Linear kernels:** For each `Linear4bit` and native `nn.Linear`, equal real
   input rows embedded in the fixed R-row tensor produce bitwise-equal real
   outputs for every supported R, context and row position. Pin BNB kernel
   dispatch, strides, accumulation order, rounding, bias, and output dtype.
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
   count, and cache type fail-closed across supported call forms. Add a
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
