import triton
import triton.language as tl

@triton.jit
def _ldcs_v4(ptr, valid):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred p;
            setp.ne.u32 p, $5, 0;
            mov.b32 $0, 0;
            mov.b32 $1, 0;
            mov.b32 $2, 0;
            mov.b32 $3, 0;
            @p ld.global.cs.v4.u32 {$0, $1, $2, $3}, [$4];
        }
        """,
        constraints="=r,=r,=r,=r,l,r",
        args=[ptr, valid.to(tl.uint32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _shfl_pair(value):
    return tl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, 1, 0x1f, -1;",
        constraints="=r,r", args=[value], dtype=tl.uint32,
        is_pure=True, pack=1,
    )

@triton.jit
def _fused_nf4_m3_scale_reuse_kernel(
    x_ptr, packed_ptr, absmax8_ptr, state2absmax_ptr, state2code_ptr,
    offset_ptr, codebook_ptr, bias_ptr, y_ptr,
    K: tl.constexpr, N: tl.constexpr, BLOCKSIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr, OUT_TILE: tl.constexpr = 4,
):
    outs = tl.program_id(0) * OUT_TILE + tl.arange(0, OUT_TILE)[None, :, None]
    lanes = tl.arange(0, 32)[None, None, :]
    acc0 = tl.zeros((1, OUT_TILE, 32), dtype=tl.float32)
    acc1 = tl.zeros((1, OUT_TILE, 32), dtype=tl.float32)
    acc2 = tl.zeros((1, OUT_TILE, 32), dtype=tl.float32)
    offset = tl.load(offset_ptr).to(tl.float32)
    block_log2: tl.constexpr = 6
    blocks_per_row: tl.constexpr = K // BLOCKSIZE

    for group in range(tl.cdiv(K, 1024)):
        inner_k = group * 1024 + lanes * 32
        scale_idx = outs * blocks_per_row + (inner_k >> block_log2)
        valid = (outs < N) & (inner_k < K)
        even_lane = (lanes & 1) == 0
        scale_valid = valid & even_lane
        scale_q_even = tl.load(absmax8_ptr + scale_idx, mask=scale_valid, other=0).to(tl.int32)
        scale_q_peer = _shfl_pair(scale_q_even.to(tl.uint32))
        scale_q = tl.where(even_lane, scale_q_even.to(tl.uint32), scale_q_peer).to(tl.int32)
        scale_code_even = tl.load(state2code_ptr + scale_q_even, mask=scale_valid, other=0).to(tl.float32)
        scale_code_peer = _shfl_pair(scale_code_even.to(tl.uint32, bitcast=True))
        scale_code = tl.where(even_lane, scale_code_even.to(tl.uint32, bitcast=True), scale_code_peer).to(tl.float32, bitcast=True)
        scale_state_even = tl.load(state2absmax_ptr + (scale_idx >> 8), mask=scale_valid, other=0).to(tl.float32)
        scale_state_peer = _shfl_pair(scale_state_even.to(tl.uint32, bitcast=True))
        scale_state = tl.where(even_lane, scale_state_even.to(tl.uint32, bitcast=True), scale_state_peer).to(tl.float32, bitcast=True)
        scale_f = tl.fma(scale_code, scale_state, offset)
        scale_h = scale_f.to(tl.float16)

        # Match BNB's 16-byte streaming load at each (output, lane) position.
        word_base = outs * (K // 8) + group * 128 + lanes * 4
        packed0, packed1, packed2, packed3 = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .pred p;
                setp.ne.u32 p, $5, 0;
                mov.b32 $0, 0;
                mov.b32 $1, 0;
                mov.b32 $2, 0;
                mov.b32 $3, 0;
                @p ld.global.cs.v4.u32 {$0, $1, $2, $3}, [$4];
            }
            """,
            constraints="=r,=r,=r,=r,l,r",
            args=[packed_ptr + word_base, valid.to(tl.uint32)],
            dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
            is_pure=True,
            pack=1,
        )

        # Each lane owns 32 adjacent FP16 activation values. Fetch four
        # aligned 16-byte vectors per physical row; each vector contains four
        # packed half2 words used by the unrolled nibble loop.
        xbase = group * 1024 + lanes * 32
        x0v0, x0v1, x0v2, x0v3 = _ldcs_v4(x_ptr + xbase, inner_k < K)
        x0v4, x0v5, x0v6, x0v7 = _ldcs_v4(x_ptr + xbase + 8, inner_k < K)
        x0v8, x0v9, x0v10, x0v11 = _ldcs_v4(x_ptr + xbase + 16, inner_k < K)
        x0v12, x0v13, x0v14, x0v15 = _ldcs_v4(x_ptr + xbase + 24, inner_k < K)
        x1v0, x1v1, x1v2, x1v3 = _ldcs_v4(x_ptr + K + xbase, inner_k < K)
        x1v4, x1v5, x1v6, x1v7 = _ldcs_v4(x_ptr + K + xbase + 8, inner_k < K)
        x1v8, x1v9, x1v10, x1v11 = _ldcs_v4(x_ptr + K + xbase + 16, inner_k < K)
        x1v12, x1v13, x1v14, x1v15 = _ldcs_v4(x_ptr + K + xbase + 24, inner_k < K)
        x2v0, x2v1, x2v2, x2v3 = _ldcs_v4(x_ptr + 2 * K + xbase, inner_k < K)
        x2v4, x2v5, x2v6, x2v7 = _ldcs_v4(x_ptr + 2 * K + xbase + 8, inner_k < K)
        x2v8, x2v9, x2v10, x2v11 = _ldcs_v4(x_ptr + 2 * K + xbase + 16, inner_k < K)
        x2v12, x2v13, x2v14, x2v15 = _ldcs_v4(x_ptr + 2 * K + xbase + 24, inner_k < K)

        for pair in tl.static_range(16):
            if pair < 4:
                packed_x0 = tl.where(pair % 4 == 0, x0v0, tl.where(pair % 4 == 1, x0v1, tl.where(pair % 4 == 2, x0v2, x0v3)))
                packed_x1 = tl.where(pair % 4 == 0, x1v0, tl.where(pair % 4 == 1, x1v1, tl.where(pair % 4 == 2, x1v2, x1v3)))
                packed_x2 = tl.where(pair % 4 == 0, x2v0, tl.where(pair % 4 == 1, x2v1, tl.where(pair % 4 == 2, x2v2, x2v3)))
            elif pair < 8:
                packed_x0 = tl.where(pair % 4 == 0, x0v4, tl.where(pair % 4 == 1, x0v5, tl.where(pair % 4 == 2, x0v6, x0v7)))
                packed_x1 = tl.where(pair % 4 == 0, x1v4, tl.where(pair % 4 == 1, x1v5, tl.where(pair % 4 == 2, x1v6, x1v7)))
                packed_x2 = tl.where(pair % 4 == 0, x2v4, tl.where(pair % 4 == 1, x2v5, tl.where(pair % 4 == 2, x2v6, x2v7)))
            elif pair < 12:
                packed_x0 = tl.where(pair % 4 == 0, x0v8, tl.where(pair % 4 == 1, x0v9, tl.where(pair % 4 == 2, x0v10, x0v11)))
                packed_x1 = tl.where(pair % 4 == 0, x1v8, tl.where(pair % 4 == 1, x1v9, tl.where(pair % 4 == 2, x1v10, x1v11)))
                packed_x2 = tl.where(pair % 4 == 0, x2v8, tl.where(pair % 4 == 1, x2v9, tl.where(pair % 4 == 2, x2v10, x2v11)))
            else:
                packed_x0 = tl.where(pair % 4 == 0, x0v12, tl.where(pair % 4 == 1, x0v13, tl.where(pair % 4 == 2, x0v14, x0v15)))
                packed_x1 = tl.where(pair % 4 == 0, x1v12, tl.where(pair % 4 == 1, x1v13, tl.where(pair % 4 == 2, x1v14, x1v15)))
                packed_x2 = tl.where(pair % 4 == 0, x2v12, tl.where(pair % 4 == 1, x2v13, tl.where(pair % 4 == 2, x2v14, x2v15)))
            if pair < 4:
                packed_word = packed0
            elif pair < 8:
                packed_word = packed1
            elif pair < 12:
                packed_word = packed2
            else:
                packed_word = packed3
            byte = ((packed_word >> ((pair % 4) * 8)) & 0xFF).to(tl.int32)
            code0 = (byte >> 4) & 0x0F
            code1 = byte & 0x0F
            centroid0 = tl.load(codebook_ptr + code0).to(tl.float16)
            centroid1 = tl.load(codebook_ptr + code1).to(tl.float16)
            weight0 = (centroid0.to(tl.float32) * scale_h.to(tl.float32)).to(tl.float16).to(tl.float32)
            weight1 = (centroid1.to(tl.float32) * scale_h.to(tl.float32)).to(tl.float16).to(tl.float32)

            packed_w = (weight0.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                        | (weight1.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32) << 16))
            prod_pair0 = tl.inline_asm_elementwise("mul.rn.f16x2 $0, $1, $2;", "=r,r,r", [packed_x0, packed_w], tl.uint32, True, 1)
            prod_pair1 = tl.inline_asm_elementwise("mul.rn.f16x2 $0, $1, $2;", "=r,r,r", [packed_x1, packed_w], tl.uint32, True, 1)
            prod_pair2 = tl.inline_asm_elementwise("mul.rn.f16x2 $0, $1, $2;", "=r,r,r", [packed_x2, packed_w], tl.uint32, True, 1)
            prod00 = prod_pair0.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod01 = (prod_pair0 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod10 = prod_pair1.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod11 = (prod_pair1 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod20 = prod_pair2.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod21 = (prod_pair2 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            acc0 = acc0 + (prod00 + prod01)
            acc1 = acc1 + (prod10 + prod11)
            acc2 = acc2 + (prod20 + prod21)

    for offset_idx in tl.static_range(5):
        shift = 16 >> offset_idx
        src = tl.broadcast_to((lanes + shift) & 31, (1, OUT_TILE, 32))
        acc0 = tl.where(lanes < 32 - shift, acc0 + tl.gather(acc0, src, axis=2), acc0)
        acc1 = tl.where(lanes < 32 - shift, acc1 + tl.gather(acc1, src, axis=2), acc1)
        acc2 = tl.where(lanes < 32 - shift, acc2 + tl.gather(acc2, src, axis=2), acc2)
    lane_idx0 = tl.full((1, OUT_TILE, 1), 0, tl.int32)
    lane0, lane1, lane2 = tl.gather(acc0, lane_idx0, axis=2), tl.gather(acc1, lane_idx0, axis=2), tl.gather(acc2, lane_idx0, axis=2)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + outs, mask=outs < N, other=0).to(tl.float32)
        lane0, lane1, lane2 = lane0 + bias, lane1 + bias, lane2 + bias
    tl.store(y_ptr + outs, lane0.to(tl.float16), outs < N)
    tl.store(y_ptr + N + outs, lane1.to(tl.float16), outs < N)
    tl.store(y_ptr + 2 * N + outs, lane2.to(tl.float16), outs < N)
