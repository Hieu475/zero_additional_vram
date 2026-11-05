import pytest, torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_rows_bitwise_independent_of_block():
    from zassd.kernels.lm_head_invariant import lm_head_invariant, MP
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(5000, 256, device="cuda", generator=g) * 0.05).half().contiguous()
    X = torch.randn(MP, 256, device="cuda", generator=g).half()
    ref = torch.cat([lm_head_invariant(X[i:i + 1], w, bk=64) for i in range(MP)])
    for M in range(2, MP + 1):
        perm = torch.randperm(MP, device="cuda")[:M]
        out = lm_head_invariant(X[perm], w, bk=64)
        for j, i in enumerate(perm.tolist()):
            assert torch.equal(out[j], ref[i])


def test_attention_rows_bitwise_independent_of_block():
    from zassd.kernels.attention_invariant import attention_invariant
    g = torch.Generator(device="cuda").manual_seed(1)
    K = torch.randn(1, 2, 700, 64, device="cuda", generator=g).half(); V = torch.randn_like(K)
    Q = torch.randn(1, 8, 8, 64, device="cuda", generator=g).half()
    for ctx in (100, 255, 256, 257, 600):
        ref = [attention_invariant(Q[:, :, i:i + 1], K, V, ctx + i) for i in range(8)]
        out = attention_invariant(Q, K, V, ctx)
        for i in range(8):
            assert torch.equal(out[:, :, i], ref[i][:, :, 0])


def test_attention_graph_mode_equals_int_mode():
    from zassd.kernels.attention_invariant import attention_invariant
    g = torch.Generator(device="cuda").manual_seed(2)
    K = torch.randn(1, 2, 900, 64, device="cuda", generator=g).bfloat16(); V = torch.randn_like(K)
    Q = torch.randn(1, 8, 3, 64, device="cuda", generator=g).bfloat16()
    for ctx in (10, 255, 256, 700):
        pt = torch.tensor([ctx], dtype=torch.int32, device="cuda")
        assert torch.equal(attention_invariant(Q, K, V, ctx), attention_invariant(Q, K, V, 0, prefix_tensor=pt))


def test_nf4_gemm_rows_bitwise_independent_of_block_and_variant():
    import bitsandbytes.functional as BF
    from zassd.kernels.nf4_gemm_cuda import nf4_gemm_cuda
    g = torch.Generator(device="cuda").manual_seed(3)
    N, K = 640, 1280
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.05).bfloat16()
    packed, qs = BF.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
    X = torch.randn(8, K, device="cuda", generator=g).bfloat16()
    bias = torch.randn(N, device="cuda", generator=g).half()
    ref = torch.cat([nf4_gemm_cuda(X[i:i + 1], packed, qs, bias, 1, 4, 2) for i in range(8)])
    for M, mp in [(2, 2), (3, 4), (4, 4), (5, 8), (8, 8)]:
        out = nf4_gemm_cuda(X[:M], packed, qs, bias, mp, 4, 2 if mp <= 4 else 4)
        assert torch.equal(out, ref[:M])
    deq = BF.dequantize_4bit(packed, qs).float()
    err = (nf4_gemm_cuda(X, packed, qs, bias, 8, 4, 4).float() - (X.float() @ deq.T + bias.float())).abs().max()
    assert err < 0.1
