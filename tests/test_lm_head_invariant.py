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
