#!/usr/bin/env python3
"""WP7: does the stock bitsandbytes NF4 projection depend on the number of rows M, per bitsandbytes version? (The invariant CUDA kernel
does not call bnb's matmul kernel; this only measures how the baseline it replaces behaves.) Prints one JSON line."""
import json, torch, bitsandbytes as bnb, bitsandbytes.functional as BF
g = torch.Generator(device="cuda").manual_seed(0)
res = {"bnb": bnb.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "shapes": {}}
for (N, K) in [(11008, 2048), (2048, 2048), (2048, 11008)]:
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.05).half()
    packed, qs = BF.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
    X = torch.randn(16, K, device="cuda", generator=g).half()
    ref = torch.cat([bnb.matmul_4bit(X[i:i + 1], packed.t(), qs) for i in range(16)])
    per_m = {}
    for M in range(2, 17):
        out = bnb.matmul_4bit(X[:M], packed.t(), qs)
        per_m[M] = round(float((out != ref[:M]).float().mean()) * 100, 3)
    res["shapes"][f"{N}x{K}"] = per_m
print(json.dumps(res))
