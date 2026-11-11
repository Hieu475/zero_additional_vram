"""Feature-conditioned sequential drafter (EAGLE-lite) with a trimmed-vocabulary head.
Step k: u = fc([s_{k-1}, emb(x_{k-1})]); s_k = u + MLP(rmsnorm(u)); logits_k = s_k @ Wv^T over a trimmed vocab.
Drafts are only proposals: exactness is guaranteed by verification with the M-invariant kernels."""
import torch, torch.nn as nn, torch.nn.functional as F

class Drafter(nn.Module):
    def __init__(self, d=2048, ff=4096, vocab_ids=None, head_init=None):
        super().__init__()
        self.fc = nn.Linear(2 * d, d, bias=True)
        self.norm_w = nn.Parameter(torch.ones(d)); self.w1 = nn.Linear(d, ff, bias=False); self.w2 = nn.Linear(ff, d, bias=False)
        self.register_buffer("vocab_ids", vocab_ids)           # [V'] original token ids
        self.head = nn.Parameter(head_init.clone())             # [V', d]
        with torch.no_grad():                                   # start near identity on the state path
            self.fc.weight.zero_(); self.fc.weight[:, :d].copy_(torch.eye(d)); self.fc.bias.zero_()
            self.w2.weight.mul_(0.1)
    def step(self, s, e):
        u = self.fc(torch.cat([s, e], -1))
        x = u * torch.rsqrt(u.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(u.dtype) * self.norm_w
        return u + self.w2(F.silu(self.w1(x)))
    def logits(self, s): return s @ self.head.t()
