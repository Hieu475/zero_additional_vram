import sys, torch
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from zassd.backends.base import load_backend
L=load_backend("hqq_gemlite_w4a16","Qwen/Qwen2.5-3B-Instruct","aa8e72537993ba99e69dfaafa59ed015b17504d1","cuda:0","sdpa"); m=L.model.eval()
lins=[(n,x) for n,x in m.named_modules() if type(x).__module__.startswith("gemlite")]
print("gemlite modules",len(lins),type(lins[0][1]).__name__)
torch.manual_seed(0)
tot={1:[0,0],3:[0,0]}
with torch.inference_mode():
    for n,x in lins[::7]:
        in_f=x.in_features if hasattr(x,"in_features") else x.W_q.shape[1]
        for M in (1,3):
            inp=torch.randn(1,M,in_f,device="cuda",dtype=torch.float16)
            outs=[x(inp.clone()) for _ in range(5)]
            bad=sum(int((o!=outs[0]).sum()) for o in outs[1:]); tot[M][0]+=bad; tot[M][1]+=outs[0].numel()*4
        # row-vs-block with identical inputs
    print("same input repeated 5x, mismatching elements vs first call: M=1 %d/%d  M=3 %d/%d"%(tot[1][0],tot[1][1],tot[3][0],tot[3][1]))
