import json, sys, torch
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from audit_pld_trajectory_divergence import reference
from zassd.backends.base import load_backend
rows=[json.loads(l) for l in open("data/benchmarks/gsm8k_eval.jsonl")][:50]
for backend in ("hqq_gemlite_w4a16","hf_bnb_nf4"):
    L=load_backend(backend,"Qwen/Qwen2.5-3B-Instruct","aa8e72537993ba99e69dfaafa59ed015b17504d1","cuda:0","sdpa"); m,tok=L.model.eval(),L.tokenizer
    diff=0; first=[]
    with torch.inference_mode():
        for r in rows:
            ids=tok(r["prompt"],return_tensors="pt")["input_ids"].to("cuda:0"); cap=ids.shape[1]+140
            a=reference(m,ids,128,cap); b=reference(m,ids,128,cap)
            if a!=b:
                diff+=1; first.append(next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y))
    print(backend,"M=1 reference run twice: differing sequences",diff,"/50 first idx",first[:8],flush=True)
    del L,m; torch.cuda.empty_cache()
