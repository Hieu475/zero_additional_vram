"""Corrected sweep with true CKA indices on benchmark prompts + 0.5B baseline N=20."""
import sys, time, json, gc, statistics
sys.path.insert(0, "/home/nguyen_quoc_hieu/Documents/zero_additional_vram")
import torch
from zassd.models.loader import load_model, load_tokenizer, get_quantization_config
from zassd.models.model_adapter import ModelAdapter
from zassd.models.layer_manager import LayerManager
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.vanilla import vanilla_generate
from zassd.decoding.speculative import self_speculative_generate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from pathlib import Path
DEVICE="cuda:0"
CKA={"cka_83":[4,5,6,7,12,13],"cka_75":[3,4,5,6,7,10,11,12,13],
 "cka_60":[3,4,5,6,7,8,10,11,12,13,14,16,17,21],
 "cka_50":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,22],
 "empirical_6":[4,5,19,20,22,23]}
# load benchmark prompts (gsm8k first 8)
recs=[]
with open("data/benchmarks/gsm8k_eval.jsonl") as f:
    for line in f:
        if line.strip(): recs.append(json.loads(line))
        if len(recs)>=8: break
prompts=[r["prompt"] for r in recs]
print(f"prompts={len(prompts)}", flush=True)
tok=load_tokenizer("Qwen/Qwen2.5-3B-Instruct")
model=load_model("Qwen/Qwen2.5-3B-Instruct", quantize=True, bits=4, device=DEVICE); model.eval()
adapt=ModelAdapter(model); lm=LayerManager(adapt)
# T_vanilla ref
ids=tok(prompts[0], return_tensors="pt").to(DEVICE)["input_ids"]
tkv=TargetKVCache()
with torch.no_grad(): pre=model(ids, past_key_values=tkv.cache, use_cache=True)
torch.cuda.synchronize(); cur=pre.logits[:,-1:,:].argmax(dim=-1)
def mt(skip, n=20):
    vals=[]
    for _ in range(n):
        torch.cuda.synchronize(); t0=time.perf_counter_ns()
        with torch.no_grad():
            if skip is None: model(cur, past_key_values=tkv.cache, use_cache=True)
            else:
                with lm.skip_layers(skip): model(cur, past_key_values=tkv.cache, use_cache=True)
        torch.cuda.synchronize(); t1=time.perf_counter_ns(); vals.append((t1-t0)/1e6)
    return sum(vals)/len(vals)
Tv=mt(None); print(f"T_vanilla={Tv:.2f}ms", flush=True)
sweep={}
for name,skip in CKA.items():
    Td=mt(skip); c=Td/Tv
    al=[]; sp=[]
    for pr in prompts:
        _,m=self_speculative_generate(model,tok,lm,skip,pr,k=1,max_new_tokens=48,device=DEVICE)
        _,mv=vanilla_generate(model,tok,pr,max_new_tokens=48,device=DEVICE)
        al.append(m.acceptance_rate); sp.append(m.tokens_per_second/max(mv.tokens_per_second,1e-6))
    sweep[name]={"skip_n":len(skip),"T_draft_ms":Td,"c":c,
      "alpha_mean":sum(al)/len(al),"alpha_std":statistics.stdev(al) if len(al)>1 else 0,
      "speedup_mean":sum(sp)/len(sp),"speedup_std":statistics.stdev(sp) if len(sp)>1 else 0}
    print(f"{name} n={len(skip)} Td={Td:.2f} c={c:.3f} alpha={sweep[name]['alpha_mean']:.3f} sp={sweep[name]['speedup_mean']:.3f}", flush=True)
json.dump({"T_vanilla_ms":Tv,"sweep":sweep}, open("/tmp/opencode/sweep_corr.json","w"), indent=2)
del model,adapt,lm; gc.collect(); torch.cuda.empty_cache()
# 0.5B baseline N=20 (10 gsm8k + 10 humaneval first 10)
recs2=[]
with open("data/benchmarks/humaneval_eval.jsonl") as f:
    for line in f:
        if line.strip(): recs2.append(json.loads(line))
        if len(recs2)>=10: break
prompts20=prompts[:10] if len(prompts)>=10 else prompts
# prompts currently 8 gsm8k; add humaneval to reach ~18
prompts20=prompts+[r["prompt"] for r in recs2[:10]]
print(f"N20={len(prompts20)}", flush=True)
tok3=load_tokenizer("Qwen/Qwen2.5-3B-Instruct")
modelT=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",quantization_config=get_quantization_config(4),device_map="auto"); modelT.eval()
modelD=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",quantization_config=get_quantization_config(4),device_map="auto"); modelD.eval()
def spec2(prompt,K=2,NT=48):
    tk=TargetKVCache(); dk=DynamicCache()
    ids=tok3(prompt,return_tensors="pt").to(DEVICE)["input_ids"]
    with torch.no_grad(): p=modelT(ids,past_key_values=tk.cache,use_cache=True)
    torch.cuda.synchronize(); cur=p.logits[:,-1:,:].argmax(dim=-1)
    d_ids=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")(prompt,return_tensors="pt").to(DEVICE)["input_ids"]
    with torch.no_grad(): modelD(d_ids,past_key_values=dk,use_cache=True)
    dcur=d_ids[:,-1:]; gen=[]; acc=0; prop=0; torch.cuda.synchronize(); t0=time.perf_counter()
    with torch.no_grad():
        while len(gen)<NT:
            drafts=[]
            for _ in range(K):
                o=modelD(dcur,past_key_values=dk,use_cache=True); nx=o.logits[:,-1:,:].argmax(dim=-1); drafts.append(int(nx.item())); dcur=nx; prop+=1
            cand=torch.cat([cur]+[torch.tensor([[d]],device=DEVICE) for d in drafts],dim=-1)
            v=modelT(cand,past_key_values=tk.cache,use_cache=True); torch.cuda.synchronize()
            vp=v.logits[0,:K,:].argmax(dim=-1).tolist()
            for i,d in enumerate(drafts):
                if vp[i]==d: gen.append(d); acc+=1
                else: gen.append(vp[i]); break
            bonus=int(v.logits[0,K,:].argmax().item()); gen.append(bonus); cur=torch.tensor([[bonus]],device=DEVICE); dcur=torch.tensor([[bonus]],device=DEVICE)
            if len(gen)>=NT: break
    torch.cuda.synchronize(); t1=time.perf_counter(); tps=len(gen)/(t1-t0)
    return tps,(acc/prop if prop else 0)
import numpy as np
sps=[]; als=[]
for pr in prompts20:
    tps,al=spec2(pr,K=2,NT=48)
    _,mv=vanilla_generate(modelT,tok3,pr,max_new_tokens=48,device=DEVICE)
    s=tps/max(mv.tokens_per_second,1e-6); sps.append(s); als.append(al)
    print(f"sp={s:.3f} a={al:.2f}", flush=True)
sps=np.array(sps)
mean=float(sps.mean()); std=float(sps.std(ddof=1)); ci=1.96*std/(len(sps)**0.5)
print(f"0.5B K=2 N={len(sps)} mean={mean:.3f} std={std:.3f} 95%CI=±{ci:.3f} alpha_mean={float(np.mean(als)):.3f}", flush=True)
json.dump({"N":len(sps),"speedups":sps.tolist(),"alphas":als,"mean":mean,"std":std,"ci95":ci}, open("/tmp/opencode/baseline05_N20.json","w"), indent=2)
print("DONE", flush=True)
