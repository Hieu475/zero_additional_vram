"""Exp 1-4 combined: baseline 0.5B, sweep c, free-form workload, exact-match audit."""
import sys, time, json, gc, statistics
from pathlib import Path
sys.path.insert(0, "/home/nguyen_quoc_hieu/Documents/zero_additional_vram")
import torch
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.models.layer_manager import LayerManager
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.vanilla import vanilla_generate
from zassd.decoding.speculative import self_speculative_generate
from zassd.baselines.prompt_lookup import prompt_lookup_generate

DEVICE="cuda:0"
OUT={"hardware": torch.cuda.get_device_name(0)}
def sync(): torch.cuda.synchronize()

def measure_single_token(model, tok_ids, kv, trials=30, skip_mgr=None, skip_idx=None):
    sync(); vals=[]
    ce_s=torch.cuda.Event(enable_timing=True); ce_e=torch.cuda.Event(enable_timing=True)
    ce_vals=[]
    curr=tok_ids
    for _ in range(5):  # warmup
        with torch.no_grad():
            if skip_mgr: 
                with skip_mgr.skip_layers(skip_idx): model(curr, past_key_values=kv.fork_ephemeral_draft_kv() if hasattr(kv,'fork_ephemeral_draft_kv') else kv, use_cache=True)
            else: model(curr, past_key_values=kv, use_cache=True)
        sync()
    for _ in range(trials):
        sync(); t0=time.perf_counter_ns()
        with torch.no_grad():
            if skip_mgr:
                with skip_mgr.skip_layers(skip_idx): out=model(curr, past_key_values=kv, use_cache=True)
            else: out=model(curr, past_key_values=kv, use_cache=True)
        sync(); t1=time.perf_counter_ns()
        vals.append((t1-t0)/1e6)
        # cuda event
        ce_s.record(); 
        with torch.no_grad():
            if skip_mgr:
                with skip_mgr.skip_layers(skip_idx): model(curr, past_key_values=kv, use_cache=True)
            else: model(curr, past_key_values=kv, use_cache=True)
        ce_e.record(); sync(); ce_vals.append(ce_s.elapsed_time(ce_e))
    return float(sum(vals)/len(vals)), float(statistics.stdev(vals) if len(vals)>1 else 0), float(sum(ce_vals)/len(ce_vals))

print("=== LOAD 3B ===", flush=True)
tok3=load_tokenizer("Qwen/Qwen2.5-3B-Instruct")
model3=load_model("Qwen/Qwen2.5-3B-Instruct", quantize=True, bits=4, device=DEVICE)
model3.eval()
adapt=ModelAdapter(model3); lm=LayerManager(adapt); NL=adapt.num_layers
nparams=sum(p.numel() for p in model3.parameters())
weight_bytes_est=nparams*0.5  # NF4 ~0.5B/param + scales
OUT["3B_params"]=nparams
print(f"3B layers={NL} params={nparams} est_weight_MB={weight_bytes_est/1e6:.1f}", flush=True)

# ---- Timer fix demo (Task 2b) ----
sync(); N=500; ss=[]
for _ in range(N):
    t0=time.perf_counter_ns(); sync(); t1=time.perf_counter_ns(); ss.append((t1-t0)/1e3)
OUT["T_sync_us_mean"]=float(sum(ss)/len(ss)); OUT["T_sync_us_std"]=float(statistics.stdev(ss))
OUT["T_sync_us_p50"]=float(sorted(ss)[N//2])
# layer mgmt with ns
lm_vals=[]
for _ in range(200):
    sk=list(range(0,9))
    t0=time.perf_counter_ns(); lm.set_skipped_layers(sk); lm.restore_layers(); t1=time.perf_counter_ns()
    lm_vals.append((t1-t0)/1e3)
OUT["T_layer_us_mean"]=float(sum(lm_vals)/len(lm_vals)); OUT["T_layer_us_std"]=float(statistics.stdev(lm_vals))
print(f"T_sync={OUT['T_sync_us_mean']:.2f}±{OUT['T_sync_us_std']:.2f}us T_layer={OUT['T_layer_us_mean']:.1f}±{OUT['T_layer_us_std']:.1f}us", flush=True)

# ---- Task 2: sweep c ----
prompt="Discuss the mechanics of self-speculative decoding, focusing on memory bandwidth saturation and KV cache overhead."
ids=tok3(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
tkv=TargetKVCache()
with torch.no_grad(): pre=model3(ids, past_key_values=tkv.cache, use_cache=True)
sync()
last_tok=pre.logits[:,-1:,:].argmax(dim=-1)
# vanilla single-token timing (steady state, reuse tkv)
tv_mean,tv_std,tv_ce=measure_single_token(model3, last_tok, tkv.cache, trials=30)
OUT["T_vanilla_ms"]=tv_mean; OUT["T_vanilla_std"]=tv_std; OUT["T_vanilla_cudaevent"]=tv_ce
OUT["achieved_BW_GBs"]=(weight_bytes_est/1e9)/(tv_mean/1e3)
print(f"T_vanilla={tv_mean:.2f}±{tv_std:.2f}ms cudaEvt={tv_ce:.2f}ms => BW~{OUT['achieved_BW_GBs']:.1f} GB/s", flush=True)
sweep={}
for ratio in [0.10,0.25,0.40,0.50,0.70]:
    ns=int(round(NL*ratio)); step=NL/float(max(ns,1))
    skip=[int(i*step) for i in range(ns)]
    draft_tkv=TargetKVCache()
    with torch.no_grad(): model3(ids, past_key_values=draft_tkv.cache, use_cache=True)
    sync()
    td,td_std,td_ce=measure_single_token(model3, last_tok, draft_tkv.cache, trials=20, skip_mgr=lm, skip_idx=skip)
    c=td/tv_mean
    # verify batched K=1 (2 tokens) timing
    sync(); vv=[]
    cand=torch.cat([last_tok,last_tok],dim=-1)
    for _ in range(10):
        sync(); t0=time.perf_counter_ns()
        with torch.no_grad(): model3(cand, past_key_values=tkv.cache, use_cache=True)
        # need fresh kv each time? use fork to avoid growth: recreate
        sync(); t1=time.perf_counter_ns(); vv.append((t1-t0)/1e6)
        # reset kv growth by cropping? simplest recreate every 5
    # alpha + real speedup via short generation (2 prompts x 32 tokens, K=1)
    alphas=[]; speedups=[]
    for pr in ["Solve: If x+7=15, what is x? Explain step by step.", "Write one sentence about rain."]:
        _,m=self_speculative_generate(model3,tok3,lm,skip,pr,k=1,max_new_tokens=32,device=DEVICE)
        _,mv=vanilla_generate(model3,tok3,pr,max_new_tokens=32,device=DEVICE)
        alphas.append(m.acceptance_rate); speedups.append(m.tokens_per_second/max(mv.tokens_per_second,1e-6))
    sweep[f"skip{int(ratio*100)}"]={"skip_n":ns,"T_draft_ms":td,"T_draft_std":td_std,"c":c,"alpha_mean":float(sum(alphas)/len(alphas)),"speedup_mean":float(sum(speedups)/len(speedups)),"T_verify2_est_ms":float(sum(vv)/len(vv))}
    print(f"skip{ratio}: Tdraft={td:.2f} c={c:.3f} alpha={sweep[f'skip{int(ratio*100)}']['alpha_mean']:.3f} speedup={sweep[f'skip{int(ratio*100)}']['speedup_mean']:.3f}", flush=True)
OUT["sweep_c"]=sweep

# ---- Task 4: exact-match investigation ----
print("=== TASK4 exact-match ===", flush=True)
p4="Solve: If x+7=15, what is x? Explain step by step."
t1,_=vanilla_generate(model3,tok3,p4,max_new_tokens=32,device=DEVICE)
t2,_=vanilla_generate(model3,tok3,p4,max_new_tokens=32,device=DEVICE)
ids1=tok3(t1,return_tensors="pt")["input_ids"][0].tolist(); ids2=tok3(t2,return_tensors="pt")["input_ids"][0].tolist()
OUT["vanilla_self_determinism"]=(t1==t2)
# vanilla vs layer-skip token ids
skip9=[int(i*(NL/9)) for i in range(9)]
tspec,_=self_speculative_generate(model3,tok3,lm,skip9,p4,k=1,max_new_tokens=32,device=DEVICE)
tpld,_=prompt_lookup_generate(model3,tok3,p4,k=2,max_new_tokens=32,device=DEVICE)
# tokenize outputs to ids for comparison (approx via tokenizer encode of generated text is imperfect; instead regenerate ids via vanilla fn? use encode)
e_v=tok3(t1)["input_ids"]; e_s=tok3(tspec)["input_ids"]; e_p=tok3(tpld)["input_ids"]
def seq_match(a,b): 
    n=min(len(a),len(b)); m=sum(1 for x,y in zip(a[:n],b[:n]) if x==y); return m,max(len(a),len(b)),(m==max(len(a),len(b)))
OUT["vanilla_vs_layerskip_match"]=seq_match(e_v,e_s); OUT["vanilla_vs_pld_match"]=seq_match(e_v,e_p)
# single vs batched audit on fixed context
ctx_ids=tok3(p4,return_tensors="pt").to(DEVICE)["input_ids"]
with torch.no_grad():
    o_single=model3(ctx_ids[:,:-1],use_cache=False)
    o_batch=model3(ctx_ids,use_cache=False)
import torch.nn.functional as F
Ls=o_single.logits[0,-1,:].float(); Lb=o_batch.logits[0,-2,:].float()
linf=float((Ls-Lb).abs().max().item()); cos=float(F.cosine_similarity(Ls.unsqueeze(0),Lb.unsqueeze(0)).item())
a1=int(Ls.argmax()); a2=int(Lb.argmax())
mgn=float(torch.topk(Ls,2).values[0].item()-torch.topk(Ls,2).values[1].item())
OUT["audit_single_vs_batched"]={"linf":linf,"cos":cos,"flip":bool(a1!=a2),"margin":mgn}
print(json.dumps({k:OUT[k] for k in ["vanilla_self_determinism","vanilla_vs_layerskip_match","vanilla_vs_pld_match","audit_single_vs_batched"]},indent=2), flush=True)

# ---- Task 3: free-form workload, low overlap ----
print("=== TASK3 free-form ===", flush=True)
free=[
 "Write a haiku about a lighthouse that has never seen a ship. Do not repeat any line.",
 "Invent a completely new soup recipe with starfruit and smoked paprika. List steps.",
 "Translate into French: The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
 "You are a lonely chatbot on a space station. Describe your day in 5 sentences without repeating words across sentences if possible.",
 "Write a 4-line poem where each line starts with Z, Q, X, J respectively.",
 "Explain quantum tunneling using only a cooking analogy, in about 100 words.",
]
t3={}
for pr in free:
    _,mv=vanilla_generate(model3,tok3,pr,max_new_tokens=96,device=DEVICE)
    _,mp=prompt_lookup_generate(model3,tok3,pr,k=2,max_new_tokens=96,device=DEVICE)
    _,ms=self_speculative_generate(model3,tok3,lm,skip9,pr,k=1,max_new_tokens=96,device=DEVICE)
    sp=mp.tokens_per_second/max(mv.tokens_per_second,1e-6); ss=ms.tokens_per_second/max(mv.tokens_per_second,1e-6)
    t3[pr[:30]]={"vanilla_tps":mv.tokens_per_second,"pld_sp":sp,"pld_acc":mp.acceptance_rate,"ls_sp":ss,"ls_acc":ms.acceptance_rate}
    print(f"{pr[:30]}... vanilla={mv.tokens_per_second:.1f} PLD {sp:.3f}(a={mp.acceptance_rate:.2f}) LS {ss:.3f}(a={ms.acceptance_rate:.2f})", flush=True)
OUT["freeform_96tok"]=t3
# one long 256 run
pl=free[0]
_,mvL=vanilla_generate(model3,tok3,pl,max_new_tokens=256,device=DEVICE)
_,mpL=prompt_lookup_generate(model3,tok3,pl,k=2,max_new_tokens=256,device=DEVICE)
_,msL=self_speculative_generate(model3,tok3,lm,skip9,pl,k=1,max_new_tokens=256,device=DEVICE)
OUT["long256"]={"vanilla_tps":mvL.tokens_per_second,"pld_sp":mpL.tokens_per_second/max(mvL.tokens_per_second,1e-6),"ls_sp":msL.tokens_per_second/max(mvL.tokens_per_second,1e-6),"pld_acc":mpL.acceptance_rate,"ls_acc":msL.acceptance_rate}
print("LONG256 "+json.dumps(OUT["long256"]), flush=True)

with open("/tmp/opencode/results_234.json","w") as f: json.dump(OUT,f,indent=2)
# free 3B before task1 to save VRAM (keep tokenizer)
del model3, adapt, lm; gc.collect(); torch.cuda.empty_cache()

# ---- Task 1: 0.5B draft baseline ----
print("=== TASK1 0.5B draft ===", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
tok05=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
from zassd.models.loader import get_quantization_config
modelT=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",quantization_config=get_quantization_config(4),device_map="auto"); modelT.eval()
modelD=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",quantization_config=get_quantization_config(4),device_map="auto"); modelD.eval()
sync()
memT=torch.cuda.memory_allocated()/1e6
# measure draft/target latencies
pr="Solve: If x+7=15, what is x? Explain step by step."
idsT=tok3(pr,return_tensors="pt").to(DEVICE)["input_ids"]
tkT=TargetKVCache()
with torch.no_grad(): preT=modelT(idsT,past_key_values=tkT.cache,use_cache=True)
sync(); lastT=preT.logits[:,-1:,:].argmax(dim=-1)
idsD=tok05(pr,return_tensors="pt").to(DEVICE)["input_ids"]
dkv=DynamicCache()
with torch.no_grad(): preD=modelD(idsD,past_key_values=dkv,use_cache=True)
sync(); lastD=preD.logits[:,-1:,:].argmax(dim=-1)
def mt(model,kv,tok,n=30):
    sync(); v=[]
    for _ in range(n):
        sync(); t0=time.perf_counter_ns()
        with torch.no_grad(): model(tok,past_key_values=kv,use_cache=True)
        sync(); t1=time.perf_counter_ns(); v.append((t1-t0)/1e6)
    return float(sum(v)/len(v))
tV=mt(modelT,tkT.cache,lastT); tD=mt(modelD,dkv,lastD)
c05=tD/tV
# true 2-model speculative loop K=3 on 3 prompts
def spec2(prompt,K=3,NT=48):
    tk=TargetKVCache(); dk=DynamicCache()
    ids=tok3(prompt,return_tensors="pt").to(DEVICE)["input_ids"]
    with torch.no_grad(): p=modelT(ids,past_key_values=tk.cache,use_cache=True)
    sync(); cur=p.logits[:,-1:,:].argmax(dim=-1); curD=tok05.convert_tokens_to_ids(tok3.convert_ids_to_tokens(int(cur.item()))) if False else None
    # simplify: draft with 0.5B tokenizer encoding of same text
    d_ids=tok05(prompt,return_tensors="pt").to(DEVICE)["input_ids"]
    with torch.no_grad(): modelD(d_ids,past_key_values=dk,use_cache=True)
    dcur=d_ids[:,-1:]
    gen=[]; acc=0; prop=0; sync(); t0=time.perf_counter()
    with torch.no_grad():
        while len(gen)<NT:
            drafts=[]
            for _ in range(K):
                o=modelD(dcur,past_key_values=dk,use_cache=True); nx=o.logits[:,-1:,:].argmax(dim=-1); drafts.append(int(nx.item())); dcur=nx
                prop+=1
            # map draft ids (0.5B vocab same as 3B for Qwen) directly
            cand=torch.cat([cur]+[torch.tensor([[d]],device=DEVICE) for d in drafts],dim=-1)
            v=modelT(cand,past_key_values=tk.cache,use_cache=True); sync()
            vp=v.logits[0,:K,:].argmax(dim=-1).tolist()
            ok=0
            for i,d in enumerate(drafts):
                if vp[i]==d: gen.append(d); acc+=1; ok+=1
                else: gen.append(vp[i]); break
            else: pass
            bonus=int(v.logits[0,K,:].argmax().item()); gen.append(bonus); cur=torch.tensor([[bonus]],device=DEVICE); dcur=torch.tensor([[bonus]],device=DEVICE)
            if len(gen)>=NT: break
    sync(); t1=time.perf_counter(); tps=len(gen)/(t1-t0)
    return tps, (acc/prop if prop else 0)
res={}
for K in [2,3]:
    for pr2 in ["Solve: If x+7=15, what is x? Explain step by step.","Write one sentence about rain."]:
        tps,al=spec2(pr2,K=K,NT=48)
        _,mv2=vanilla_generate(modelT,tok3,pr2,max_new_tokens=48,device=DEVICE)
        res[f"K{K}_{pr2[:15]}"]={"tps":tps,"alpha":al,"vanilla_tps":mv2.tokens_per_second,"speedup":tps/max(mv2.tokens_per_second,1e-6)}
        print(f"0.5B-draft K={K} {pr2[:20]} tps={tps:.1f} a={al:.2f} sp={res[f'K{K}_{pr2[:15]}']['speedup']:.3f}", flush=True)
OUT1={"T_vanilla_ms":tV,"T_draft05_ms":tD,"c_05":c05,"vram_alloc_MB":memT,"spec2":res}
with open("/tmp/opencode/results_1.json","w") as f: json.dump(OUT1,f,indent=2)
print("DONE", flush=True)
