import sys
sys.path.insert(0, "/home/nguyen_quoc_hieu/Documents/zero_additional_vram")
import torch
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.models.layer_manager import LayerManager
from zassd.cache.kv_cache import TargetKVCache
DEVICE="cuda:0"
tok=load_tokenizer("Qwen/Qwen2.5-3B-Instruct")
model=load_model("Qwen/Qwen2.5-3B-Instruct", quantize=True, bits=4, device=DEVICE)
model.eval()
adapt=ModelAdapter(model); lm=LayerManager(adapt); NL=adapt.num_layers
prompt="Discuss self-speculative decoding and memory bandwidth."
ids=tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]
tkv=TargetKVCache()
with torch.no_grad(): pre=model(ids, past_key_values=tkv.cache, use_cache=True)
torch.cuda.synchronize()
cur=pre.logits[:,-1:,:].argmax(dim=-1)
# VANILLA region: 15 single-token forwards
for i in range(15):
    with torch.no_grad(): out=model(cur, past_key_values=tkv.cache, use_cache=True)
    cur=out.logits[:,-1:,:].argmax(dim=-1)
torch.cuda.synchronize()
# DRAFT region: skip 25%, 15 forwards
ns=9; step=NL/float(ns); skip=[int(i*step) for i in range(ns)]
dkv=TargetKVCache()
with torch.no_grad(): model(ids, past_key_values=dkv.cache, use_cache=True)
torch.cuda.synchronize()
dcur=cur
for i in range(15):
    with torch.no_grad():
        with lm.skip_layers(skip): out=model(dcur, past_key_values=dkv.cache, use_cache=True)
    dcur=out.logits[:,-1:,:].argmax(dim=-1)
torch.cuda.synchronize()
print("WORKLOAD_DONE", flush=True)
