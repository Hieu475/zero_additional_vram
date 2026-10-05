import sys
sys.path.insert(0, "/home/nguyen_quoc_hieu/Documents/zero_additional_vram")
import torch
from zassd.models.loader import load_model, load_tokenizer
from zassd.cache.kv_cache import TargetKVCache
DEVICE="cuda:0"
tok=load_tokenizer("Qwen/Qwen2.5-3B-Instruct")
model=load_model("Qwen/Qwen2.5-3B-Instruct", quantize=True, bits=4, device=DEVICE)
model.eval()
ids=tok("Hello world test.", return_tensors="pt").to(DEVICE)["input_ids"]
tkv=TargetKVCache()
with torch.no_grad(): pre=model(ids, past_key_values=tkv.cache, use_cache=True)
torch.cuda.synchronize()
cur=pre.logits[:,-1:,:].argmax(dim=-1)
for i in range(3):
    with torch.no_grad(): out=model(cur, past_key_values=tkv.cache, use_cache=True)
    cur=out.logits[:,-1:,:].argmax(dim=-1)
torch.cuda.synchronize()
print("TINY_DONE", flush=True)
