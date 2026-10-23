#!/usr/bin/env python3
"""Repeat identical greedy backend forwards to detect run-to-run token variation."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys, time
from pathlib import Path

import torch
import transformers
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache

MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def generate(model, prompt_ids, eos_id, budget, capacity, device):
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    tokens, margins, top2_token_ids = [], [], []
    with torch.inference_mode():
        out = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
        for i in range(budget):
            logits = out.logits[0, -1].float()
            top = torch.topk(logits, 2)
            token = int(logits.argmax().item())
            tokens.append(token)
            margins.append(float((top.values[0] - top.values[1]).item()))
            top2_token_ids.append([int(x) for x in top.indices.tolist()])
            if token == eos_id or i + 1 >= budget:
                break
            out = model(torch.tensor([[token]], dtype=torch.long, device=device),
                        past_key_values=cache.cache, use_cache=True)
    return tokens, margins, top2_token_ids, cache.get_seq_length()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['hf_bnb_nf4','hqq_gemlite_w4a16'], required=True)
    ap.add_argument('--inputs', type=Path, default=ROOT/'experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl')
    ap.add_argument('--ids', nargs='+', required=True)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--max-new-tokens', type=int, default=32)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    rows={}
    for line in args.inputs.read_text().splitlines():
        if line.strip():
            item=json.loads(line); rows[str(item.get('prompt_id',item.get('id')))]=item
    missing=set(args.ids)-rows.keys()
    if missing: raise ValueError(f'missing prompt IDs: {sorted(missing)}')
    loaded=load_backend(args.backend,'Qwen/Qwen2.5-3B-Instruct',MODEL_REVISION,
                        device=args.device,attention='sdpa')
    execution_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,
                                    capture_output=True,text=True).stdout.strip()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    existing=[]
    if args.output.exists():
        existing=[json.loads(x) for x in args.output.read_text().splitlines() if x.strip()]
        if any(r['backend']!=args.backend for r in existing): raise ValueError('output backend mismatch')
    done={(r['prompt_id'],r['repeat']) for r in existing}
    with args.output.open('a',encoding='utf-8') as f:
        for prompt_id in args.ids:
            input_ids=rows[prompt_id]['input_ids']
            prompt=torch.tensor([input_ids],dtype=torch.long,device=args.device)
            prompt_hash=hashlib.sha256(json.dumps(input_ids,separators=(',',':')).encode()).hexdigest()
            capacity=len(input_ids)+args.max_new_tokens+16
            for repeat in range(args.repeats):
                if (prompt_id,repeat) in done: continue
                torch.cuda.synchronize(args.device); started=time.perf_counter()
                ids,margins,top2_ids,kv_length=generate(loaded.model,prompt,loaded.tokenizer.eos_token_id,
                                                       args.max_new_tokens,capacity,args.device)
                torch.cuda.synchronize(args.device); elapsed=time.perf_counter()-started
                row={'prompt_id':prompt_id,'repeat':repeat,'backend':args.backend,
                     'prompt_token_ids_sha256':prompt_hash,'token_ids':ids,'top1_margins':margins,
                     'top2_token_ids_by_step':top2_ids,
                     'final_kv_length':kv_length,'elapsed_s':elapsed,'capacity':capacity,
                     'model_revision':MODEL_REVISION,'backend_metadata':loaded.metadata,
                     'execution_commit':execution_commit,'gpu':torch.cuda.get_device_name(args.device),
                     'torch':torch.__version__,'transformers':transformers.__version__,
                     'python':platform.python_version()}
                f.write(json.dumps(row)+'\n'); f.flush()
                print(f'{prompt_id} repeat={repeat+1}/{args.repeats} tokens={len(ids)}',flush=True)
    all_rows=[json.loads(x) for x in args.output.read_text().splitlines() if x.strip()]
    summary=[]
    for prompt_id in args.ids:
        items=sorted([r for r in all_rows if r['prompt_id']==prompt_id],key=lambda r:r['repeat'])
        ref=items[0]['token_ids'] if items else []
        diffs=[]
        for row in items[1:]:
            first=next((i for i,(a,b) in enumerate(zip(ref,row['token_ids'])) if a!=b),None)
            if first is None and len(ref)!=len(row['token_ids']): first=min(len(ref),len(row['token_ids']))
            diffs.append({'repeat':row['repeat'],'first_difference_index':first})
        summary.append({'prompt_id':prompt_id,'repeats':len(items),
                        'exact_to_repeat0':sum(r['token_ids']==ref for r in items),
                        'pairwise_first_diff_to_repeat0':diffs,
                        'min_observed_top1_margin':min((m for r in items for m in r['top1_margins']),default=None)})
    summary_path=args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps({'backend':args.backend,'summary':summary},indent=2)+'\n')
    print(summary_path)

if __name__=='__main__': main()
