#!/usr/bin/env python3
"""Summarize two independent 256-token ngram sweeps per prompt across six prompts."""
from __future__ import annotations
import hashlib,json,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
PROMPTS={'HumanEval/112':'HumanEval112','HumanEval/114':'HumanEval114','HumanEval/121':'HumanEval121','HumanEval/123':'HumanEval123','cnndm_004':'cnndm_004','cnndm_003':'cnndm_003'}
MANIFESTS={hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ('fallback_ngram_sweep_256_manifest.json','fallback_ngram_sweep_256_repeat_manifest.json')}
def gm(xs):return math.exp(sum(math.log(float(v)) for v in xs)/len(xs))
def boot(xs,seed=24917,draws=20000):
 r=random.Random(seed);a=sorted(gm([r.choice(xs) for _ in xs]) for _ in range(draws));return [a[int(.025*draws)],a[int(.975*draws)]]
def main():
 prompts=[];hashes={};bnb_counts={f'ngram{g}':0 for g in (2,3,4)};processes=0
 for pid,tag in PROMPTS.items():
  bnb={f'ngram{g}':[] for g in (2,3,4)};vs3={f'ngram{g}':[] for g in (2,4)};sessions=[];accept={g:[0,0] for g in (2,3,4)}
  for suffix in ('v1','repeat_v1'):
   name=f'fallback_ngram_sweep_{tag}_256_{suffix}.json';p=BASE/name;d=json.loads(p.read_text());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
   if d.get('schema')!='zassd-fallback-ngram-sweep-256-v1' or d.get('prompt_id')!=pid or d.get('sweep_manifest_sha256') not in MANIFESTS:raise RuntimeError(f'invalid/unpinned {name}')
   if not(all(d['b1_preflight']['token_match'].values()) and all(d['b1_preflight']['state_domains_equal'].values()) and not any(d['b1_preflight']['state_mismatches'].values()) and d['timed_token_match_b1']):raise RuntimeError(f'B1 exactness failed {name}')
   processes+=1
   session_ratios={f'ngram{g}':[] for g in (2,3,4)};session_vs3={f'ngram{g}':[] for g in (2,4)}
   for g in (2,3,4):
    mode=f'ngram{g}_fallback_all'
    if d['original_bnb_token_match_rounds'][f'ng{g}']!=d['timing_rounds']:raise RuntimeError(f'BNB mismatch {name}:{g}')
    bnb_counts[f'ngram{g}']+=d['original_bnb_token_match_rounds'][f'ng{g}']
   for row in d['paired_rounds']:
    base=row['methods']['original_bnb_greedy']['total_s']
    for g in (2,3,4):
     name_mode=f'ngram{g}_fallback_all';v=base/row['methods'][name_mode]['total_s'];bnb[f'ngram{g}'].append(v);session_ratios[f'ngram{g}'].append(v)
     m=row['methods'][name_mode];accept[g][0]+=m['accepted_draft_tokens'];accept[g][1]+=m['draft_tokens']
    for g in (2,4):
     ratio=row['methods']['ngram3_fallback_all']['total_s']/row['methods'][f'ngram{g}_fallback_all']['total_s'];vs3[f'ngram{g}'].append(ratio);session_vs3[f'ngram{g}'].append(ratio)
   sessions.append({'artifact':name,'speedup_vs_bnb':{k:gm(v) for k,v in session_ratios.items()},'speedup_vs_ngram3':{k:gm(v) for k,v in session_vs3.items()},'original_bnb_token_matches':d['original_bnb_token_match_rounds']})
  prompts.append({'prompt_id':pid,'independent_processes':2,'speedup_vs_bnb':{k:gm(v) for k,v in bnb.items()},'speedup_vs_ngram3':{k:gm(v) for k,v in vs3.items()},'acceptance_fraction':{f'ngram{g}':accept[g][0]/accept[g][1] for g in (2,3,4)},'sessions':sessions})
 agg={}
 for g in (2,3,4):
  key=f'ngram{g}';xs=[p['speedup_vs_bnb'][key] for p in prompts]
  agg[key]={'geomean_across_prompts':gm(xs),'prompt_cluster_bootstrap_95pct':boot(xs),'prompt_min':min(xs),'prompt_max':max(xs),'prompts_faster_than_bnb':sum(v>1 for v in xs)}
 inc={}
 for g in (2,4):
  key=f'ngram{g}';xs=[p['speedup_vs_ngram3'][key] for p in prompts]
  inc[key]={'geomean_across_prompts':gm(xs),'prompt_cluster_bootstrap_95pct':boot(xs),'prompt_min':min(xs),'prompt_max':max(xs),'prompts_faster_than_ngram3':sum(v>1 for v in xs)}
 result={'schema':'zassd-fallback-ngram-sweep-summary-256-repeat-v1','classification':'six selected development prompts; two independent processes and eight paired rounds per prompt','scope':'Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K=2/R=3, 256 tokens, full no-proposal fallback','ratio_definition':'speedup = baseline latency / candidate latency; >1 means candidate faster','prompt_count':6,'independent_processes':processes,'timing_rounds_per_ngram':48,'all_ngram_sizes_exact_to_b1_token_and_kv':True,'timed_tokens_match_b1':True,'original_bnb_token_matches':bnb_counts,'aggregate_speedup_vs_bnb':agg,'aggregate_speedup_vs_ngram3':inc,'prompts':prompts,'input_sha256':hashes,'manifest_sha256s':sorted(MANIFESTS),'heldout_used':False,'limits':['Six selected prompts, two processes each, one model and GPU; prompt bootstrap is descriptive.','Original-BNB comparison is token/latency only; no BNB KV-state identity claim.','GSM8K held-out indices 300-349 excluded.'],'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=BASE/'fallback_ngram_sweep_256_summary_v2.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'aggregate_speedup_vs_bnb':agg,'aggregate_speedup_vs_ngram3':inc,'bnb_matches_per_ngram':bnb_counts,'processes':processes,'output':str(out)},indent=2))
if __name__=='__main__':main()
