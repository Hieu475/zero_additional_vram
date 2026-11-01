#!/usr/bin/env python3
"""Summarize two-process K=1/K=2 paired sweep across six 512-token prompts."""
from __future__ import annotations
import hashlib,json,math,random,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
PROMPTS={'HumanEval/112':'HumanEval112','HumanEval/114':'HumanEval114','HumanEval/121':'HumanEval121','HumanEval/123':'HumanEval123','cnndm_004':'cnndm_004','cnndm_003':'cnndm_003'}
MANIFESTS={hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ('fallback_k_sweep_512_manifest.json','fallback_k_sweep_512_expand_manifest.json','fallback_k_sweep_512_repeat_manifest.json')}
def gm(xs):return math.exp(sum(math.log(float(x)) for x in xs)/len(xs))
def boot(xs,seed=18793,draws=20000):
 r=random.Random(seed);v=sorted(gm([r.choice(xs) for _ in xs]) for _ in range(draws));return [v[int(.025*draws)],v[int(.975*draws)]]
def main():
 prompts=[];hashes={};bnb_counts={'k1_fallback_all':0,'k2_fallback_all':0};process_count=0
 for pid,tag in PROMPTS.items():
  sessions=[];ratios={m:[] for m in ('k1_fallback_all','k2_fallback_all')}; k21=[];drafts={1:0,2:0};accepts={1:0,2:0}
  for suffix in ('v1','rep2_v1'):
   name=f'fallback_k_sweep_{tag}_512_{suffix}.json';p=BASE/name;d=json.loads(p.read_text());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
   if d.get('schema')!='zassd-fallback-k-sweep-512-v1' or d.get('prompt_id')!=pid or d.get('sweep_manifest_sha256') not in MANIFESTS:raise RuntimeError(f'invalid/unpinned artifact {name}')
   if not(all(d['b1_preflight']['token_match'].values()) and all(d['b1_preflight']['state_domains_equal'].values()) and not any(d['b1_preflight']['state_mismatches'].values()) and d['timed_candidate_token_match_b1']):raise RuntimeError(f'B1 exactness failed {name}')
   process_count+=1; ratios_this={m:[] for m in ratios}; k21_this=[]
   for row in d['paired_rounds']:
    b=row['methods']['original_bnb_greedy']['total_s']
    for k,m in ((1,'k1_fallback_all'),(2,'k2_fallback_all')):
     ratios_this[m].append(b/row['methods'][m]['total_s']);bnb_counts[m]+=int(row['candidate_token_match_bnb'][m])
     drafts[k]+=row['methods'][m]['draft_tokens'];accepts[k]+=row['methods'][m]['accepted_draft_tokens']
    k21_this.append(row['methods']['k1_fallback_all']['total_s']/row['methods']['k2_fallback_all']['total_s'])
   for m in ratios:ratios[m].extend(ratios_this[m])
   k21.extend(k21_this)
   sessions.append({'artifact':name,'speedup_vs_bnb':{m:gm(v) for m,v in ratios_this.items()},'k2_speedup_over_k1':gm(k21_this),'bnb_token_matches':d['original_bnb_token_match_rounds'],'empty_proposal_cycles':d['b1_preflight']['empty_proposal_cycles']})
  per_mode={m:gm(v) for m,v in ratios.items()}
  prompts.append({'prompt_id':pid,'independent_processes':2,'speedup_vs_bnb':per_mode,'k2_speedup_over_k1':gm(k21),'acceptance_fraction':{'k1':accepts[1]/drafts[1],'k2':accepts[2]/drafts[2]},'sessions':sessions})
 aggregate={}
 for m in ('k1_fallback_all','k2_fallback_all'):
  xs=[p['speedup_vs_bnb'][m] for p in prompts]
  aggregate[m]={'geomean_across_prompts':gm(xs),'prompt_cluster_bootstrap_95pct':boot(xs),'prompt_min':min(xs),'prompt_max':max(xs),'prompts_faster_than_bnb':sum(x>1 for x in xs)}
 ks=[p['k2_speedup_over_k1'] for p in prompts]
 result={'schema':'zassd-fallback-k-sweep-summary-512-repeat-v1','classification':'six selected development prompts; two independent processes and eight paired rounds per prompt','scope':'Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, 512 tokens, physical_rows=3, native BNB fallback on every empty-proposal cycle','ratio_definition':'speedup = baseline latency / candidate latency; >1 means candidate faster','prompt_count':6,'independent_processes':process_count,'timing_rounds_per_k':48,'all_k_values_exact_to_b1_tokens_and_kv':True,'timed_candidate_tokens_match_b1':True,'original_bnb_token_matches_per_k':bnb_counts,'aggregate_speedup_vs_bnb':aggregate,'k2_speedup_over_k1':{'geomean_across_prompts':gm(ks),'prompt_cluster_bootstrap_95pct':boot(ks),'prompt_min':min(ks),'prompt_max':max(ks),'prompts_k2_faster':sum(x>1 for x in ks)},'prompts':prompts,'input_sha256':hashes,'manifest_sha256s':sorted(MANIFESTS),'heldout_used':False,'limits':['Six selected prompts, two processes per prompt, one model and GPU; prompt-cluster bootstrap is descriptive.','Original-BNB comparison is token/latency only; no original-BNB KV-state identity claim.','GSM8K held-out indices 300-349 excluded.'],'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=BASE/'fallback_k_sweep_512_summary_v3.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'aggregate_speedup_vs_bnb':aggregate,'k2_speedup_over_k1':result['k2_speedup_over_k1'],'bnb_token_matches':bnb_counts,'processes':process_count,'output':str(out)},indent=2))
if __name__=='__main__':main()
