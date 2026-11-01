#!/usr/bin/env python3
"""Summarize paired K=1/K=2 full-fallback sweep across six prompts."""
from __future__ import annotations
import hashlib,json,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
PROMPTS={'HumanEval/112':'HumanEval112','HumanEval/114':'HumanEval114','HumanEval/121':'HumanEval121','HumanEval/123':'HumanEval123','cnndm_004':'cnndm_004','cnndm_003':'cnndm_003'}
OLD=hashlib.sha256((BASE/'fallback_k_sweep_512_manifest.json').read_bytes()).hexdigest()
NEW=hashlib.sha256((BASE/'fallback_k_sweep_512_expand_manifest.json').read_bytes()).hexdigest()
MANIFESTS={OLD,NEW}
def gm(xs):return math.exp(sum(math.log(float(x)) for x in xs)/len(xs))
def boot(xs,seed=15139,draws=20000):
 r=random.Random(seed);a=sorted(gm([r.choice(xs) for _ in xs]) for _ in range(draws));return [a[int(.025*draws)],a[int(.975*draws)]]
def main():
 prompts=[]; hashes={};matches={"k1_fallback_all":0,"k2_fallback_all":0}
 for pid,tag in PROMPTS.items():
  name=f'fallback_k_sweep_{tag}_512_v1.json';p=BASE/name;d=json.loads(p.read_text());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  if d.get('schema')!='zassd-fallback-k-sweep-512-v1' or d.get('prompt_id')!=pid or d.get('sweep_manifest_sha256') not in MANIFESTS:raise RuntimeError(f'invalid/unpinned artifact {name}')
  if not(all(d['b1_preflight']['token_match'].values()) and all(d['b1_preflight']['state_domains_equal'].values()) and not any(d['b1_preflight']['state_mismatches'].values()) and d['timed_candidate_token_match_b1']):raise RuntimeError(f'B1 exactness failed {name}')
  for m in matches:
   c=d['original_bnb_token_match_rounds'][m]
   if c!=d['timing_rounds']:raise RuntimeError(f'BNB token mismatch {name} {m}')
   matches[m]+=c
  modes={}
  for k in (1,2):
   mode=f'k{k}_fallback_all'; rows=[r['methods'][mode] for r in d['paired_rounds']]
   ratios=[r['methods']['original_bnb_greedy']['total_s']/r['methods'][mode]['total_s'] for r in d['paired_rounds']]
   modes[mode]={'speedup_vs_bnb_geomean':gm(ratios),'timing_speedups_vs_bnb':ratios,'latency_s_median':sorted(x['total_s'] for x in rows)[len(rows)//2-1:len(rows)//2+1] and __import__('statistics').median(x['total_s'] for x in rows),'cycles_median':__import__('statistics').median(x['cycles'] for x in rows),'draft_tokens_median':__import__('statistics').median(x['draft_tokens'] for x in rows),'accepted_tokens_median':__import__('statistics').median(x['accepted_draft_tokens'] for x in rows),'acceptance_fraction':sum(x['accepted_draft_tokens'] for x in rows)/sum(x['draft_tokens'] for x in rows),'target_rows_median':__import__('statistics').median(x['target_rows'] for x in rows)}
  k21=[r['methods']['k1_fallback_all']['total_s']/r['methods']['k2_fallback_all']['total_s'] for r in d['paired_rounds']]
  prompts.append({'prompt_id':pid,'empty_proposal_cycles':d['b1_preflight']['empty_proposal_cycles'],'k1_vs_k2':modes,'k2_speedup_over_k1_geomean':gm(k21),'k2_speedup_over_k1_rounds':k21,'artifact':name})
 aggregate={}
 for mode in ('k1_fallback_all','k2_fallback_all'):
  vs=[p['k1_vs_k2'][mode]['speedup_vs_bnb_geomean'] for p in prompts]
  aggregate[mode]={'geomean_across_prompts':gm(vs),'prompt_cluster_bootstrap_95pct':boot(vs),'prompt_min':min(vs),'prompt_max':max(vs),'prompts_faster_than_bnb':sum(v>1 for v in vs)}
 ks=[p['k2_speedup_over_k1_geomean'] for p in prompts]
 result={'schema':'zassd-fallback-k-sweep-summary-512-expanded-v1','classification':'six selected development prompts; one process and four paired timing rounds per prompt','scope':'Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, 512 tokens, physical_rows=3, native BNB fallback on all empty-proposal cycles','ratio_definition':'speedup = baseline latency / candidate latency; >1 means candidate faster','prompt_count':6,'independent_processes':6,'timing_rounds_per_k':24,'all_k_values_exact_to_b1_tokens_and_kv':True,'timed_tokens_match_b1':True,'original_bnb_token_matches_per_k':matches,'aggregate_speedup_vs_bnb':aggregate,'k2_speedup_over_k1':{'geomean_across_prompts':gm(ks),'prompt_cluster_bootstrap_95pct':boot(ks),'prompt_min':min(ks),'prompt_max':max(ks),'prompts_k2_faster':sum(v>1 for v in ks)},'prompts':prompts,'input_sha256':hashes,'manifest_sha256s':sorted(MANIFESTS),'heldout_used':False,'limits':['Six selected prompts, one process per prompt, one model and GPU; prompt bootstrap is descriptive.','Original-BNB comparison is token/latency only; no original-BNB KV identity claim.','GSM8K held-out indices 300-349 excluded.'],'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=BASE/'fallback_k_sweep_512_summary_v2.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'aggregate_speedup_vs_bnb':aggregate,'k2_speedup_over_k1':result['k2_speedup_over_k1'],'original_bnb_matches_per_k':matches,'output':str(out)},indent=2))
if __name__=='__main__':main()
