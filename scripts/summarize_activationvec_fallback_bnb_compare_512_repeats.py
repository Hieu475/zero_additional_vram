#!/usr/bin/env python3
"""Summarize two independent paired BNB comparison processes per prompt."""
from __future__ import annotations
import hashlib,json,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
PROMPTS={'HumanEval/112':'HumanEval112','HumanEval/114':'HumanEval114','HumanEval/121':'HumanEval121','HumanEval/123':'HumanEval123','cnndm_004':'cnndm_004','cnndm_003':'cnndm_003'}
MODES=('activationvec','fallback_every_other','fallback_all')
MANIFESTS={hashlib.sha256((BASE/'fallback_fraction_bnb_compare_512_manifest.json').read_bytes()).hexdigest(),hashlib.sha256((BASE/'fallback_fraction_bnb_compare_512_repeat_manifest.json').read_bytes()).hexdigest()}
ABLATION_HASH=hashlib.sha256((BASE/'fallback_fraction_ablation_512_manifest.json').read_bytes()).hexdigest()
def gmean(xs):return math.exp(sum(math.log(float(x)) for x in xs)/len(xs))
def bootstrap(xs,seed=125903,draws=20000):
 r=random.Random(seed); a=sorted(gmean([r.choice(xs) for _ in xs]) for _ in range(draws));return [a[int(.025*draws)],a[int(.975*draws)]]
def main():
 prompts=[]; hashes={}; total_tokens={m:0 for m in MODES}; exact_processes=0
 for pid,tag in PROMPTS.items():
  sessions=[]
  for suffix in ('v1','rep2_v1'):
   name=f'fallback_bnb_compare_{tag}_512_{suffix}.json'; p=BASE/name; d=json.loads(p.read_text()); hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
   if d.get('schema')!='zassd-fallback-bnb-compare-512-v1' or d.get('prompt_id')!=pid or d.get('max_new_tokens')!=512: raise RuntimeError(f'invalid artifact {name}')
   if d.get('comparison_manifest_sha256') not in MANIFESTS or d.get('ablation_manifest_sha256')!=ABLATION_HASH: raise RuntimeError(f'manifest mismatch {name}')
   if not (all(d['b1_preflight']['token_match'].values()) and all(d['b1_preflight']['state_domains_equal'].values()) and not any(d['b1_preflight']['state_mismatches'].values()) and d['timed_candidate_tokens_match_b1']): raise RuntimeError(f'B1 exactness failed {name}')
   exact_processes+=1
   for m in MODES:
    total_tokens[m]+=d['original_bnb_token_match_rounds'][m]
   bnb=d['bnb_over_candidate_speedup_per_round']
   inc={m:[r['methods']['activationvec']['total_s']/r['methods'][m]['total_s'] for r in d['paired_rounds']] for m in ('fallback_every_other','fallback_all')}
   sessions.append({'artifact':name,'rounds':d['timing_rounds'],'empty_proposal_cycles':d['b1_preflight']['empty_proposal_cycles'],'speedup_vs_bnb':{m:gmean(bnb[m]) for m in MODES},'speedup_vs_activationvec':{m:gmean(inc[m]) for m in inc},'bnb_token_match_counts':d['original_bnb_token_match_rounds']})
  per_prompt={m:gmean([v for s in sessions for v in json.loads((BASE/s['artifact']).read_text())['bnb_over_candidate_speedup_per_round'][m]]) for m in MODES}
  inc_prompt={m:gmean([v for s in sessions for v in [r['methods']['activationvec']['total_s']/r['methods'][m]['total_s'] for r in json.loads((BASE/s['artifact']).read_text())['paired_rounds']]]) for m in ('fallback_every_other','fallback_all')}
  prompts.append({'prompt_id':pid,'independent_processes':2,'speedup_vs_bnb':per_prompt,'speedup_vs_activationvec':inc_prompt,'sessions':sessions})
 aggregate={}
 for mode in MODES:
  xs=[p['speedup_vs_bnb'][mode] for p in prompts]
  aggregate[mode]={'geomean_across_prompts':gmean(xs),'prompt_cluster_bootstrap_95pct':bootstrap(xs),'prompt_min':min(xs),'prompt_max':max(xs),'prompts_faster_than_bnb':sum(x>1 for x in xs)}
 incremental={}
 for mode in ('fallback_every_other','fallback_all'):
  xs=[p['speedup_vs_activationvec'][mode] for p in prompts]
  incremental[mode]={'geomean_across_prompts':gmean(xs),'prompt_cluster_bootstrap_95pct':bootstrap(xs),'prompt_min':min(xs),'prompt_max':max(xs),'prompts_improved':sum(x>1 for x in xs)}
 result={'schema':'zassd-fallback-bnb-compare-summary-512-repeat-v1','classification':'six selected development prompt clusters; two independent paired processes per prompt; descriptive cluster bootstrap','runtime_scope':'Qwen/Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K=2/R=3, 512 tokens, four paired rounds per process','ratio_definition':'speedup = original BNB greedy latency / candidate latency; >1 means candidate faster','prompt_count':6,'independent_processes':exact_processes,'timing_rounds_per_mode':48,'b1_exact_processes':exact_processes,'timed_candidate_token_match_b1':True,'original_bnb_token_match_rounds_per_mode':total_tokens,'aggregate_speedup_vs_bnb':aggregate,'fallback_incremental_speedup_vs_activationvec':incremental,'prompts':prompts,'input_sha256':hashes,'ablation_manifest_sha256':ABLATION_HASH,'comparison_manifest_sha256s':sorted(MANIFESTS),'heldout_used':False,'limits':['Six selected development prompts, one model and one GPU; prompt bootstrap is descriptive.','Original-BNB comparison is latency/token-ID only; no original-BNB KV-state identity claim.','GSM8K held-out indices 300-349 excluded.'],'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=BASE/'fallback_bnb_compare_512_summary_v2.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'aggregate_speedup_vs_bnb':aggregate,'fallback_incremental_speedup_vs_activationvec':incremental,'processes':exact_processes,'original_bnb_matches_per_mode':total_tokens,'output':str(out)},indent=2))
if __name__=='__main__':main()
