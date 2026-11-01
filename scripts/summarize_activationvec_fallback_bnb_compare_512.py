#!/usr/bin/env python3
"""Summarize paired original-BNB latency comparison for 512-token fallback runs."""
from __future__ import annotations
import hashlib,json,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
FILES={'HumanEval/112':'fallback_bnb_compare_HumanEval112_512_v1.json','HumanEval/114':'fallback_bnb_compare_HumanEval114_512_v1.json','HumanEval/121':'fallback_bnb_compare_HumanEval121_512_v1.json','HumanEval/123':'fallback_bnb_compare_HumanEval123_512_v1.json','cnndm_004':'fallback_bnb_compare_cnndm_004_512_v1.json','cnndm_003':'fallback_bnb_compare_cnndm_003_512_v1.json'}
MODES=('activationvec','fallback_every_other','fallback_all')
def gmean(xs): return math.exp(sum(math.log(float(x)) for x in xs)/len(xs))
def bootstrap(xs,seed=85177,draws=20000):
 rng=random.Random(seed); vals=sorted(gmean([rng.choice(xs) for _ in xs]) for _ in range(draws))
 return [vals[int(.025*draws)],vals[int(.975*draws)]]
def main():
 mh=hashlib.sha256((BASE/'fallback_fraction_bnb_compare_512_manifest.json').read_bytes()).hexdigest()
 ah=hashlib.sha256((BASE/'fallback_fraction_ablation_512_manifest.json').read_bytes()).hexdigest()
 prompts=[]; hashes={}; bnb_matches=True
 for pid,name in FILES.items():
  p=BASE/name; d=json.loads(p.read_text()); hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  if d.get('schema')!='zassd-fallback-bnb-compare-512-v1' or d.get('prompt_id')!=pid or d.get('max_new_tokens')!=512: raise RuntimeError(f'invalid artifact {name}')
  if d.get('comparison_manifest_sha256')!=mh or d.get('ablation_manifest_sha256')!=ah: raise RuntimeError(f'manifest mismatch {name}')
  if not (all(d['b1_preflight']['token_match'].values()) and all(d['b1_preflight']['state_domains_equal'].values()) and not any(d['b1_preflight']['state_mismatches'].values()) and d['timed_candidate_tokens_match_b1']): raise RuntimeError(f'B1 exactness failed {name}')
  counts=d['original_bnb_token_match_rounds']
  if any(counts[m]!=d['timing_rounds'] for m in MODES): bnb_matches=False
  ratios=d['bnb_over_candidate_speedup_per_round']
  within={m:[r['methods']['activationvec']['total_s']/r['methods'][m]['total_s'] for r in d['paired_rounds']] for m in ('fallback_every_other','fallback_all')}
  prompts.append({'prompt_id':pid,'empty_proposal_cycles':d['b1_preflight']['empty_proposal_cycles'],'timing_rounds':d['timing_rounds'],'original_bnb_token_matches':counts,'paired_speedup_geomean':{m:gmean(ratios[m]) for m in MODES},'paired_speedup_rounds':ratios,'fallback_speedup_vs_activationvec':{m:gmean(v) for m,v in within.items()},'artifact':name})
 aggregate={}
 for mode in MODES:
  vs=[p['paired_speedup_geomean'][mode] for p in prompts]
  aggregate[mode]={'geomean_across_prompts':gmean(vs),'prompt_cluster_bootstrap_95pct':bootstrap(vs),'prompt_min':min(vs),'prompt_max':max(vs),'prompts_faster_than_original_bnb':sum(v>1 for v in vs)}
 within_aggregate={m:{'geomean_across_prompts':gmean([p['fallback_speedup_vs_activationvec'][m] for p in prompts]),'prompt_cluster_bootstrap_95pct':bootstrap([p['fallback_speedup_vs_activationvec'][m] for p in prompts]),'prompt_min':min(p['fallback_speedup_vs_activationvec'][m] for p in prompts),'prompt_max':max(p['fallback_speedup_vs_activationvec'][m] for p in prompts)} for m in ('fallback_every_other','fallback_all')}
 result={'schema':'zassd-fallback-bnb-compare-summary-512-v1','classification':'six selected development prompts; paired original-BNB latency comparison; descriptive prompt-cluster bootstrap','runtime_scope':'Qwen/Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K=2/R=3, 512 tokens, four paired rounds per prompt','ratio_definition':'speedup = original BNB greedy latency / candidate latency; >1 means candidate faster','prompt_count':len(prompts),'timing_rounds_per_prompt':4,'all_candidate_modes_exact_to_b1_tokens_and_kv':True,'all_candidate_timed_tokens_match_b1':True,'all_candidate_tokens_match_original_bnb_in_all_rounds':bnb_matches,'aggregate':aggregate,'fallback_incremental_speedup_vs_activationvec':within_aggregate,'prompts':prompts,'input_sha256':hashes,'comparison_manifest_sha256':mh,'ablation_manifest_sha256':ah,'heldout_used':False,'limits':['Six selected development prompts, one model and one GPU; prompt-cluster bootstrap is descriptive.','Original-BNB comparison is latency/token-ID only; no original-BNB KV-state identity claim.','GSM8K held-out indices 300-349 excluded.'],'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=BASE/'fallback_bnb_compare_512_summary_v1.json'; out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'prompt_count':len(prompts),'aggregate':aggregate,'all_bnb_token_matches':bnb_matches,'output':str(out)},indent=2))
if __name__=='__main__': main()
