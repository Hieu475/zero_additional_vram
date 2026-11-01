#!/usr/bin/env python3
"""Summarize six-prompt 256-token K=2 ngram-size sweep."""
from __future__ import annotations
import hashlib,json,math,random,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
PROMPTS={'HumanEval/112':'HumanEval112','HumanEval/114':'HumanEval114','HumanEval/121':'HumanEval121','HumanEval/123':'HumanEval123','cnndm_004':'cnndm_004','cnndm_003':'cnndm_003'}
MHS={hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ('fallback_ngram_sweep_256_manifest.json',)}
def gm(x):return math.exp(sum(math.log(float(v)) for v in x)/len(x))
def boot(xs,seed=37379,draws=20000):
 r=random.Random(seed);v=sorted(gm([r.choice(xs) for _ in xs]) for _ in range(draws));return [v[int(.025*draws)],v[int(.975*draws)]]
def main():
 prompts=[];hashes={};bnb_counts={f'ngram{g}':0 for g in (2,3,4)}
 for pid,tag in PROMPTS.items():
  name=f'fallback_ngram_sweep_{tag}_256_v1.json';p=BASE/name;d=json.loads(p.read_text());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  if d.get('schema')!='zassd-fallback-ngram-sweep-256-v1' or d.get('prompt_id')!=pid or d.get('sweep_manifest_sha256') not in MHS:raise RuntimeError(f'invalid/unpinned {name}')
  if not(all(d['b1_preflight']['token_match'].values()) and all(d['b1_preflight']['state_domains_equal'].values()) and not any(d['b1_preflight']['state_mismatches'].values()) and d['timed_token_match_b1']):raise RuntimeError(f'B1 mismatch {name}')
  ratios={f'ngram{g}':[] for g in (2,3,4)};vs3={f'ngram{g}':[] for g in (2,4)};mode_stats={}
  for g in (2,3,4):
   mode=f'ngram{g}_fallback_all';rows=[r['methods'][mode] for r in d['paired_rounds']]
   if d['original_bnb_token_match_rounds'][f'ng{g}']!=d['timing_rounds']:raise RuntimeError(f'BNB token mismatch {name} {g}')
   bnb_counts[f'ngram{g}']+=d['original_bnb_token_match_rounds'][f'ng{g}']
   mode_stats[f'ngram{g}']={'empty_proposal_cycles':d['b1_preflight']['empty_proposal_cycles'][f'ng{g}'],'cycles_median':statistics.median(x['cycles'] for x in rows),'draft_tokens_median':statistics.median(x['draft_tokens'] for x in rows),'accepted_tokens_median':statistics.median(x['accepted_draft_tokens'] for x in rows),'acceptance_fraction':sum(x['accepted_draft_tokens'] for x in rows)/sum(x['draft_tokens'] for x in rows),'target_rows_median':statistics.median(x['target_rows'] for x in rows)}
   for r in d['paired_rounds']:
    b=r['methods']['original_bnb_greedy']['total_s'];ratios[f'ngram{g}'].append(b/r['methods'][mode]['total_s'])
    if g in (2,4):vs3[f'ngram{g}'].append(r['methods']['ngram3_fallback_all']['total_s']/r['methods'][mode]['total_s'])
  prompts.append({'prompt_id':pid,'speedup_vs_bnb':{k:gm(v) for k,v in ratios.items()},'speedup_vs_ngram3':{k:gm(v) for k,v in vs3.items()},'ngram_metrics':mode_stats,'artifact':name})
 aggregate={}
 for g in (2,3,4):
  key=f'ngram{g}';xs=[p['speedup_vs_bnb'][key] for p in prompts]
  aggregate[key]={'geomean_across_prompts':gm(xs),'prompt_cluster_bootstrap_95pct':boot(xs),'prompt_min':min(xs),'prompt_max':max(xs),'prompts_faster_than_bnb':sum(x>1 for x in xs)}
 against3={}
 for g in (2,4):
  key=f'ngram{g}';xs=[p['speedup_vs_ngram3'][key] for p in prompts]
  against3[key]={'geomean_across_prompts':gm(xs),'prompt_cluster_bootstrap_95pct':boot(xs),'prompt_min':min(xs),'prompt_max':max(xs),'prompts_faster_than_ngram3':sum(x>1 for x in xs)}
 result={'schema':'zassd-fallback-ngram-sweep-summary-256-v1','classification':'six selected development prompts; one process and four paired timing rounds per prompt','scope':'Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K=2/R=3, 256 tokens, full no-proposal fallback','ratio_definition':'speedup = baseline latency / candidate latency; >1 means candidate faster','prompt_count':6,'process_count':6,'timing_rounds_per_ngram':24,'all_ngram_sizes_exact_to_b1_token_and_kv':True,'timed_tokens_match_b1':True,'original_bnb_token_matches':bnb_counts,'aggregate_speedup_vs_bnb':aggregate,'aggregate_speedup_vs_ngram3':against3,'prompts':prompts,'input_sha256':hashes,'manifest_sha256s':sorted(MHS),'heldout_used':False,'limits':['Six selected prompts, one process each, one model and GPU; prompt bootstrap is descriptive.','Original-BNB comparison is token/latency only; no BNB KV-state identity claim.','GSM8K held-out indices 300-349 excluded.'],'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=BASE/'fallback_ngram_sweep_256_summary_v1.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'aggregate_speedup_vs_bnb':aggregate,'aggregate_speedup_vs_ngram3':against3,'bnb_matches':bnb_counts,'output':str(out)},indent=2))
if __name__=='__main__':main()
