#!/usr/bin/env python3
"""Summarize paired ngram-size sweep for two HumanEval prompts."""
from __future__ import annotations
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
PROMPTS={'HumanEval/123':'HumanEval123','HumanEval/114':'HumanEval114'}
MH=hashlib.sha256((BASE/'fallback_ngram_sweep_512_manifest.json').read_bytes()).hexdigest()
def gm(x):return math.exp(sum(math.log(float(v)) for v in x)/len(x))
def main():
 ps=[];hashes={};matches={f'ngram{g}':0 for g in (2,3,4)}
 for pid,tag in PROMPTS.items():
  name=f'fallback_ngram_sweep_{tag}_512_v1.json';p=BASE/name;d=json.loads(p.read_text());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  if d.get('schema')!='zassd-fallback-ngram-sweep-512-v1' or d.get('prompt_id')!=pid or d.get('sweep_manifest_sha256')!=MH:raise RuntimeError(f'invalid/unpinned {name}')
  if not(all(d['b1_preflight']['token_match'].values()) and all(d['b1_preflight']['state_domains_equal'].values()) and not any(d['b1_preflight']['state_mismatches'].values()) and d['timed_token_match_b1']):raise RuntimeError(f'B1 exactness failed {name}')
  grams={}
  for g in (2,3,4):
   m=f'ngram{g}_fallback_all';rows=[x['methods'][m] for x in d['paired_rounds']]
   if d['original_bnb_token_match_rounds'][f'ng{g}']!=d['timing_rounds']:raise RuntimeError(f'BNB token mismatch {name} {g}')
   matches[f'ngram{g}']+=d['original_bnb_token_match_rounds'][f'ng{g}']
   ratios=[x['methods']['original_bnb_greedy']['total_s']/x['methods'][m]['total_s'] for x in d['paired_rounds']]
   grams[f'ngram{g}']={'speedup_vs_bnb_geomean':gm(ratios),'speedup_vs_bnb_rounds':ratios,'speedup_vs_ngram3':d['ratios'][m]['speedup_vs_ngram3_median'],'empty_proposal_cycles':d['b1_preflight']['empty_proposal_cycles'][f'ng{g}'],'cycles_median':statistics.median(x['cycles'] for x in rows),'draft_tokens_median':statistics.median(x['draft_tokens'] for x in rows),'accepted_tokens_median':statistics.median(x['accepted_draft_tokens'] for x in rows),'acceptance_fraction':sum(x['accepted_draft_tokens'] for x in rows)/sum(x['draft_tokens'] for x in rows),'target_rows_median':statistics.median(x['target_rows'] for x in rows)}
  ps.append({'prompt_id':pid,'ngrams':grams})
 aggregate={f'ngram{g}':{'geomean_across_two_prompts':gm([p['ngrams'][f'ngram{g}']['speedup_vs_bnb_geomean'] for p in ps]),'prompts_faster_than_bnb':sum(p['ngrams'][f'ngram{g}']['speedup_vs_bnb_geomean']>1 for p in ps)} for g in (2,3,4)}
 result={'schema':'zassd-fallback-ngram-sweep-summary-512-v1','classification':'two selected prompts; one process per prompt; paired ngram sweep','scope':'Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K=2/R=3, 512 tokens, full no-proposal fallback','ratio_definition':'speedup = original BNB latency / candidate latency; >1 means candidate faster','prompt_count':2,'process_count':2,'timing_rounds_per_ngram':8,'all_ngram_sizes_exact_to_b1_token_and_kv':True,'timed_tokens_match_b1':True,'original_bnb_token_matches':matches,'aggregate':aggregate,'prompts':ps,'input_sha256':hashes,'manifest_sha256':MH,'heldout_used':False,'limits':['Two selected HumanEval prompts, one process per prompt and one GPU; descriptive only.','Original-BNB comparison is token/latency only; no BNB KV-state claim.','GSM8K held-out indices 300-349 excluded.'],'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=BASE/'fallback_ngram_sweep_512_summary_v1.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'aggregate':aggregate,'bnb_token_matches':matches,'output':str(out)},indent=2))
if __name__=='__main__':main()
