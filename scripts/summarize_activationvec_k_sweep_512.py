#!/usr/bin/env python3
"""Summarize paired K=1/K=2 results on two development prompts."""
from __future__ import annotations
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
PROMPTS={'HumanEval/123':'HumanEval123','HumanEval/114':'HumanEval114'}
MANIFEST_HASH=hashlib.sha256((BASE/'fallback_k_sweep_512_manifest.json').read_bytes()).hexdigest()
def gmean(x):return math.exp(sum(math.log(v) for v in x)/len(x))
def main():
 prompts=[];hashes={}
 for pid,tag in PROMPTS.items():
  name=f'fallback_k_sweep_{tag}_512_v1.json';p=BASE/name;d=json.loads(p.read_text());hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  if d.get('schema')!='zassd-fallback-k-sweep-512-v1' or d.get('prompt_id')!=pid or d.get('sweep_manifest_sha256')!=MANIFEST_HASH: raise RuntimeError(f'invalid/unpinned artifact {name}')
  if not (all(d['b1_preflight']['token_match'].values()) and all(d['b1_preflight']['state_domains_equal'].values()) and not any(d['b1_preflight']['state_mismatches'].values()) and d['timed_candidate_token_match_b1']):raise RuntimeError(f'B1 exactness failed {name}')
  if any(n!=d['timing_rounds'] for n in d['original_bnb_token_match_rounds'].values()):raise RuntimeError(f'BNB token mismatch {name}')
  modes={}
  for k in (1,2):
   key=f'k{k}_fallback_all';rows=[r['methods'][key] for r in d['paired_rounds']]
   modes[key]={'speedup_vs_bnb_geomean':gmean([r['methods']['original_bnb_greedy']['total_s']/r['methods'][key]['total_s'] for r in d['paired_rounds']]),'latency_s_median':statistics.median(r['total_s'] for r in rows),'decode_s_median':statistics.median(r['decode_s'] for r in rows),'cycles_median':statistics.median(r['cycles'] for r in rows),'draft_tokens_median':statistics.median(r['draft_tokens'] for r in rows),'accepted_tokens_median':statistics.median(r['accepted_draft_tokens'] for r in rows),'acceptance_fraction':sum(r['accepted_draft_tokens'] for r in rows)/sum(r['draft_tokens'] for r in rows),'target_rows_median':statistics.median(r['target_rows'] for r in rows)}
  prompts.append({'prompt_id':pid,'empty_proposal_cycles':d['b1_preflight']['empty_proposal_cycles'],'original_bnb_token_matches':d['original_bnb_token_match_rounds'],'k1_vs_k2':modes,'k2_speedup_over_k1':d['k2_over_k1_latency_ratio_median'],'artifact':name})
 result={'schema':'zassd-fallback-k-sweep-summary-512-v1','classification':'two selected prompts, one process each, four paired rounds; descriptive K sweep','scope':'Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, 512 tokens, physical_rows=3, native BNB fallback on every empty-proposal cycle','ratio_definition':'speedup = baseline latency / candidate latency; >1 means candidate faster','prompts':prompts,'prompt_count':2,'process_count':2,'all_k_values_exact_to_b1_token_and_kv':True,'all_timed_tokens_match_b1':True,'all_timed_tokens_match_original_bnb':True,'geomean_across_two_prompts':{'k1_vs_bnb':gmean([x['k1_fallback_all']['speedup_vs_bnb_geomean'] for p in prompts for x in [p['k1_vs_k2']]]),'k2_vs_bnb':gmean([p['k1_vs_k2']['k2_fallback_all']['speedup_vs_bnb_geomean'] for p in prompts]),'k2_speedup_over_k1':gmean([p['k2_speedup_over_k1'] for p in prompts])},'input_sha256':hashes,'manifest_sha256':MANIFEST_HASH,'heldout_used':False,'limits':['Only two selected prompts and one process per prompt; descriptive evidence, no population claim.','Original-BNB comparison is token/latency only; no original-BNB KV identity claim.','GSM8K held-out indices 300-349 excluded.'],'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=BASE/'fallback_k_sweep_512_summary_v1.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'geomean':result['geomean_across_two_prompts'],'prompts':prompts,'output':str(out)},indent=2))
if __name__=='__main__':main()
