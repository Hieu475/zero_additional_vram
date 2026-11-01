#!/usr/bin/env python3
"""Summarize same-process 256-token decode phase and acceptance diagnostics."""
from __future__ import annotations
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
PROMPTS={'HumanEval/123':'HumanEval123','HumanEval/114':'HumanEval114'}
MODES=('original_bnb_greedy','activationvec','fallback_every_other','fallback_all')
MANIFEST_HASH=hashlib.sha256((BASE/'fallback_fraction_bnb_decomposition_256_manifest.json').read_bytes()).hexdigest()
def median(rows,key): return statistics.median(r[key] for r in rows)
def main():
 prompts=[]; input_hashes={}
 for pid,tag in PROMPTS.items():
  name=f'fallback_bnb_decomposition_{tag}_256_v1.json'; p=BASE/name; d=json.loads(p.read_text()); input_hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  if d.get('schema')!='zassd-fallback-decomposition-256-v1' or d.get('prompt_id')!=pid or d.get('comparison_manifest_sha256')!=MANIFEST_HASH: raise RuntimeError(f'invalid or unpinned profile {name}')
  if not (all(d['b1_preflight']['token_match'].values()) and all(d['b1_preflight']['state_domains_equal'].values()) and not any(d['b1_preflight']['state_mismatches'].values()) and d['timed_candidate_tokens_match_b1']): raise RuntimeError(f'B1 mismatch {name}')
  if any(v!=d['timing_rounds'] for v in d['original_bnb_token_match_rounds'].values()): raise RuntimeError(f'BNB token mismatch {name}')
  modes={}
  for m in MODES:
   rows=[r['methods'][m] for r in d['paired_rounds']]
   fields=['total_s','prefill_s','decode_s'] if m=='original_bnb_greedy' else ['total_s','prefill_s','decode_s','draft_s','verify_s','other_decode_s','cycles','draft_tokens','accepted_draft_tokens','target_rows']
   modes[m]={k:median(rows,k) for k in fields}
   if m!='original_bnb_greedy': modes[m]['acceptance_fraction']=sum(r['accepted_draft_tokens'] for r in rows)/max(1,sum(r['draft_tokens'] for r in rows))
  prompts.append({'prompt_id':pid,'empty_proposal_cycles':d['b1_preflight']['empty_proposal_cycles'],'b1_exact':True,'original_bnb_token_matches_per_mode':d['original_bnb_token_match_rounds'],'median_metrics':modes})
 result={'schema':'zassd-fallback-decomposition-summary-512-v1','classification':'diagnostic same-process timing decomposition; two selected prompts; not a kernel profiler result','scope':'Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K=2/R=3, 256 output tokens, four paired rounds per prompt','ratio_definition':'speedup >1 means baseline time divided by candidate time is greater than one','prompts':prompts,'prompt_count':2,'process_count':2,'all_candidate_tokens_and_kv_exact_to_b1':True,'all_candidate_tokens_match_original_bnb':True,'input_sha256':input_hashes,'manifest_sha256':MANIFEST_HASH,'heldout_used':False,'limits':['Two selected HumanEval prompts on one GPU; timing medians are descriptive.','This is decoder phase timing and acceptance accounting, not CUDA-kernel attribution.','Original-BNB KV-state identity is not claimed.','GSM8K held-out indices 300-349 excluded.'],'source':str(Path(__file__).resolve()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 out=BASE/'fallback_bnb_decomposition_256_summary_v1.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'output':str(out),'prompts':[{p['prompt_id']:p['median_metrics']} for p in prompts]},indent=2))
if __name__=='__main__':main()
