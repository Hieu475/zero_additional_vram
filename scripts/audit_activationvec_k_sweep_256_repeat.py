#!/usr/bin/env python3
"""Paired K=1 vs K=2 fallback-all comparison on selected 512-token prompts."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL,REVISION
from audit_activationvec_fallback_fraction_ablation_512 import MANIFEST as ABLATION_MANIFEST,validate_manifest,fallback_fraction_schedule,cache_observer,run_candidate
from _common import run_vanilla_timed
from zassd.decoding.b1_exact import b1_fixed_row_generate,b1_fixed_row_block_generate
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles'
MANIFEST=BASE/'fallback_k_sweep_256_repeat_manifest.json'
PROMPTS=[('data/benchmarks/humaneval_ext114.jsonl','HumanEval/112'),('data/benchmarks/humaneval_ext114.jsonl','HumanEval/114'),('data/benchmarks/humaneval_ext114.jsonl','HumanEval/121'),('data/benchmarks/humaneval_ext114.jsonl','HumanEval/123'),('data/benchmarks/cnndm_eval.jsonl','cnndm_004'),('data/benchmarks/cnndm_eval.jsonl','cnndm_003')]
KS=(1,2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate():
 m=json.loads(MANIFEST.read_text())
 if m.get('schema')!='zassd-fallback-k-sweep-manifest-v1':raise RuntimeError('bad K-sweep manifest')
 for f,h in m['source_sha256'].items():
  if sha(ROOT/f)!=h:raise RuntimeError(f'source pin mismatch: {f}')
 for f,h in m['benchmark_sha256'].items():
  if sha(ROOT/f)!=h:raise RuntimeError(f'benchmark pin mismatch: {f}')
 if sha(ABLATION_MANIFEST)!=m['ablation_manifest_sha256']:raise RuntimeError('ablation manifest changed')
 validate_manifest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prompt-id',choices=[x[1] for x in PROMPTS],required=True);ap.add_argument('--timing-rounds',type=int,default=4)
 a=ap.parse_args()
 if a.timing_rounds<2:raise ValueError('at least two timing rounds required')
 validate(); benchmark,pid=next(x for x in PROMPTS if x[1]==a.prompt_id)
 out=BASE/f"fallback_k_sweep_{pid.replace('/','')}_256_rep2_v1.json"
 if out.exists():raise FileExistsError(out)
 rows=[json.loads(x) for x in (ROOT/benchmark).read_text().splitlines() if x.strip()];prompt=next(x for x in rows if str(x['id'])==pid)
 loaded=load_backend('hf_bnb_nf4',MODEL,REVISION,'cuda:0','sdpa');model,tokenizer=loaded.model.eval(),loaded.tokenizer
 prompt_ids=tokenizer(prompt['prompt'],return_tensors='pt')['input_ids'].to('cuda:0');plen=int(prompt_ids.shape[-1]);capacity=plen+264
 ref_states={'ref':{}}
 ref_common=dict(k=2,ngram_size=3,max_new_tokens=256,device='cuda:0',kv_cache_max_capacity=capacity,physical_rows=3,attention_strategy='bnb_rowwise',quantized_linear_strategy='fixed_rows')
 with torch.inference_mode():
  _,ref=b1_fixed_row_generate(model,tokenizer,prompt_ids,**ref_common,state_observer=cache_observer(ref_states['ref'],plen))
 preflight={};states={f'k{k}':{} for k in KS};controls={}
 for k in KS:
  common={**ref_common,'k':k,'trace':True};control={'no_proposal_cycles':0,'fallback_forwards':0,'empty_cycle_index':0,'use_native':False}
  with fallback_fraction_schedule(model,control,1),torch.inference_mode():
   _,metrics=b1_fixed_row_block_generate(model,tokenizer,prompt_ids,**common,state_observer=cache_observer(states[f'k{k}'],plen))
  preflight[f'k{k}']=metrics;controls[f'k{k}']=control
 tokens={f'k{k}':preflight[f'k{k}'].generated_token_ids==ref.generated_token_ids for k in KS}
 domains={f'k{k}':set(states[f'k{k}'])==set(ref_states['ref']) for k in KS}
 mismatches={f'k{k}':[i for i in sorted(set(states[f'k{k}'])&set(ref_states['ref'])) if states[f'k{k}'][i]!=ref_states['ref'][i]] for k in KS}
 if not(all(tokens.values()) and all(domains.values()) and not any(mismatches.values())):raise RuntimeError(f'preflight B1 exactness failed: {tokens} {mismatches}')
 with torch.inference_mode():
  run_vanilla_timed(model,tokenizer,prompt_ids,8,'cuda:0')
  for k in KS:
   c={'no_proposal_cycles':0,'fallback_forwards':0,'empty_cycle_index':0,'use_native':False}
   with fallback_fraction_schedule(model,c,1):b1_fixed_row_block_generate(model,tokenizer,prompt_ids,**{**ref_common,'k':k,'max_new_tokens':8,'trace':True})
 methods=['original_bnb_greedy','k1_fallback_all','k2_fallback_all'];paired=[]
 for rid in range(a.timing_rounds):
  order=methods[rid%len(methods):]+methods[:rid%len(methods)];sample={'round':rid,'method_order':order,'methods':{}}
  for method in order:
   torch.cuda.synchronize('cuda:0')
   if method=='original_bnb_greedy':
    ids,pre,dec,total=run_vanilla_timed(model,tokenizer,prompt_ids,256,'cuda:0')
    sample['methods'][method]={'token_ids':ids,'total_s':total/1000,'prefill_s':pre/1000,'decode_s':dec/1000}
   else:
    k=1 if method.startswith('k1') else 2;c={'no_proposal_cycles':0,'fallback_forwards':0,'empty_cycle_index':0,'use_native':False}
    with fallback_fraction_schedule(model,c,1),torch.inference_mode():_,m=b1_fixed_row_block_generate(model,tokenizer,prompt_ids,**{**ref_common,'k':k,'trace':True})
    sample['methods'][method]={'token_ids':m.generated_token_ids,'total_s':m.total_time_s,'prefill_s':m.prefill_time_s,'decode_s':m.total_time_s-m.prefill_time_s,'verify_s':m.verify_time_s,'draft_s':m.draft_time_s,'cycles':m.cycles,'draft_tokens':m.draft_tokens,'accepted_draft_tokens':m.accepted_draft_tokens,'target_rows':m.target_rows,'fallback_count':c['fallback_forwards']}
  sample['candidate_token_match_b1']={m:sample['methods'][m]['token_ids']==ref.generated_token_ids for m in methods[1:]}
  sample['candidate_token_match_bnb']={m:sample['methods'][m]['token_ids']==sample['methods']['original_bnb_greedy']['token_ids'] for m in methods[1:]}
  if not all(sample['candidate_token_match_b1'].values()):raise RuntimeError(f'timed B1 mismatch: {pid} round {rid}')
  paired.append(sample)
 result={'schema':'zassd-fallback-k-sweep-256-v1','classification':'paired K=1/K=2 fallback-all development comparison','model':MODEL,'revision':REVISION,'gpu':torch.cuda.get_device_name(0),'prompt_id':pid,'prompt_sha256':hashlib.sha256(prompt['prompt'].encode()).hexdigest(),'max_new_tokens':256,'timing_rounds':a.timing_rounds,'physical_rows':3,'attention':'bnb_rowwise','fallback_policy':'native BNB M=1 on every empty-proposal cycle','b1_preflight':{'token_match':tokens,'state_domains_equal':domains,'state_mismatches':mismatches,'shared_prefix_count':len(ref_states['ref']),'empty_proposal_cycles':{k:controls[k]['no_proposal_cycles'] for k in controls}},'timed_candidate_token_match_b1':True,'original_bnb_token_match_rounds':{m:sum(r['candidate_token_match_bnb'][m] for r in paired) for m in methods[1:]},'paired_rounds':paired,'speedup_vs_bnb_medians':{m:statistics.median([r['methods']['original_bnb_greedy']['total_s']/r['methods'][m]['total_s'] for r in paired]) for m in methods[1:]},'k2_over_k1_latency_ratio_median':statistics.median([r['methods']['k1_fallback_all']['total_s']/r['methods']['k2_fallback_all']['total_s'] for r in paired]),'ablation_manifest_sha256':sha(ABLATION_MANIFEST),'sweep_manifest_sha256':sha(MANIFEST),'runner_sha256':sha(__file__),'heldout_used':False,'limits':['Six selected development prompts on one GPU; paired rounds nested within prompt.','Original-BNB comparison has no KV-state identity claim.','GSM8K held-out indices 300-349 excluded.']}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'prompt_id':pid,'exactness':tokens,'speedup_vs_bnb_medians':result['speedup_vs_bnb_medians'],'k2_over_k1_latency_ratio_median':result['k2_over_k1_latency_ratio_median'],'output':str(out)},indent=2))
if __name__=='__main__':main()
