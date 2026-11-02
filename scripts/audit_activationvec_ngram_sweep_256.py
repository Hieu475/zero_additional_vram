#!/usr/bin/env python3
"""Paired ngram-size sweep for K=2 with full no-proposal fallback."""
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
MANIFEST=BASE/'fallback_ngram_sweep_256_manifest.json'
PROMPTS=[('data/benchmarks/humaneval_ext114.jsonl',f'HumanEval/{i}') for i in (112,114,121,123)]+[('data/benchmarks/cnndm_eval.jsonl',f'cnndm_{i:03d}') for i in (3,4)]
GRAMS=(2,3,4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate():
 m=json.loads(MANIFEST.read_text())
 if m.get('schema')!='zassd-fallback-ngram-sweep-manifest-v1':raise RuntimeError('bad ngram sweep manifest')
 for f,h in m['source_sha256'].items():
  if sha(ROOT/f)!=h:raise RuntimeError(f'source pin mismatch: {f}')
 for f,h in m['benchmark_sha256'].items():
  if sha(ROOT/f)!=h:raise RuntimeError(f'benchmark pin mismatch: {f}')
 if sha(ABLATION_MANIFEST)!=m['ablation_manifest_sha256']:raise RuntimeError('ablation manifest changed')
 validate_manifest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prompt-id',choices=[p[1] for p in PROMPTS],required=True);ap.add_argument('--timing-rounds',type=int,default=4);args=ap.parse_args()
 if args.timing_rounds<2:raise ValueError('at least two paired rounds required')
 validate();benchmark,pid=next(x for x in PROMPTS if x[1]==args.prompt_id)
 out=BASE/f"fallback_ngram_sweep_{pid.replace('/','')}_256_v1.json"
 if out.exists():raise FileExistsError(out)
 rows=[json.loads(x) for x in (ROOT/benchmark).read_text().splitlines() if x.strip()];prompt=next(x for x in rows if str(x['id'])==pid)
 loaded=load_backend('hf_bnb_nf4',MODEL,REVISION,'cuda:0','sdpa');model,tokenizer=loaded.model.eval(),loaded.tokenizer
 prompt_ids=tokenizer(prompt['prompt'],return_tensors='pt')['input_ids'].to('cuda:0');plen=int(prompt_ids.shape[-1]);capacity=plen+264
 base_common=dict(k=2,ngram_size=3,max_new_tokens=256,device='cuda:0',kv_cache_max_capacity=capacity,physical_rows=3,attention_strategy='bnb_rowwise',quantized_linear_strategy='fixed_rows')
 ref_states={'ref':{}}
 with torch.inference_mode():_,ref=b1_fixed_row_generate(model,tokenizer,prompt_ids,**base_common,state_observer=cache_observer(ref_states['ref'],plen))
 preflight={};states={f'ng{g}':{} for g in GRAMS};controls={}
 for g in GRAMS:
  common={**base_common,'ngram_size':g,'trace':True};c={'no_proposal_cycles':0,'fallback_forwards':0,'empty_cycle_index':0,'use_native':False}
  m,c=run_candidate(model,tokenizer,prompt_ids,common,plen,1,states,f'ng{g}')
  preflight[f'ng{g}']=m;controls[f'ng{g}']=c
 token_ok={f'ng{g}':preflight[f'ng{g}'].generated_token_ids==ref.generated_token_ids for g in GRAMS}
 domain_ok={f'ng{g}':set(states[f'ng{g}'])==set(ref_states['ref']) for g in GRAMS}
 mismatches={f'ng{g}':[i for i in sorted(set(states[f'ng{g}'])&set(ref_states['ref'])) if states[f'ng{g}'][i]!=ref_states['ref'][i]] for g in GRAMS}
 if not(all(token_ok.values()) and all(domain_ok.values()) and not any(mismatches.values())):raise RuntimeError(f'B1 exactness failed: {token_ok} {mismatches}')
 with torch.inference_mode():
  run_vanilla_timed(model,tokenizer,prompt_ids,8,'cuda:0')
  for g in GRAMS:
   c={'no_proposal_cycles':0,'fallback_forwards':0,'empty_cycle_index':0,'use_native':False}
   with fallback_fraction_schedule(model,c,1):b1_fixed_row_block_generate(model,tokenizer,prompt_ids,**{**base_common,'ngram_size':g,'max_new_tokens':8,'trace':True})
 methods=['original_bnb_greedy',*[f'ngram{g}_fallback_all' for g in GRAMS]];paired=[]
 for rid in range(args.timing_rounds):
  order=methods[rid%len(methods):]+methods[:rid%len(methods)];sample={'round':rid,'method_order':order,'methods':{}}
  for method in order:
   torch.cuda.synchronize('cuda:0')
   if method=='original_bnb_greedy':
    ids,pref,dec,total=run_vanilla_timed(model,tokenizer,prompt_ids,256,'cuda:0');sample['methods'][method]={'token_ids':ids,'total_s':total/1000,'prefill_s':pref/1000,'decode_s':dec/1000}
   else:
    g=int(method[5]);c={'no_proposal_cycles':0,'fallback_forwards':0,'empty_cycle_index':0,'use_native':False}
    with fallback_fraction_schedule(model,c,1),torch.inference_mode():_,m=b1_fixed_row_block_generate(model,tokenizer,prompt_ids,**{**base_common,'ngram_size':g,'trace':True})
    sample['methods'][method]={'token_ids':m.generated_token_ids,'total_s':m.total_time_s,'prefill_s':m.prefill_time_s,'decode_s':m.total_time_s-m.prefill_time_s,'draft_s':m.draft_time_s,'verify_s':m.verify_time_s,'cycles':m.cycles,'draft_tokens':m.draft_tokens,'accepted_draft_tokens':m.accepted_draft_tokens,'target_rows':m.target_rows,'fallback_forwards':c['fallback_forwards'],'empty_proposal_cycles':c['no_proposal_cycles']}
  sample['candidate_token_match_b1']={f'ng{g}':sample['methods'][f'ngram{g}_fallback_all']['token_ids']==ref.generated_token_ids for g in GRAMS}
  sample['candidate_token_match_bnb']={f'ng{g}':sample['methods'][f'ngram{g}_fallback_all']['token_ids']==sample['methods']['original_bnb_greedy']['token_ids'] for g in GRAMS}
  if not all(sample['candidate_token_match_b1'].values()):raise RuntimeError(f'timed B1 mismatch {pid} round {rid}')
  paired.append(sample)
 results={}
 for g in GRAMS:
  name=f'ngram{g}_fallback_all';results[name]={'speedup_vs_bnb_median':statistics.median([r['methods']['original_bnb_greedy']['total_s']/r['methods'][name]['total_s'] for r in paired]),'speedup_vs_ngram3_median':statistics.median([r['methods']['ngram3_fallback_all']['total_s']/r['methods'][name]['total_s'] for r in paired]) if g!=3 else 1.0}
 payload={'schema':'zassd-fallback-ngram-sweep-256-v1','classification':'paired ngram-size development sweep; full native-BNB empty-proposal fallback','model':MODEL,'revision':REVISION,'gpu':torch.cuda.get_device_name(0),'prompt_id':pid,'benchmark':benchmark,'prompt_sha256':hashlib.sha256(prompt['prompt'].encode()).hexdigest(),'max_new_tokens':256,'timing_rounds':args.timing_rounds,'schedule':{'k':2,'ngram_sizes':list(GRAMS),'physical_rows':3,'attention':'bnb_rowwise','fallback':'native BNB M=1 on every empty-proposal cycle'},'b1_preflight':{'token_match':token_ok,'state_domains_equal':domain_ok,'state_mismatches':mismatches,'shared_prefix_count':len(ref_states['ref']),'empty_proposal_cycles':{f'ng{g}':controls[f'ng{g}']['no_proposal_cycles'] for g in GRAMS}},'timed_token_match_b1':True,'original_bnb_token_match_rounds':{f'ng{g}':sum(r['candidate_token_match_bnb'][f'ng{g}'] for r in paired) for g in GRAMS},'paired_rounds':paired,'ratios':results,'ablation_manifest_sha256':sha(ABLATION_MANIFEST),'sweep_manifest_sha256':sha(MANIFEST),'runner_sha256':sha(__file__),'heldout_used':False,'limits':['Six selected development prompts on one model and GPU; paired rounds are nested within prompt.','Original-BNB comparison is latency/token-ID only; no BNB KV-state identity claim.','GSM8K held-out indices 300-349 excluded.']}
 out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'prompt_id':pid,'exactness':token_ok,'cycles':payload['b1_preflight']['empty_proposal_cycles'],'ratios':results,'bnb_matches':payload['original_bnb_token_match_rounds'],'output':str(out)},indent=2))
if __name__=='__main__':main()
