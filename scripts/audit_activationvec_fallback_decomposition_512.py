#!/usr/bin/env python3
"""Paired original-BNB latency comparison for the 512-token fallback schedules."""
from __future__ import annotations
import argparse, hashlib, json, math, random, statistics
from pathlib import Path
import torch
from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_activationvec_fallback_fraction_ablation_512 import (
    MANIFEST as ABLATION_MANIFEST, validate_manifest, fallback_fraction_schedule,
    cache_observer, run_candidate,
)
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from run_strong_baseline_comparison import run_vanilla_timed
from zassd.decoding.b1_exact import b1_fixed_row_generate, b1_fixed_row_block_generate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/fallback_fraction_bnb_decomposition_512_manifest.json"
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
MODES = {"activationvec": 0, "fallback_every_other": 2, "fallback_all": 1}
PROMPTS = [
    ("data/benchmarks/humaneval_ext114.jsonl", "HumanEval/112"),
    ("data/benchmarks/humaneval_ext114.jsonl", "HumanEval/114"),
    ("data/benchmarks/humaneval_ext114.jsonl", "HumanEval/121"),
    ("data/benchmarks/humaneval_ext114.jsonl", "HumanEval/123"),
    ("data/benchmarks/cnndm_eval.jsonl", "cnndm_004"),
    ("data/benchmarks/cnndm_eval.jsonl", "cnndm_003"),
]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_comparison_manifest():
    m=json.loads(MANIFEST.read_text())
    if m.get("schema") != "zassd-fallback-fraction-bnb-compare-manifest-v1":
        raise RuntimeError("invalid comparison manifest")
    for rel, expected in m["source_sha256"].items():
        if sha(ROOT/rel) != expected: raise RuntimeError(f"comparison source pin mismatch: {rel}")
    if sha(ABLATION_MANIFEST) != m["ablation_manifest_sha256"]:
        raise RuntimeError("base ablation manifest changed")
    for rel, expected in m["benchmark_sha256"].items():
        if sha(ROOT/rel) != expected: raise RuntimeError(f"benchmark changed: {rel}")
    validate_manifest()

def prompt_row(benchmark, prompt_id):
    rows=[json.loads(x) for x in (ROOT/benchmark).read_text().splitlines() if x.strip()]
    row=next((r for r in rows if str(r["id"]) == prompt_id),None)
    if row is None: raise KeyError(prompt_id)
    return row

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timing-rounds", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--prompt-id", choices=[p[1] for p in PROMPTS], help="run one prompt; omit for the full cohort")
    args=ap.parse_args()
    if args.timing_rounds < 2 or args.max_new_tokens != 512:
        raise ValueError("comparison manifest is pinned to 512 tokens and at least two timing rounds")
    validate_comparison_manifest()
    results=[]
    selected=[p for p in PROMPTS if args.prompt_id is None or p[1] == args.prompt_id]
    for benchmark,prompt_id in selected:
        out=BASE/f"fallback_bnb_decomposition_{prompt_id.replace('/', '')}_512_v1.json"
        if out.exists(): raise FileExistsError(out)
        prompt=prompt_row(benchmark,prompt_id)
        loaded=load_backend("hf_bnb_nf4",MODEL,REVISION,"cuda:0","sdpa")
        model,tokenizer=loaded.model.eval(),loaded.tokenizer
        prompt_ids=tokenizer(prompt["prompt"],return_tensors="pt")["input_ids"].to("cuda:0")
        prompt_len=int(prompt_ids.shape[-1]); capacity=prompt_len+512+8
        common=dict(k=2,ngram_size=3,max_new_tokens=512,device="cuda:0",kv_cache_max_capacity=capacity,
                    physical_rows=3,attention_strategy="bnb_rowwise",quantized_linear_strategy="fixed_rows",trace=True)
        singleton_states={"singleton":{}}
        singleton_common={k:v for k,v in common.items() if k!="trace"}
        with torch.inference_mode():
            _,singleton=b1_fixed_row_generate(model,tokenizer,prompt_ids,**singleton_common,
                state_observer=cache_observer(singleton_states["singleton"],prompt_len))
        preflight={}; states={n:{} for n in MODES}
        for name,stride in MODES.items():
            metrics,control=run_candidate(model,tokenizer,prompt_ids,common,prompt_len,stride,states,name)
            preflight[name]={"metrics":metrics,"control":control}
        mismatches={n:[p for p in sorted(set(singleton_states["singleton"]) & set(states[n]))
                       if singleton_states["singleton"][p]!=states[n][p]] for n in MODES}
        domains={n:set(states[n])==set(singleton_states["singleton"]) for n in MODES}
        tokens={n:preflight[n]["metrics"].generated_token_ids==singleton.generated_token_ids for n in MODES}
        if not (all(tokens.values()) and all(domains.values()) and not any(mismatches.values())):
            raise RuntimeError(f"preflight B1 exactness failed for {prompt_id}: tokens={tokens}, state={mismatches}")
        # Warm all four methods equally before paired measurement.
        with torch.inference_mode():
            run_vanilla_timed(model,tokenizer,prompt_ids,8,"cuda:0")
            for stride in MODES.values():
                c={"no_proposal_cycles":0,"fallback_forwards":0,"empty_cycle_index":0,"use_native":False}
                with fallback_fraction_schedule(model,c,stride):
                    b1_fixed_row_block_generate(model,tokenizer,prompt_ids,**{**common,"max_new_tokens":8})
        paired=[]; method_names=["original_bnb_greedy",*MODES]
        for rid in range(args.timing_rounds):
            order=method_names[rid%len(method_names):]+method_names[:rid%len(method_names)]
            sample={"round":rid,"method_order":order,"methods":{}}
            for method in order:
                torch.cuda.synchronize("cuda:0")
                if method=="original_bnb_greedy":
                    ids,prefill_ms,decode_ms,total_ms=run_vanilla_timed(model,tokenizer,prompt_ids,512,"cuda:0")
                    sample["methods"][method]={"token_ids":ids,"total_s":total_ms/1000,
                        "prefill_s":prefill_ms/1000,"decode_s":decode_ms/1000}
                else:
                    stride=MODES[method]; c={"no_proposal_cycles":0,"fallback_forwards":0,"empty_cycle_index":0,"use_native":False}
                    with fallback_fraction_schedule(model,c,stride),torch.inference_mode():
                        _,metrics=b1_fixed_row_block_generate(model,tokenizer,prompt_ids,**common)
                    sample["methods"][method]={"token_ids":metrics.generated_token_ids,"total_s":metrics.total_time_s,
                        "prefill_s":metrics.prefill_time_s,"decode_s":metrics.total_time_s-metrics.prefill_time_s,
                        "draft_s":metrics.draft_time_s,"verify_s":metrics.verify_time_s,
                        "other_decode_s":max(0.0,metrics.total_time_s-metrics.prefill_time_s-metrics.draft_time_s-metrics.verify_time_s),
                        "cycles":metrics.cycles,"draft_tokens":metrics.draft_tokens,
                        "accepted_draft_tokens":metrics.accepted_draft_tokens,"target_forwards":metrics.target_forwards,
                        "target_rows":metrics.target_rows,"fallback_counts":c}
            sample["candidate_token_match_b1"]={n:sample["methods"][n]["token_ids"]==singleton.generated_token_ids for n in MODES}
            sample["candidate_token_match_original_bnb"]={n:sample["methods"][n]["token_ids"]==sample["methods"]["original_bnb_greedy"]["token_ids"] for n in MODES}
            if not all(sample["candidate_token_match_b1"].values()): raise RuntimeError(f"timed B1 mismatch {prompt_id} round {rid}")
            paired.append(sample)
        ratios={n:[r["methods"]["original_bnb_greedy"]["total_s"]/r["methods"][n]["total_s"] for r in paired] for n in MODES}
        payload={"schema":"zassd-fallback-decomposition-512-v1","classification":"decode timing decomposition; paired original-BNB comparison; diagnostic only",
            "model":MODEL,"revision":REVISION,"gpu":torch.cuda.get_device_name(0),"prompt_id":prompt_id,
            "benchmark":benchmark,"prompt_sha256":hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
            "max_new_tokens":512,"timing_rounds":args.timing_rounds,"schedule":{"k":2,"ngram_size":3,"physical_rows":3,"attention":"bnb_rowwise"},
            "b1_preflight":{"token_match":tokens,"state_domains_equal":domains,"state_mismatches":mismatches,
                "shared_prefix_count":len(singleton_states["singleton"]),
                "empty_proposal_cycles":preflight["fallback_all"]["control"]["no_proposal_cycles"]},
            "timed_candidate_tokens_match_b1":True,"paired_rounds":paired,
            "bnb_over_candidate_speedup_medians":{n:statistics.median(v) for n,v in ratios.items()},
            "bnb_over_candidate_speedup_per_round":ratios,
            "original_bnb_token_match_rounds":{n:sum(r["candidate_token_match_original_bnb"][n] for r in paired) for n in MODES},
            "ablation_manifest_sha256":sha(ABLATION_MANIFEST),"comparison_manifest_sha256":sha(MANIFEST),
            "runner_sha256":sha(__file__),"heldout_used":False,
            "limits":["Latency-only comparison to original BNB; no original-BNB KV-state identity claim.",
                      "Two selected development prompts on one model and GPU; paired rounds nested within process.",
                      "GSM8K held-out indices 300-349 excluded."]}
        out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n")
        results.append({"prompt_id":prompt_id,"empty_proposal_cycles":payload["b1_preflight"]["empty_proposal_cycles"],
            "speedup_medians":payload["bnb_over_candidate_speedup_medians"],"original_bnb_token_match_rounds":payload["original_bnb_token_match_rounds"],"output":str(out)})
        del loaded,model
        torch.cuda.empty_cache()
    print(json.dumps({"completed":len(results),"results":results},indent=2))
if __name__=="__main__": main()
