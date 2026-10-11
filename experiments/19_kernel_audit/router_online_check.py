"""GPU check: legacy routed vs online-calibrated routed (6 GSM8K prompts, 48 tok)."""
import sys, json
sys.path.insert(0, "/home/nguyen_quoc_hieu/Documents/zero_additional_vram")
from zassd.models.loader import load_tokenizer, load_model
from zassd.models.model_adapter import ModelAdapter
from zassd.models.layer_manager import LayerManager
from zassd.decoding.speculative import self_speculative_generate
from zassd.decoding.vanilla import vanilla_generate
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.profiling.online_cost import OnlineLatencyTracker, CalibratedAcceptancePredictor, RegretDecomposition
from zassd.routing.hybrid_router import HybridDraftRouter
from zassd.cli import SKIP_STRATEGIES

recs=[]
with open("data/benchmarks/gsm8k_eval.jsonl") as f:
    for line in f:
        if line.strip(): recs.append(json.loads(line))
        if len(recs)>=6: break
prompts=[r["prompt"] for r in recs]
tok=load_tokenizer("Qwen/Qwen2.5-3B-Instruct")
model=load_model("Qwen/Qwen2.5-3B-Instruct", quantize=True, bits=4, device="cuda:0"); model.eval()
adapt=ModelAdapter(model); lm=LayerManager(adapt)
skip=SKIP_STRATEGIES["empirical_6"] if adapt.num_layers==36 else list(range(6))
r_leg=HybridDraftRouter(cost_model=MeasuredActionCostModel.for_model("qwen25_3b"), pld_k=4)
r_on=HybridDraftRouter(cost_model=MeasuredActionCostModel.for_model("qwen25_3b"), pld_k=4,
    latency_tracker=OnlineLatencyTracker(), alpha_predictor=CalibratedAcceptancePredictor())
out={"legacy":[],"online":[]}
for pr in prompts:
    _,mv=vanilla_generate(model,tok,pr,max_new_tokens=48,device="cuda:0")
    _,ml=self_speculative_generate(model,tok,lm,skip,pr,k=2,draft_mode="routed",router=r_leg,config_name="cka_75",max_new_tokens=48,device="cuda:0")
    _,mo=self_speculative_generate(model,tok,lm,skip,pr,k=2,draft_mode="routed",router=r_on,config_name="cka_75",max_new_tokens=48,device="cuda:0")
    sl=ml.tokens_per_second/max(mv.tokens_per_second,1e-9); so=mo.tokens_per_second/max(mv.tokens_per_second,1e-9)
    out["legacy"].append(sl); out["online"].append(so)
    print(f"leg={sl:.3f}(a={ml.acceptance_rate:.2f}) on={so:.3f}(a={mo.acceptance_rate:.2f}) van={mv.tokens_per_second:.1f}", flush=True)
import statistics
mlm=statistics.mean(out["legacy"]); mom=statistics.mean(out["online"])
print(f"LEGACY mean={mlm:.3f} ONLINE mean={mom:.3f}", flush=True)
print("CALIB "+json.dumps(r_on.calibration_report()), flush=True)
print("TRACKER n_vanilla="+str(r_on.latency_tracker.n_vanilla_obs), flush=True)
json.dump({"legacy":out["legacy"],"online":out["online"],"calib":r_on.calibration_report()}, open("/tmp/opencode/router_online_check.json","w"), indent=2)
print("DONE", flush=True)
