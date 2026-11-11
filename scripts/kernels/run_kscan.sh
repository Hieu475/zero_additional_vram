#!/usr/bin/env bash
# K scan with all block sizes (2..K+1) captured; interleaved repeats to estimate run-to-run noise. Resumable.
cd "$(dirname "$0")/../.."
PY=${PY:-/home/nguyen_quoc_hieu/miniconda3/envs/ai/bin/python}
export PYTHONPATH=src:scripts/lib
B=data/benchmarks/humaneval_wp5_100.jsonl
IDS=$($PY -c "import json;print(' '.join(json.loads(l)['id'] for l in open('$B') if l.strip()))")
Q3=aa8e72537993ba99e69dfaafa59ed015b17504d1
for rep in 1 2 3; do for K in 2 3; do
  out=results/kernels/kscan/he100_k${K}_rep${rep}.json
  [ -e $out ] && continue
  $PY scripts/kernels/wp5_invariant_graph_pld.py --model Qwen/Qwen2.5-3B-Instruct --revision $Q3 --prompt-ids $IDS --benchmark $B --max-cache-len 1024 --all-m --k $K --ops lm_head,rmsnorm,sdpa,nf4 --output $out >/dev/null 2>&1
  $PY -c "import json;d=json.load(open('$out'));print('K',d['k'],'rep',$rep,'exact',d['all_exact'],'x',round(d['speedup_a3_over_graph_m1'],3),'tok/s',round(d['tok_per_s_graph_m1'],1),round(d['tok_per_s_a3'],1))" >> results/kernels/kscan/log.txt
done; done
echo DONE >> results/kernels/kscan/log.txt
