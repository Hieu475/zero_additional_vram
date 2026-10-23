#!/usr/bin/env python3
"""Analyze paired BNB/HQQ Nsight captures from one matched workload."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'experiments/backend_validation/kernel_profile'
METHODS=('vanilla','pld_k4','cka90_k2')
BACKEND_PREFIX={'hf_bnb_nf4':('bnb_matched','bnb_matched'),
                'hqq_gemlite_w4a16':('hqq','hqq')}


def read_json(path: Path): return json.loads(path.read_text())
def kernel_name(row): return row.get('Name','')
def matching(rows,fragment):
    found=[r for r in rows if fragment.lower() in kernel_name(r).lower()]
    return {'time_ns':sum(r.get('Total Time (ns)',0) for r in found),
            'instances':sum(r.get('Instances',r.get('Num Calls',0)) for r in found)}


def profile(backend,method):
    prefix,stats_prefix=BACKEND_PREFIX[backend]
    kernel=read_json(DIR/f'{stats_prefix}_{method}_stats_cuda_gpu_kern_sum.json')
    api=read_json(DIR/f'{stats_prefix}_{method}_stats_cuda_api_sum.json')
    nvtx=read_json(DIR/f'{stats_prefix}_{method}_stats_nvtx_sum.json')
    mem=read_json(DIR/f'{stats_prefix}_{method}_stats_cuda_gpu_mem_time_sum.json')
    raw=read_json(DIR/f'{prefix}_{method}.json')
    total=sum(r.get('Total Time (ns)',0) for r in kernel)
    sync=[r for r in api if 'Synchronize' in r.get('Name','')]
    launch=[r for r in api if r.get('Name') in ('cudaLaunchKernel','cuLaunchKernelEx')]
    ranges={r.get('Range','').lstrip(':'):r for r in nvtx}
    return {
        'backend':backend,'method':method,'raw_json':str((DIR/f'{prefix}_{method}.json').relative_to(ROOT)),
        'capture':str((DIR/f'{prefix}_{method}.nsys-rep').relative_to(ROOT)),
        'prompt_id':raw['prompt_id'],'prompt_token_ids_sha256':raw['prompt_token_ids_sha256'],
        'input_tokens':raw['prompt_tokens'],'requested_output_tokens':raw['max_new_tokens'],
        'generated_tokens':len(raw['measured_token_ids']),'capacity':raw['kv_capacity'],
        'reported_measured_seconds_under_profiler':raw['measured_e2e_seconds_under_profiler'],
        'peak_allocated_mb':raw['peak_allocated_mb'],'peak_reserved_mb':raw['peak_reserved_mb'],
        'total_gpu_kernel_time_ns':total,'kernel_instances':sum(r.get('Instances',0) for r in kernel),
        'synchronization_api_ns':sum(r.get('Total Time (ns)',0) for r in sync),
        'synchronization_calls':sum(r.get('Num Calls',0) for r in sync),
        'launch_api_ns':sum(r.get('Total Time (ns)',0) for r in launch),
        'launch_calls':sum(r.get('Num Calls',0) for r in launch),
        'method_and_phase_ranges':{k:{'total_ns':v.get('Total Time (ns)'), 'instances':v.get('Instances')}
            for k,v in ranges.items() if k in ('ncu_measurement','vanilla_decode','vanilla_prefill','pld_k4_total','cka90_k2_total','draft_forward','target_verify','kv_crop','kv_commit')},
        'top_kernels':[{'name':r.get('Name'),'total_ns':r.get('Total Time (ns)'),
                        'instances':r.get('Instances'),'share_pct':r.get('Time (%)')}
                       for r in kernel[:8]],
        'memory_operations':mem,
        'backend_metadata':raw.get('backend_metadata'),
        'execution_commit':raw.get('execution_commit'),
        'software':{'torch':raw.get('torch'),'cuda':raw.get('cuda'),
                    'transformers':raw.get('transformers'),'backend_versions':raw.get('backend_versions')},
        'gpu':raw.get('gpu'),'compute_capability':raw.get('compute_capability'),
        'warmup_tokens':raw.get('warmup_token_count'),
    }


def main():
    rows=[profile(b,m) for b in BACKEND_PREFIX for m in METHODS]
    hashes={}
    for row in rows:
        for key in ('raw_json','capture'):
            p=ROOT/row[key]; hashes[row[key]]=hashlib.sha256(p.read_bytes()).hexdigest()
        for kind in ('cuda_gpu_kern_sum','cuda_api_sum','nvtx_sum','cuda_gpu_mem_time_sum'):
            p=DIR/f"{BACKEND_PREFIX[row['backend']][1]}_{row['method']}_stats_{kind}.json"
            hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    payload={
        'scope':'single prompt, one process per cell, 32-token full warmup, 32-token measured Nsight Systems capture; diagnostic only',
        'matched_inputs':len({r['prompt_token_ids_sha256'] for r in rows})==1,
        'matched_capacity':len({r['capacity'] for r in rows})==1,
        'matched_requested_output_tokens':len({r['requested_output_tokens'] for r in rows})==1,
        'execution_commits':sorted({r['execution_commit'] for r in rows}),
        'profiles':rows,'sha256':hashes,
        'limits':['Quantized weights differ between BNB NF4 and HQQ W4A16, so cross-backend kernel differences are stack comparisons, not an isolated kernel treatment.',
        'Nsight instrumentation changes timings; the TPS fields are not production benchmarks.',
        'One prompt and one session per cell cannot support general performance claims.',
        'The host environment disabled CPU context-switch tracing, so host scheduling gaps are not directly observable.',
        'Nsight Systems captures API and kernel activity; no matched FLOP/byte or arithmetic-intensity measurement is included.'],
    }
    out=DIR/'paired_backend_profiles.json'
    out.write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Paired BNB/HQQ kernel profile','',
           'Fixed Qwen2.5-3B prompt `qasper_119_ctx384` (384 input tokens), 32 measured output tokens, static KV capacity 436, SDPA, 32-token full warmup. Each capture is one separate process, instrumented by Nsight Systems. These are diagnostic profiles, not production TPS.',
           '', '| Backend | Method | GPU kernel sum (ms) | Top kernel (share) | Sync API calls | Launch API calls | Profiled E2E (s) | Peak allocated (MB) |', '|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        top=r['top_kernels'][0] if r['top_kernels'] else {'name':'n/a','share_pct':0}
        lines.append(f"| {r['backend']} | {r['method']} | {r['total_gpu_kernel_time_ns']/1e6:.1f} | `{top['name']}` ({top['share_pct']}%) | {r['synchronization_calls']} | {r['launch_calls']} | {r['reported_measured_seconds_under_profiler']:.3f} | {r['peak_allocated_mb']:.0f} |")
    lines += ['', '## NVTX phase observations','']
    for r in rows:
        phases=r['method_and_phase_ranges']
        d=phases.get('draft_forward',{}).get('total_ns'); v=phases.get('target_verify',{}).get('total_ns')
        if d is not None or v is not None:
            lines.append(f"- `{r['backend']}` / `{r['method']}`: draft forward {d/1e6:.1f} ms; target verify {v/1e6:.1f} ms (summed NVTX durations).")
    lines += ['', '## Limits','']+[f'- {x}' for x in payload['limits']]
    lines += ['', 'Capture hashes and per-cell metadata are in `paired_backend_profiles.json`.']
    (DIR/'paired_backend_profiles.md').write_text('\n'.join(lines)+'\n')
    print(out)

if __name__=='__main__': main()
