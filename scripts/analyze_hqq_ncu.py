#!/usr/bin/env python3
"""Extract targeted HQQ/GemLite Nsight Compute samples."""
from __future__ import annotations
import csv, hashlib, io, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'experiments/backend_validation/kernel_profile'
CAPTURES={'gemv_INT_revsplitK_kernel':'hqq_vanilla_gemv.ncu-rep',
          'gemm_INT_kernel':'hqq_vanilla_gemm.ncu-rep'}
METRICS={'gpu__time_duration.avg':'duration_us',
         'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed':'dram_throughput_pct_peak',
         'sm__throughput.avg.pct_of_peak_sustained_elapsed':'sm_throughput_pct_peak',
         'launch__registers_per_thread':'registers_per_thread'}

def main():
    results={}
    for family,path in CAPTURES.items():
        raw=subprocess.run(['ncu','--import',str(DIR/path),'--page','raw','--csv'],
                           capture_output=True,text=True,check=True)
        rows=[r for r in csv.DictReader(io.StringIO(raw.stdout)) if r.get('Kernel Name')]
        if len(rows)!=3 or any(family not in r['Kernel Name'] for r in rows):
            raise ValueError(f'expected three {family} samples; got {[r.get("Kernel Name") for r in rows]}')
        results[family]=[{'kernel_name':r['Kernel Name'],'grid_size':r.get('Grid Size'),
                          'block_size':r.get('Block Size'),
                          **{alias:float(r[key]) for key,alias in METRICS.items()}}
                         for r in rows]
    out=DIR/'hqq_ncu_results.json'
    out.write_text(json.dumps(results,indent=2)+'\n')
    lines=['# HQQ + GemLite targeted Nsight Compute samples','',
           'Qwen2.5-3B-Instruct, RTX 4050 Laptop GPU (SM 8.9), 384-token prompt, 32-token output budget, SDPA/static KV=436. Three first matching launches per kernel family, captured after a 32-token warmup using Nsight Compute basic set. NCU replays kernels and perturbs timing; samples are not production latency.', '',
           '| Kernel | Launch | Grid | Block | Duration (µs) | DRAM throughput (% peak) | SM throughput (% peak) | Registers/thread |','|---|---:|---|---|---:|---:|---:|---:|']
    for family,samples in results.items():
        for i,r in enumerate(samples,1):
            lines.append(f"| `{family}` | {i} | {r['grid_size']} | {r['block_size']} | {r['duration_us']:.2f} | {r['dram_throughput_pct_peak']:.1f} | {r['sm_throughput_pct_peak']:.1f} | {r['registers_per_thread']:.0f} |")
    lines += ['', 'The sampled HQQ GEMV launches range from about 4.5 to 16.5 µs; the three GEMM launches range from about 52.5 to 259.0 µs. The launch-to-launch spread indicates that shape/work phase matters and the first three launches should not be generalized to the whole model. These basic counters show moderate DRAM utilization for sampled GEMV and low DRAM utilization for GEMM, but do not provide matched FLOP/byte, arithmetic intensity, or a roofline classification.', '', 'Raw reports: `hqq_vanilla_gemv.ncu-rep` and `hqq_vanilla_gemm.ncu-rep`.']
    (DIR/'hqq_ncu_report.md').write_text('\n'.join(lines)+'\n')
    tracked=[DIR/p for p in CAPTURES.values()]
    manifest={'scope':'three targeted invocations per HQQ/GemLite kernel family; NCU replay diagnostic only',
              'hardware':'NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9',
              'ncu_version':subprocess.run(['ncu','--version'],capture_output=True,text=True,check=True).stdout.strip(),
              'collection':{'set':'basic','profile_from_start':'off','launch_count':3,
                            'warmup_tokens':32,'capture_range':'torch.cuda.profiler.start/stop',
                            'workload':'qasper_119_ctx384, 384 input, output budget 32, HQQ W4A16 GemLite, FP16, SDPA, static KV=436'},
              'captures_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
              'summary_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    (DIR/'hqq_ncu_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)
if __name__=='__main__': main()
