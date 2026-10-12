# Nsight Systems layer-skip profile

Raw `.nsys-rep` captures and JSON metrics are stored here. SQLite files generated
by `nsys stats` are local derived files and are ignored. Reproduce the profiled
configuration with:

```bash
mkdir -p experiments/layer_skip_profile
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o experiments/layer_skip_profile/cka90_k2 \
  python scripts/profile_layer_skip_nsight.py --mode layer_skip \
  --config cka_90 --k 2 --max-new-tokens 24 \
  --output-json experiments/layer_skip_profile/cka90_k2.json
nsys stats --report nvtx_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  experiments/layer_skip_profile/cka90_k2.nsys-rep
```

Profile metadata: RTX 4050 Laptop GPU, Qwen2.5-3B-Instruct at pinned revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, NF4/FP16 compute with SDPA, prompt 1
(22 input tokens), 24 generated tokens. The trace was captured from commit
`be533cfbd8ac5e2344c19d329629cf199a731801`; warmup ran before the CUDA profiler
range. The vanilla comparison uses the same prompt, model and output budget.

## What this profile shows

The CKA-90/K2 generation completed eight cycles, drafted 16 tokens, and accepted
all 16. Its instrumented generation metrics were 511.3 ms draft, 312.6 ms target
verification, 2.27 ms cache commit/crop, 30.6 ms other work, and 39.1 ms prefill.
The Nsight NVTX summary reports eight draft ranges averaging 63.9 ms, eight verify
ranges averaging 39.1 ms, eight cache commit/crop ranges averaging 0.25 ms, and one
39.0 ms prefill range. Thus the draft path remains the largest cost even with 32 of
36 layers retained; batched verification is cheaper per cycle than drafting, and KV
commit/crop is negligible in this trace.

CUDA API summary:

| Signal | CKA-90/K2 | Vanilla |
|---|---:|---:|
| `cudaLaunchKernel` calls | 47,969 | 47,183 |
| `cudaStreamSynchronize` calls | 54 | 25 |
| `cudaDeviceSynchronize` calls | 28 | 2 |
| Device-to-host copies | 52 | 24 |

The copied values are tiny scalars (Nsight reports about 47 microseconds total GPU
memcpy time for CKA-90/K2). The larger concern is synchronization frequency and its
host wait time, not transfer bandwidth. Draft generation converts each generated
token to a Python integer for EOS handling, and acceptance/entropy decisions also
read scalar tensors; these are plausible sources of the extra synchronization.
This trace identifies the pattern but cannot attribute every call to a source line:
the environment disabled Nsight CPU IP/backtrace sampling and context-switch
tracing. The `.item()` attribution remains a code-based hypothesis to test with a
reduced-sync implementation.

These timings are profiler diagnostics from one short prompt, not performance
benchmark estimates. NVTX instrumentation and Nsight alter execution; the runs do
not establish a speedup. Next profiling step is a controlled reduced-sync ablation
with matching output and EOS semantics, followed by repeat measurements without
Nsight.
