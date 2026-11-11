# Removed local profiler captures (11/11/2026)

These were never in git (see the ignore rules in `.gitignore`) and were deleted from the working copy to reduce size.
They came from the legacy B1/B2 Triton-singleton profiling under `experiments/exact_quantized_feasibility/` and were not used by the
M-invariant kernel or learned-drafter work. Compact summary JSON files in the same directories are unchanged.

| Type | Files | Size |
|---|---|---|
| `nsys-rep` | 4 | 0.15 GB |
| `sqlite` | 4 | 0.42 GB |
| `trace.json` | 23 | 5.26 GB |

Total about 5.8 GB. Kept on purpose: `experiments/backend_validation/kernel_profile/*.sqlite` (74 MB), the raw source of the
Nsight timings quoted in `bao_cao/contents/chuong5.tex`. The legacy profiling scripts are no longer in the repository, so these captures
cannot be regenerated from it.
