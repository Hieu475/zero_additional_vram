# RTX 4050 thermal robustness pilot

## Protocol

Two paired sessions used the same 12 prepared 384-token LongBench prompts and
the same NF4/FP16, SDPA, static-KV runtime. Each prompt ran direct vanilla,
forced-miss PLD K=4, prompt+generated PLD K=4, and PLD K=8 in randomized method
order. Telemetry was sampled once per second. No GPU power or clock settings
were changed.

- `rtx4050_session1/` began at 48°C and ran through a natural warm-up.
- `rtx4050_session2_hot/` preheated with vanilla generation to 78°C (139.6 s),
  then ran the paired measurements. The recorded telemetry reached 84°C.
- Both runs used the same 12 prompt IDs; the order seed changed between runs.
- `thermal_paired_analysis.json` contains per-session prompt-bootstrap intervals
  and the paired hot-minus-cool speedup differences.

## Results

| Method | Cool-start mean speedup (95% prompt CI) | Heat-soaked mean speedup (95% prompt CI) | Hot minus cool (95% paired prompt CI) | Exact sequences, cool / hot |
|---|---:|---:|---:|---:|
| Forced-miss PLD K=4 | 0.968 (0.957–0.978) | 0.970 (0.937–1.004) | +0.002 (-0.031–0.040) | 12/12, 12/12 |
| PLD K=4 | 1.055 (0.961–1.164) | 1.038 (0.940–1.145) | -0.016 (-0.056–0.022) | 7/12, 7/12 |
| PLD K=8 | 1.089 (0.954–1.250) | 1.089 (0.962–1.242) | +0.000 (-0.033–0.038) | 7/12, 7/12 |

The cool-start telemetry ranged 51–73°C with mean SM clock 2549 MHz. The
heat-soaked session ranged 52–84°C across preheat plus measurement, with mean
SM clock 2530 MHz. Its PLD measurement rows were mostly around 77–79°C; only a
small tail reached 80°C or above. The apparent session-to-session speedup
changes are small relative to prompt variation; these data do not show a clear
thermal effect on relative speedup. The pilot is also too small to establish
thermal robustness.

PLD K=4/K=8 had only 7/12 sequence matches in both sessions. Their throughput
numbers therefore do not support a lossless-acceleration claim. Forced-miss
PLD matched vanilla on all prompts and was about 3% slower, showing framework
overhead in this run.

## Limits and next step

There is only one session per condition, on the same day and GPU. The reported
intervals resample prompts, not independent sessions; temperature is partly
confounded with session order and runtime. The experiment is diagnostic only.
For a defensible thermal claim, repeat at least three sessions per condition
on separate occasions, preserve the same prompt set, randomize method order,
and report a session-level interval in addition to prompt-level paired results.

Reproduce with `scripts/run_thermal_pld_pilot.py` and analyze with
`scripts/analyze_thermal_pld_pilot.py`. Raw generation records, one-second
telemetry, run configuration, environment, and preheat details are stored in
each session directory.
