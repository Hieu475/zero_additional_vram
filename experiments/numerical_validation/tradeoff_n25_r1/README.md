# Exactness and latency tradeoff (25 mismatch prompts)

This controlled study uses the 25 prompts that diverged under default batched
PLD verification in the RTX 4050 N=50 benchmark. Each method was measured with
three rotated order seeds; intervals bootstrap prompts, not individual repeats.

| Method | Exact sequence match | End-to-end throughput vs vanilla (95% CI) |
|---|---:|---:|
| Vanilla | 100% | Reference |
| Batched PLD | 0% | 1.017x (0.984–1.054) |
| Sequential PLD verification | 100% | 0.968x (0.959–0.976) |
| Canonical KV replay | 68% | 0.166x (0.148–0.186) |
| Margin guard (0.05) | 36% | 0.725x (0.615–0.836) |

Sequential verification is the only tested recovery path that restored exact
sequence identity for all selected prompts. Its measured throughput was about
3.2% lower than vanilla. KV replay and the margin guard were substantially
slower and did not restore exactness on every prompt.

Files:

- `raw_measurements.jsonl`: 375 per-method observations with token traces.
- `correctness_latency_tradeoff.json`: prompt-level summaries and bootstrap CIs.
- `config.yaml`, `environment.json`: run configuration and runtime metadata.

The result applies to this model, NF4 configuration, backend, device, and
mismatch-selected prompt set; it does not estimate exactness over a general
workload distribution.
