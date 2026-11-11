# scripts/

Run everything from the repository root. Scripts import a few shared modules by plain name, so put
`scripts/lib` and `src` on the path:

```bash
export PYTHONPATH=src:scripts/lib
python scripts/audit/shape_audit.py summarize
```

The `run_*.sh` runners set this themselves and `cd` to the repository root, so they can be started from anywhere.

| Folder | What is in it |
|---|---|
| `audit/` | Direction C: per-operator shape-sensitivity audits, naive PLD / draft-model divergence studies, row-wise loop prototypes (`a2_`, `a3_`), `shape_audit.py` (single entry point), and the `run_*.sh` runners that produced `experiments/` results. |
| `kernels/` | M-invariant kernel work packages: `wp1`-`wp4` kernel gates, `wp5` exact PLD CUDA-graph loop and block-cost profile, `wp5b` draft-model variant, `wp6` reference agreement and HumanEval, `wp7` bitsandbytes versions, plus their `run_wp*.sh` runners and `run_kscan.sh`. |
| `drafter/` | Learned drafter and hybrid: data generation, training, `drafter_eval_sim.py`, `sim_adaptive_block.py`, `wp9` (drafter loop), `wp10` (PLD + drafter hybrid), `summarize_drafter_runs.py`, and the Qwen2.5-1.5B / repeat runners. |
| `tools/` | `check_environment.py`, `download_model.py`, figure builders, `make_long_context_prompts.py`, `run_with_model.py`. |
| `lib/` | Modules that other scripts import by name: `audit_bnb_nf4_gate_reduction_orders` (MODEL / REVISION constants and reduction-order screen), `a1_rowwise_static_block`, `audit_pld_trajectory_divergence`, `sim_draft`. `_common.py` is not imported by any current script (kept from the old B1 benchmarks). |

## Moved from the flat layout

Provenance strings inside `results/` and `experiments/` JSON files, and `docs/archive/`, still use the old flat
paths (`scripts/<name>`); they were left unchanged on purpose. To find the file, look it up here:
every script that used to be `scripts/<name>` is now `scripts/<folder>/<name>` where the folder is given by the
table above (for example `scripts/wp5_invariant_graph_pld.py` is `scripts/kernels/wp5_invariant_graph_pld.py`,
`scripts/run_c2_sweep.sh` is `scripts/audit/run_c2_sweep.sh`). `git log --follow <new path>` keeps the history.
