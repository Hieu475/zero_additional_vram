# Unified paired router evaluation

Prompts: 30; best fixed selected on development: `pld_k2`.
Development fixed-method mean decode TPS: `{'pld_k2': 41.31562228417908, 'pld_k4': 39.67193443395708, 'pld_k8': 40.855523491213916, 'cka90_k2': 34.75011696139854}`.
Direct vanilla matched WP3 canonical token IDs on 30/30 prompts.

All speedups use per-prompt decode TPS. CIs bootstrap prompts; exactness compares raw output token IDs with direct vanilla.

## all

| Method | N | Decode TPS | Speedup vs vanilla (95% CI) | TPS ratio vs PLD K4 (95% CI) | Exact match | Slowdown | Regret vs offline fixed (TPS) | VRAM MB | Acceptance |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| direct_vanilla | 30 | 38.29 | 1.000 [1.000, 1.000] | 0.956 [0.919, 0.991] | 30.0/30 | 0.0/30 | 5.44 | 2107 | — |
| vanilla_speculative_core_k0 | 30 | 37.01 | 0.968 [0.952, 0.984] | 0.924 [0.888, 0.961] | 30.0/30 | 21.0/30 | 6.57 | 2109 | 0.000 |
| pld_k2 | 30 | 41.89 | 1.095 [1.054, 1.139] | 1.037 [1.013, 1.061] | 18.0/30 | 7.0/30 | 1.58 | 2110 | 0.393 |
| pld_k4 | 30 | 40.50 | 1.059 [1.018, 1.103] | 1.000 [1.000, 1.000] | 14.0/30 | 11.0/30 | 2.97 | 2111 | 0.269 |
| pld_k8 | 30 | 42.04 | 1.099 [1.040, 1.163] | 1.034 [1.011, 1.057] | 14.0/30 | 9.0/30 | 1.43 | 2113 | 0.213 |
| cka90_k2 | 30 | 34.50 | 0.902 [0.875, 0.929] | 0.861 [0.822, 0.899] | 16.0/30 | 27.0/30 | 8.96 | 2112 | 0.872 |
| hybrid_router | 30 | 39.95 | 1.045 [1.001, 1.090] | 0.987 [0.970, 1.003] | 16.0/30 | 13.0/30 | 3.58 | 2113 | 0.261 |
| best_fixed_selected_on_development | 30 | 41.89 | 1.095 [1.055, 1.139] | 1.037 [1.013, 1.061] | 18.0/30 | 7.0/30 | 1.58 | 2110 | 0.393 |
| offline_best_fixed_per_prompt | 30 | 43.47 | 1.136 [1.084, 1.194] | 1.072 [1.052, 1.092] | 15.0/30 | 4.0/30 | 0.00 | 2111 | 0.323 |

Router action distribution: `{'vanilla': 653, 'pld': 153}`.

## heldout

| Method | N | Decode TPS | Speedup vs vanilla (95% CI) | TPS ratio vs PLD K4 (95% CI) | Exact match | Slowdown | Regret vs offline fixed (TPS) | VRAM MB | Acceptance |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| direct_vanilla | 14 | 38.54 | 1.000 [1.000, 1.000] | 0.931 [0.886, 0.973] | 14.0/14 | 0.0/14 | 7.30 | 2107 | — |
| vanilla_speculative_core_k0 | 14 | 36.68 | 0.953 [0.925, 0.979] | 0.886 [0.848, 0.927] | 14.0/14 | 10.0/14 | 9.17 | 2109 | 0.000 |
| pld_k2 | 14 | 43.63 | 1.134 [1.081, 1.194] | 1.049 [1.012, 1.085] | 11.0/14 | 1.0/14 | 2.21 | 2110 | 0.479 |
| pld_k4 | 14 | 41.69 | 1.083 [1.031, 1.142] | 1.000 [1.000, 1.000] | 14.0/14 | 3.0/14 | 4.16 | 2111 | 0.346 |
| pld_k8 | 14 | 44.98 | 1.169 [1.093, 1.253] | 1.076 [1.051, 1.102] | 13.0/14 | 1.0/14 | 0.86 | 2113 | 0.275 |
| cka90_k2 | 14 | 35.28 | 0.917 [0.880, 0.956] | 0.852 [0.803, 0.902] | 12.0/14 | 12.0/14 | 10.56 | 2112 | 0.887 |
| hybrid_router | 14 | 41.48 | 1.078 [1.022, 1.142] | 0.995 [0.975, 1.017] | 14.0/14 | 3.0/14 | 4.37 | 2113 | 0.319 |
| best_fixed_selected_on_development | 14 | 43.63 | 1.134 [1.080, 1.192] | 1.049 [1.010, 1.084] | 11.0/14 | 1.0/14 | 2.21 | 2110 | 0.479 |
| offline_best_fixed_per_prompt | 14 | 45.84 | 1.191 [1.123, 1.271] | 1.098 [1.077, 1.120] | 12.0/14 | 0.0/14 | 0.00 | 2112 | 0.377 |

Router action distribution: `{'vanilla': 285, 'pld': 76}`.

## diagnostic

| Method | N | Decode TPS | Speedup vs vanilla (95% CI) | TPS ratio vs PLD K4 (95% CI) | Exact match | Slowdown | Regret vs offline fixed (TPS) | VRAM MB | Acceptance |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| direct_vanilla | 16 | 38.07 | 1.000 [1.000, 1.000] | 0.977 [0.922, 1.029] | 16.0/16 | 0.0/16 | 3.80 | 2107 | — |
| vanilla_speculative_core_k0 | 16 | 37.30 | 0.981 [0.964, 0.998] | 0.958 [0.903, 1.007] | 16.0/16 | 11.0/16 | 4.29 | 2109 | 0.000 |
| pld_k2 | 16 | 40.36 | 1.061 [1.008, 1.124] | 1.026 [0.997, 1.058] | 7.0/16 | 6.0/16 | 1.02 | 2110 | 0.318 |
| pld_k4 | 16 | 39.46 | 1.037 [0.980, 1.103] | 1.000 [1.000, 1.000] | 0.0/16 | 8.0/16 | 1.93 | 2111 | 0.202 |
| pld_k8 | 16 | 39.46 | 1.037 [0.965, 1.127] | 0.997 [0.972, 1.026] | 1.0/16 | 8.0/16 | 1.93 | 2112 | 0.158 |
| cka90_k2 | 16 | 33.83 | 0.889 [0.852, 0.924] | 0.869 [0.809, 0.924] | 4.0/16 | 15.0/16 | 7.56 | 2112 | 0.860 |
| hybrid_router | 16 | 38.61 | 1.015 [0.956, 1.077] | 0.979 [0.955, 1.003] | 2.0/16 | 10.0/16 | 2.90 | 2113 | 0.210 |
| best_fixed_selected_on_development | 16 | 40.36 | 1.061 [1.008, 1.123] | 1.026 [0.997, 1.059] | 7.0/16 | 6.0/16 | 1.02 | 2110 | 0.318 |
| offline_best_fixed_per_prompt | 16 | 41.39 | 1.088 [1.022, 1.169] | 1.049 [1.024, 1.078] | 3.0/16 | 4.0/16 | 0.00 | 2111 | 0.276 |

Router action distribution: `{'vanilla': 368, 'pld': 77}`.

## diagnostic_evaluation

| Method | N | Decode TPS | Speedup vs vanilla (95% CI) | TPS ratio vs PLD K4 (95% CI) | Exact match | Slowdown | Regret vs offline fixed (TPS) | VRAM MB | Acceptance |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| direct_vanilla | 8 | 37.55 | 1.000 [1.000, 1.000] | 0.967 [0.899, 1.034] | 8.0/8 | 0.0/8 | 3.44 | 2107 | — |
| vanilla_speculative_core_k0 | 8 | 37.40 | 0.997 [0.970, 1.023] | 0.964 [0.888, 1.025] | 8.0/8 | 3.0/8 | 3.27 | 2109 | 0.000 |
| pld_k2 | 8 | 39.41 | 1.051 [0.986, 1.113] | 1.009 [0.969, 1.065] | 4.0/8 | 3.0/8 | 0.95 | 2110 | 0.289 |
| pld_k4 | 8 | 39.24 | 1.046 [0.971, 1.128] | 1.000 [1.000, 1.000] | 0.0/8 | 3.0/8 | 1.12 | 2111 | 0.208 |
| pld_k8 | 8 | 38.06 | 1.015 [0.942, 1.085] | 0.972 [0.944, 0.998] | 1.0/8 | 3.0/8 | 2.30 | 2112 | 0.146 |
| cka90_k2 | 8 | 32.90 | 0.878 [0.812, 0.939] | 0.848 [0.768, 0.926] | 1.0/8 | 7.0/8 | 7.46 | 2112 | 0.827 |
| hybrid_router | 8 | 38.09 | 1.014 [0.944, 1.106] | 0.971 [0.937, 1.006] | 1.0/8 | 5.0/8 | 2.51 | 2112 | 0.203 |
| best_fixed_selected_on_development | 8 | 39.41 | 1.051 [0.984, 1.115] | 1.009 [0.968, 1.065] | 4.0/8 | 3.0/8 | 0.95 | 2110 | 0.289 |
| offline_best_fixed_per_prompt | 8 | 40.36 | 1.076 [1.000, 1.151] | 1.031 [1.003, 1.077] | 1.0/8 | 2.0/8 | 0.00 | 2111 | 0.235 |

Router action distribution: `{'vanilla': 194, 'pld': 33}`.

## Limits

The 8 development prompts are selected from the known mismatch diagnostic cohort, so best-fixed selection may not generalize. Held-out results use the 14 unselected WP3 prompts. Offline best-per-prompt is an upper-bound diagnostic using future per-prompt timing and is not deployable. Each method is run once per prompt; prompt bootstrap intervals do not represent between-session or thermal variation. No cycle-level oracle is constructed.
