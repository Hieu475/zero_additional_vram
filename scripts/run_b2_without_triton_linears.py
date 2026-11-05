#!/usr/bin/env python3
"""Run benchmark_b2_lazy_lm_head.py with the Triton M=3 projection override replaced by a no-op (standard bitsandbytes
projections inside the same fixed-row/row-wise schedule). Usage: run_b2_without_triton_linears.py [benchmark args...]"""

import runpy
import sys
from contextlib import contextmanager

import benchmark_bnb_nf4_activationvec_fallback_closed_loop as fallback_mod


@contextmanager
def no_triton(model, config_selector=None):
    yield


fallback_mod.triton_m3_overrides = no_triton
sys.argv = ["scripts/benchmark_b2_lazy_lm_head.py", *sys.argv[1:]]
runpy.run_path("scripts/benchmark_b2_lazy_lm_head.py", run_name="__main__")
