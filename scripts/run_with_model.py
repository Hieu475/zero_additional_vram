#!/usr/bin/env python3
"""Run an existing script on another Qwen2 checkpoint (experiment-only, in-process allowlist extension).

Usage: run_with_model.py MODEL REVISION script.py [args...]
The fail-closed default in src/zassd/certification/model_capabilities.py is not modified.
"""

import dataclasses
import runpy
import sys

import audit_bnb_nf4_gate_reduction_orders as base
from zassd.certification import model_capabilities as caps

model, revision, script, *rest = sys.argv[1:]
base.MODEL, base.REVISION = model, revision
spec = caps.FIXED_ROW_ADAPTERS["qwen2"]
caps.FIXED_ROW_ADAPTERS["qwen2"] = dataclasses.replace(
    spec, allowed_model_ids=spec.allowed_model_ids + (model,),
    allowed_model_revisions=spec.allowed_model_revisions + (revision,))
sys.argv = [script, *rest]
runpy.run_path(script, run_name="__main__")
