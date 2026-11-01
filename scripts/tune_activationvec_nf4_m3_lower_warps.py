#!/usr/bin/env python3
"""Screen lower-warp and smaller-output-tile launch configurations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import tune_activationvec_nf4_m3_launch as sweep


CONFIGS = ((4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4), (8, 8),
           (16, 1), (16, 2), (16, 4))


def main():
    sweep.CONFIGS = CONFIGS
    sweep.main()
    # The delegated runner owns result construction; attach this driver source
    # and the effective configurations so the expanded screen is self-describing.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    payload = json.loads(args.output.read_text())
    payload["launch_sweep_driver"] = {
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_set": [{"out_tile": n, "num_warps": w} for n, w in CONFIGS],
        "kernel_sweep_source_sha256": hashlib.sha256(Path(sweep.__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
