#!/usr/bin/env python3
"""Closed-loop K2/R3 screen with actual fixed-three-row NF4 dispatch."""

from __future__ import annotations

import audit_nf4_singleton_k2_r3_closed_loop as base_runner

FIXED_THREE_ROWS_STRATEGY = "triton_nf4_fixed_three_rows_group_scale_packed"


def main() -> None:
    original_strategy = base_runner.STRATEGY
    base_runner.STRATEGY = FIXED_THREE_ROWS_STRATEGY
    try:
        base_runner.main()
    finally:
        base_runner.STRATEGY = original_strategy


if __name__ == "__main__":
    main()
