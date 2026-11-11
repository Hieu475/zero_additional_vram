#!/usr/bin/env bash
# Quick checks: unit tests, then regenerate the audit summary table.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PWD/scripts/lib"
python3 -m pytest tests -q
python3 scripts/audit/shape_audit.py summarize
