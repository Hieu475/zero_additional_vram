#!/usr/bin/env python3
"""Shape-invariance audit for quantized LLM inference (direction C): single entry point.

  shape_audit.py ops         [args of audit_op_shape_sensitivity.py]   per-operator bitwise sensitivity to block shape
  shape_audit.py trajectory  [args of audit_pld_trajectory_divergence.py]   naive-PLD trajectory divergence vs M=1 greedy
  shape_audit.py plan        OPS_JSON [OPS_JSON ...]                   operator classes to run row-wise (union over the given audits)
  shape_audit.py summarize   [--dir DIR] [--output FILE.md]            Markdown tables from all result JSONs in DIR

Example:
  PYTHONPATH=scripts:src python scripts/shape_audit.py ops --backend hf_bnb_nf4 --block-size 4 \
      --prompt-ids gsm8k_000 gsm8k_001 --output out.json
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

DEFAULT_DIR = Path("experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles")


LEGACY = {"qwen05b": "Qwen2.5-0.5B-Instruct", "qwen15b": "Qwen2.5-1.5B-Instruct"}


def _label(d: dict) -> str:
    model = d.get("model", "Qwen/Qwen2.5-3B-Instruct").split("/")[-1]
    if "model" not in d:  # legacy JSONs written before the model field existed (C3 runs via run_with_model.py)
        model = next((v for k, v in LEGACY.items() if k in d.get("_file", "")), model)
    backend = d.get("backend", "hf_bnb_nf4")
    note = " [NONDETERMINISTIC run-to-run: not a valid shape-sensitivity measurement]" if backend.startswith("hqq") else ""
    return f"{model} / {backend}{note}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def merged_trajectory(directory: Path) -> str:
    groups: dict[tuple, dict] = {}
    for f in sorted(list(directory.glob("c6*_*.json")) + list(directory.glob("fix_*.json"))):
        d = json.loads(f.read_text())
        if d.get("schema") != "zassd-pld-trajectory-divergence-v1":
            continue
        key = (_label(d), d["k"], "+".join(d.get("rowwise", [])) or "none")
        g = groups.setdefault(key, {"n": 0, "div": 0, "files": []})
        g["n"] += d["prompts"]; g["div"] += d["diverged"]; g["files"].append(f.name)
    lines = ["", "## PLD trajectory divergence vs sequential M=1 greedy, prompt ranges merged (Wilson 95% CI)", "",
             "Row-wise = operator classes run one row at a time during block verification ('none' = naive PLD).", "",
             "| Config | K | row-wise ops | prompts | diverged | rate | 95% CI |", "|---|---:|---|---:|---:|---:|---|"]
    for (label, k, ops), g in sorted(groups.items()):
        lo, hi = wilson(g["div"], g["n"])
        lines.append(f"| {label} | {k} | {ops} | {g['n']} | {g['div']} | {g['div'] / g['n']:.0%} | [{lo:.0%}, {hi:.0%}] |")
    return "\n".join(lines) + "\n"


def summarize(directory: Path) -> str:
    ops = []
    for f in sorted(directory.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("schema") == "zassd-op-shape-sensitivity-v1":
            d["_file"] = f.name
            ops.append((f, d))
    lines = ["## Per-operator bitwise mismatch vs row-by-row (block of M rows)", "",
             "| Config | M | ctx | proj elems mismatched | RMSNorm elems | LM head | SDPA | argmax flips / rows | file |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for f, d in sorted(ops, key=lambda x: (_label(x[1]), x[1].get("context_len", 0), x[1].get("block_size", 3))):
        o, e = d["per_op_class"], d["end_to_end"]
        lin = [v for k, v in o.items() if k.startswith("linear")]
        pt, pe = sum(v["elements"] for v in lin), sum(v["mismatches"] for v in lin)
        rms = sum(v["mismatches"] for k, v in o.items() if k.startswith("rms"))
        lm = o.get("lm_head", {}).get("mismatch_rate", 0.0)
        sd = o.get("sdpa_attention", {}).get("mismatch_rate", 0.0)
        lines.append(f"| {_label(d)} | {d.get('block_size', 3)} | {d.get('context_len', 0) or 'nat.'} | {pe / max(pt, 1):.3%} | {rms} | "
                     f"{lm:.3%} | {sd:.1%} | {e['argmax_flips']}/{e['rows']} | {f.name} |")
    traj = []
    for f in sorted(directory.glob("c6_*.json")):
        d = json.loads(f.read_text())
        if d.get("schema") == "zassd-pld-trajectory-divergence-v1":
            traj.append((f, d))
    lines += ["", "## Naive PLD trajectory divergence vs sequential M=1 greedy", "",
              "| Config | K | prompts | max new | diverged | median first divergence | file |", "|---|---:|---:|---:|---:|---:|---|"]
    for f, d in sorted(traj, key=lambda x: (_label(x[1]), x[1]["k"])):
        lines.append(f"| {_label(d)} | {d['k']} | {d['prompts']} | {d['max_new']} | {d['diverged']}/{d['prompts']} | "
                     f"{d['first_divergence_median']} | {f.name} |")
    return "\n".join(lines) + "\n" + merged_trajectory(directory)


def plan(files: list[Path]) -> str:
    """Operator classes (as accepted by --rowwise) that showed any bitwise mismatch in any given ops audit."""
    need = set()
    for f in files:
        d = json.loads(f.read_text())
        for k, v in d["per_op_class"].items():
            if v["mismatches"] == 0:
                continue
            if k.startswith("linear"): need.add("linear")
            elif k.startswith("rmsnorm"): need.add("rmsnorm")
            elif k == "lm_head": need.add("lm_head")
            elif k == "sdpa_attention": need.add("sdpa")
    return ",".join(sorted(need)) or "none"


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"ops", "trajectory", "summarize", "plan"}:
        print(__doc__); sys.exit(2)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "plan":
        print(plan([Path(x) for x in rest])); return
    here = Path(__file__).parent
    if cmd == "summarize":
        directory, out = DEFAULT_DIR, None
        if "--dir" in rest: directory = Path(rest[rest.index("--dir") + 1])
        if "--output" in rest: out = Path(rest[rest.index("--output") + 1])
        text = summarize(directory)
        out.write_text(text) if out else print(text)
        return
    script = "audit_op_shape_sensitivity.py" if cmd == "ops" else "audit_pld_trajectory_divergence.py"
    sys.argv = [script, *rest]
    runpy.run_path(str(here / script), run_name="__main__")


if __name__ == "__main__":
    main()
