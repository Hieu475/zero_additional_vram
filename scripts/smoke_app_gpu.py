"""Manual GPU smoke test for the app path (not run in CPU CI).

Covers chat / summarize / qa / code on Qwen2.5-3B NF4 and checks:
  - no crash on any task;
  - output has no duplicate "User:" turn;
  - method_used is reported and sane;
  - tokens/tps are plausible (tokens>0, tps>0);
  - the routed path really goes through the core HybridDraftRouter
    (acceptance_rate present in info).

Run:  python scripts/smoke_app_gpu.py [--max-new-tokens 48]
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU smoke test for zassd app path")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    from zassd.app.assistant import IntelligentAssistant

    a = IntelligentAssistant(
        model_name=args.model, method="auto",
        max_new_tokens=args.max_new_tokens,
    )
    a.store.add("d1", "ZASSD uses zero extra model-weight VRAM for draft weights.")

    cases = [
        ("chat", "Say hello in one short sentence.", ""),
        ("summarize", "The cat sat on the mat. " * 20, ""),
        ("qa", "What does ZASSD use?", ""),
        ("code", "write a python function adding two numbers", ""),
    ]
    failures: list[str] = []
    for task, user_input, ctx in cases:
        if task == "qa":
            r = a.ask(user_input)
        elif task == "summarize":
            r = a.summarize(user_input)
        elif task == "code":
            r = a.code_assist(user_input)
        else:
            r = a.generate("chat", user_input)
        print(f"[{task}] method={r.method_used} tokens={r.tokens} "
              f"tps={r.tps:.1f} text={r.text[:120]!r}", flush=True)
        if not r.text.strip():
            failures.append(f"{task}: empty output")
        if "User:" in r.text:
            failures.append(f"{task}: duplicate User: turn in output")
        if r.method_used not in ("routed", "vanilla", "prompt_lookup",
                                 "zassd", "hybrid"):
            failures.append(f"{task}: unexpected method {r.method_used}")
        if not (r.tokens > 0 and r.tps > 0):
            failures.append(f"{task}: bad tokens/tps {r.tokens}/{r.tps}")
        if r.method_used == "routed" and not (
                r.info and "acceptance_rate" in r.info):
            failures.append(f"{task}: routed path missing acceptance_rate")

    # Explicit routed call must also carry router metrics.
    r = a.generate("chat", "Hello.", method="routed")
    assert r.info and "acceptance_rate" in r.info, "routed info missing"

    if failures:
        print("SMOKE FAIL:"); [print(" -", f) for f in failures]
        sys.exit(1)
    print("SMOKE PASS: 4 tasks ok, routed path verified")


if __name__ == "__main__":
    main()
