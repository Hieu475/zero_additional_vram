"""Tests for the thin application layer (CPU-only, no model download)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from zassd.app.assistant import IntelligentAssistant, choose_method
from zassd.app.server import make_handler
from zassd.app.store import SimpleDocStore


def _fake_generate(prompt: str, method: str):
    return f"[{method}] echo:{prompt[:40]}", {"tokens": 8, "tps": 16.0}


class TestAppLayer:
    def test_choose_method_prefers_pld_for_long_context(self):
        long_prompt = "x" * 2000
        assert choose_method("summarize", long_prompt, "auto") == "prompt_lookup"
        assert choose_method("qa", long_prompt, "auto") == "prompt_lookup"
        assert choose_method("chat", "hi", "auto") == "routed"
        assert choose_method("chat", "hi", "vanilla") == "vanilla"

    def test_generate_and_tasks_use_injected_fn(self):
        a = IntelligentAssistant(generate_fn=_fake_generate)
        r = a.generate("chat", "hello")
        assert r.method_used == "routed" and "echo" in r.text

        r2 = a.summarize("x" * 100)
        assert r2.task == "summarize"

        a.store.add("d1", "ZASSD uses zero extra VRAM for draft weights")
        r3 = a.ask("What does ZASSD use?")
        assert r3.task == "qa"
        assert r3.info and r3.info["retrieved_ids"] == ["d1"]

        r4 = a.code_assist("write quicksort")
        assert r4.task == "code"

    def test_compare_runs_all_methods(self):
        a = IntelligentAssistant(generate_fn=_fake_generate)
        out = a.compare("chat", "hello", methods=("vanilla", "prompt_lookup"))
        assert [r.method_used for r in out] == ["vanilla", "prompt_lookup"]

    def test_conversation_keeps_history(self):
        seen: list[str] = []

        def fake(prompt: str, method: str):
            seen.append(prompt)
            return f"reply-{len(seen)}", {"tokens": 4, "tps": 10.0}

        a = IntelligentAssistant(generate_fn=fake)
        conv = a.start_conversation()
        r1 = conv.ask("Giới thiệu Đà Lạt")
        r2 = conv.ask("Còn món ăn thì sao?")
        assert r1.text == "reply-1" and r2.text == "reply-2"
        assert len(conv) == 2
        # Second prompt must contain first turn's context.
        assert "Giới thiệu Đà Lạt" in seen[1] and "reply-1" in seen[1]
        conv.reset()
        assert len(conv) == 0

    def test_doc_store_ranking(self):
        s = SimpleDocStore()
        s.add("a", "cats and dogs play together")
        s.add("b", "quantum computing qubits")
        assert s.query("quantum qubits")[0].doc_id == "b"
        assert len(s) == 2

    def test_http_api_roundtrip(self):
        a = IntelligentAssistant(generate_fn=_fake_generate)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(a))
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            body = json.dumps({"task": "chat", "input": "hi"}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/generate", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode())
            assert "text" in payload and payload["method_used"] == "routed"

            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/health", timeout=5) as resp:
                assert json.loads(resp.read().decode())["status"] == "ok"
        finally:
            srv.shutdown()
