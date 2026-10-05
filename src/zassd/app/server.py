"""Stdlib-only HTTP JSON API for the thesis demo.

No FastAPI/uvicorn dependency on purpose: the 6GB laptop target
and CPU-only CI both run this with the standard library.

Endpoints:
  GET  /api/health
  POST /api/generate   {task, input, context?, method?}
  POST /api/summarize  {document, method?}
  POST /api/qa         {question, top_k?}
  POST /api/code       {request, context?}
  POST /api/chat       {message, session_id?, method?}  (multi-turn)

Run:
  python -m zassd.app.server --port 8000
"""

from __future__ import annotations

import argparse
import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from zassd.app.assistant import Conversation, IntelligentAssistant


def make_handler(assistant: IntelligentAssistant) -> type[BaseHTTPRequestHandler]:
    sessions: dict[str, Conversation] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # quieter logs
            pass

        def _send(self, payload: dict, code: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("", "/"):
                self._send({"service": "zassd-app", "endpoints": [
                    "GET /api/health", "POST /api/generate",
                    "POST /api/summarize", "POST /api/qa", "POST /api/code",
                    "POST /api/chat"]})
            elif self.path.startswith("/api/health"):
                self._send({"status": "ok", "model": assistant.model_name})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self) -> None:
            try:
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            except Exception as e:
                self._send({"error": f"bad json: {e}"}, 400)
                return
            try:
                if self.path == "/api/generate":
                    r = assistant.generate(
                        data.get("task", "chat"), data.get("input", ""),
                        context=data.get("context", ""),
                        method=data.get("method"),
                    )
                elif self.path == "/api/summarize":
                    r = assistant.summarize(data.get("document", ""),
                                            method=data.get("method"))
                elif self.path == "/api/qa":
                    r = assistant.ask(data.get("question", ""),
                                      top_k=int(data.get("top_k", 2)))
                elif self.path == "/api/code":
                    r = assistant.code_assist(data.get("request", ""),
                                              context=data.get("context", ""))
                elif self.path == "/api/chat":
                    sid = data.get("session_id") or uuid.uuid4().hex[:8]
                    conv = sessions.get(sid)
                    if conv is None:
                        conv = assistant.start_conversation()
                        sessions[sid] = conv
                    if data.get("reset"):
                        conv.reset()
                    r = conv.ask(data.get("message", ""),
                                 method=data.get("method"))
                    self._send({"text": r.text, "method_used": r.method_used,
                                "task": r.task, "session_id": sid,
                                "turns": len(conv),
                                "latency_s": round(r.latency_s, 3),
                                "tokens": r.tokens, "tps": round(r.tps, 2)})
                    return
                else:
                    self._send({"error": "not found"}, 404)
                    return
                self._send({"text": r.text, "method_used": r.method_used,
                            "task": r.task, "latency_s": round(r.latency_s, 3),
                            "tokens": r.tokens, "tps": round(r.tps, 2),
                            "info": r.info or {}})
            except Exception as e:
                self._send({"error": str(e)}, 500)

    return Handler


def serve(port: int = 8000, **assistant_kw: object) -> None:
    assistant = IntelligentAssistant(**assistant_kw)  # type: ignore[arg-type]
    # Seed a couple of demo docs so /api/qa works out of the box.
    assistant.store.add("zassd", "ZASSD costs 0 MB of extra model-weight VRAM. "
                                 "Prompt Lookup reaches 1.39x on repetitive context.")
    assistant.store.add("pld", "Prompt Lookup Decoding extracts n-gram candidates "
                               "at ~0ms draft cost and verifies them in parallel.")
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(assistant))
    print(f"[zassd-app] serving on http://127.0.0.1:{port} (Ctrl+C to stop)")
    srv.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="ZASSD app HTTP server (stdlib only)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--method", type=str, default="auto")
    args = ap.parse_args()
    serve(port=args.port, model_name=args.model, method=args.method)


if __name__ == "__main__":
    main()
