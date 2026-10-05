"""IntelligentAssistant: thin orchestration over existing ZASSD engines.

Design rules:
  - No new decoding logic here. All generation goes through the
    existing core: vanilla / prompt_lookup / zassd / routed.
  - `generate_fn` injection makes the whole layer unit-testable on
    CPU without loading a 3B model or touching CUDA.
  - `choose_method("auto")` is a documented heuristic, not a claim
    of optimality: long repetitive context -> prompt_lookup,
    otherwise routed (falls back to vanilla on CPU-only boxes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from zassd.app import prompts as P
from zassd.app.store import SimpleDocStore

AppMethod = Literal["auto", "vanilla", "prompt_lookup", "zassd", "routed", "hybrid"]
AppTask = Literal["chat", "summarize", "qa", "code"]

# (text, info_dict) where info_dict may hold tps/tokens/acceptance.
GenerateFn = Callable[[str, str], tuple[str, dict[str, Any]]]


@dataclass
class AppResult:
    text: str
    method_used: str
    task: str
    latency_s: float = 0.0
    tokens: int = 0
    tps: float = 0.0
    info: dict | None = None


def choose_method(task: str, full_prompt: str, requested: str = "auto") -> str:
    """Dispatcher for the app layer.

    With ``auto`` (the default), always delegate to ``routed`` and let the
    core cost-aware router decide per cycle between AR / PLD / layer-skip
    from measured draft costs. The app layer must not second-guess the
    router with its own char-count heuristic (no router-outside-router);
    token-aware policy lives in one place: the decoding core.
    An explicit ``requested`` value is passed through unchanged.
    """
    if requested != "auto":
        return requested
    return "routed"


class Conversation:
    """Multi-turn chat state. Keeps full message history for context.

    Usage:
        conv = assistant.start_conversation()
        conv.ask("Giới thiệu Đà Lạt")
        conv.ask("Còn món ăn thì sao?")
    """

    def __init__(self, assistant: IntelligentAssistant, max_turns: int = 10) -> None:
        self._assistant = assistant
        self.max_turns = max_turns
        self.messages: list[dict] = [
            {"role": "system", "content": P.SYSTEM_PROMPT}
        ]

    def __len__(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": P.SYSTEM_PROMPT}]

    def _trim(self) -> None:
        # Keep system + last max_turns*2 messages (user+assistant per turn).
        keep = self.max_turns * 2
        if len(self.messages) - 1 > keep:
            self.messages = [self.messages[0]] + self.messages[-keep:]

    def ask(self, user_text: str, method: AppMethod | None = None) -> AppResult:
        a = self._assistant
        self.messages.append({"role": "user", "content": user_text})
        self._trim()
        requested = method or a.method
        t0 = time.perf_counter()
        if a._generate_fn is not None:
            full_prompt = P.format_messages(None, self.messages)
            method_used = choose_method("chat", full_prompt, requested)
            text, info = a._generate_fn(full_prompt, method_used)
        else:
            a._ensure_backend()
            full_prompt = P.format_messages(a._tokenizer, self.messages)
            method_used = choose_method("chat", full_prompt, requested)
            text, info = a._run_real(full_prompt, method_used)
        text = P.truncate_at_stop(text)
        dt = time.perf_counter() - t0
        toks = int(info.get("tokens", 0) or 0)
        tps = float(info.get("tps", 0.0) or 0.0) or (toks / dt if dt > 0 else 0.0)
        self.messages.append({"role": "assistant", "content": text})
        res = AppResult(text=text, method_used=method_used, task="chat",
                        latency_s=dt, tokens=toks, tps=tps, info=info)
        a.history.append(res)
        return res


class IntelligentAssistant:
    """High-level facade used by CLI demo and HTTP server."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        method: AppMethod = "auto",
        device: str = "cuda:0",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        k: int = 4,
        skip_strategy: str = "mid_12",
        quantize: bool = True,
        generate_fn: GenerateFn | None = None,
    ) -> None:
        self.model_name = model_name
        self.method = method
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.k = k
        self.skip_strategy = skip_strategy
        self.quantize = quantize
        self._generate_fn = generate_fn
        # Lazy-loaded real-model handles (only when generate_fn is None).
        self._model: Any = None
        self._tokenizer: Any = None
        self._layer_mgr: Any = None
        self.store = SimpleDocStore()
        self.history: list[AppResult] = []

    # -- prompt building -------------------------------------------------
    def build_prompt(self, task: str, user_input: str, context: str = "") -> str:
        # Uses native ChatML template when tokenizer is loaded,
        # otherwise plain-text fallback (tests / pre-load).
        return P.format_prompt(self._tokenizer, task, user_input, context)

    # -- core dispatch ---------------------------------------------------
    def _ensure_backend(self) -> None:
        if self._generate_fn is not None or self._model is not None:
            return
        # Imported lazily so `import zassd.app` works on CPU-only CI.
        import torch  # noqa: F401
        from zassd.models.layer_manager import LayerManager
        from zassd.models.loader import load_model, load_tokenizer
        from zassd.models.model_adapter import ModelAdapter

        use_quant = self.quantize
        try:
            import torch as _t

            if not _t.cuda.is_available():
                use_quant = False
        except Exception:
            use_quant = False
        self._model = load_model(self.model_name, quantize=use_quant)
        self._tokenizer = load_tokenizer(self.model_name)
        self._layer_mgr = LayerManager(ModelAdapter(self._model))

    def _run_real(self, full_prompt: str, method: str) -> tuple[str, dict]:
        from zassd.baselines.prompt_lookup import prompt_lookup_generate
        from zassd.cli import SKIP_STRATEGIES, run_vanilla
        from zassd.decoding.speculative import self_speculative_generate

        assert self._model is not None and self._tokenizer is not None
        kw = dict(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=self.device,
        )
        if method == "vanilla":
            text, tps, n = run_vanilla(
                self._model, self._tokenizer, full_prompt, **kw
            )
            return text, {"tps": tps, "tokens": n}
        if method == "prompt_lookup":
            text, m = prompt_lookup_generate(
                self._model, self._tokenizer, full_prompt, k=self.k, **kw
            )
            return text, {"tps": m.tokens_per_second, "tokens": m.total_tokens,
                           "acceptance_rate": m.acceptance_rate}
        skip = SKIP_STRATEGIES.get(self.skip_strategy, SKIP_STRATEGIES["mid_12"])
        if method == "routed":
            from zassd.profiling.action_cost_model import MeasuredActionCostModel
            from zassd.routing.hybrid_router import HybridDraftRouter

            cm = MeasuredActionCostModel.for_model(self.model_name)
            router = HybridDraftRouter(cost_model=cm, pld_k=4)
            text, m = self.self_speculative_generate_with(
                draft_mode="routed", router=router, skip=skip,
                prompt=full_prompt, **kw,
            )
        elif method == "hybrid":
            text, m = self.self_speculative_generate_with(
                draft_mode="hybrid", skip=skip, prompt=full_prompt, **kw,
            )
        else:  # zassd layer-skip
            text, m = self.self_speculative_generate_with(
                draft_mode="layer_skip", skip=skip, prompt=full_prompt, **kw,
            )
        return text, {"tps": m.tokens_per_second, "tokens": m.total_tokens,
                       "acceptance_rate": m.acceptance_rate}

    def self_speculative_generate_with(self, **kwargs: Any) -> Any:
        from zassd.decoding.speculative import self_speculative_generate

        return self_speculative_generate(
            self._model, self._tokenizer, self._layer_mgr,
            kwargs.pop("skip"), kwargs.pop("prompt"),
            k=self.k, **kwargs,
        )

    # -- public API ------------------------------------------------------
    def generate(
        self, task: AppTask, user_input: str, context: str = "",
        method: AppMethod | None = None,
    ) -> AppResult:
        requested = method or self.method
        t0 = time.perf_counter()
        if self._generate_fn is not None:
            # Test path: no model, use plain-text prompt.
            full_prompt = self.build_prompt(task, user_input, context)
            method_used = choose_method(task, full_prompt, requested)
            text, info = self._generate_fn(full_prompt, method_used)
        else:
            # Real path: load model first so ChatML template is available,
            # then build the templated prompt for method selection + run.
            self._ensure_backend()
            full_prompt = self.build_prompt(task, user_input, context)
            method_used = choose_method(task, full_prompt, requested)
            text, info = self._run_real(full_prompt, method_used)
        text = P.truncate_at_stop(text)
        dt = time.perf_counter() - t0
        toks = int(info.get("tokens", 0) or 0)
        tps = float(info.get("tps", 0.0) or 0.0) or (toks / dt if dt > 0 else 0.0)
        res = AppResult(text=text, method_used=method_used, task=task,
                        latency_s=dt, tokens=toks, tps=tps, info=info)
        self.history.append(res)
        return res

    def summarize(self, document: str, **kw: Any) -> AppResult:
        return self.generate("summarize", document, **kw)

    def ask(self, question: str, top_k: int = 2, **kw: Any) -> AppResult:
        docs = self.store.query(question, top_k=top_k)
        ctx = self.store.to_context(docs)
        res = self.generate("qa", question, context=ctx, **kw)
        res.info = {**(res.info or {}), "retrieved_ids": [d.doc_id for d in docs]}
        return res

    def code_assist(self, request: str, context: str = "", **kw: Any) -> AppResult:
        return self.generate("code", request, context=context, **kw)

    def start_conversation(self, max_turns: int = 10) -> Conversation:
        """Start a multi-turn conversation keeping full history."""
        return Conversation(self, max_turns=max_turns)

    def run_interactive(self, method: AppMethod | None = None) -> None:
        """Blocking REPL: keeps context across turns. /reset, /quit."""
        conv = self.start_conversation()
        print("[zassd-app] Hoi thoai lien tuc. Go /reset de xoa nho, /quit de thoat.")
        while True:
            try:
                user = input("\nBan: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nTam biet!")
                break
            if not user:
                continue
            if user in ("/quit", "/bye", "/exit"):
                print("Tam biet!")
                break
            if user == "/reset":
                conv.reset()
                print("Da xoa lich su hoi thoai.")
                continue
            res = conv.ask(user, method=method)
            print(f"\nAI [{res.method_used}, {res.tps:.1f} tok/s]: {res.text}")

    def compare(
        self, task: AppTask, user_input: str, context: str = "",
        methods: tuple[str, ...] = ("vanilla", "prompt_lookup", "routed"),
    ) -> list[AppResult]:
        """Run the same prompt through several engines (for thesis table)."""
        return [self.generate(task, user_input, context=context, method=m)  # type: ignore[arg-type]
                for m in methods]
