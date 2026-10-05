"""Prompt templates for the thin application layer.

Keeps thesis demo prompts in one place so they can be cited
in the report (prompt design is part of the application work).
"""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a helpful AI assistant running on a memory-constrained laptop GPU. "
    "Answer concisely in the same language as the user. "
    "Give ONE single answer only. Do NOT invent extra turns, "
    "do NOT write 'User:' again."
)

# Generation must stop when the model tries to start a new turn.
STOP_SEQUENCES = ["\nUser:", "\n\nUser:", "User:", "<|im_end|>", "<|eot_id|>", "</s>"]


def build_messages(task: str, user_input: str, context: str = "") -> list[dict]:
    """ChatML messages; used when tokenizer has a chat template (Qwen/Llama)."""
    if task == "summarize":
        user = (
            "Summarize the document below in at most 5 sentences, "
            "preserving key facts.\n\n"
            f"Document:\n{user_input}"
        )
    elif task == "qa":
        user = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {user_input}"
        )
    elif task == "code":
        ctx = f"\nExisting code/context:\n{context}\n" if context else ""
        user = (
            "Respond with a short explanation followed by clean Python code "
            f"with comments.{ctx}\nRequest: {user_input}"
        )
    else:
        user = f"{context}\n\n{user_input}" if context else user_input
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def format_messages(tokenizer: object | None, messages: list[dict]) -> str:
    """Render an arbitrary multi-turn message list via ChatML template."""
    try:
        if (
            tokenizer is not None
            and hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None)
        ):
            return tokenizer.apply_chat_template(  # type: ignore[union-attr]
                messages, tokenize=False, add_generation_prompt=True  # type: ignore[arg-type]
            )
    except Exception:
        pass
    # Plain-text fallback: concatenate turns.
    parts = []
    for m in messages:
        role = m.get("role", "user")
        parts.append(f"{role.capitalize()}: {m.get('content', '')}")
    return "\n\n".join(parts) + "\nAssistant:"


def format_prompt(tokenizer: object | None, task: str, user_input: str, context: str = "") -> str:
    """Prefer the model's native chat template; fallback to plain text."""
    messages = build_messages(task, user_input, context)
    try:
        if (
            tokenizer is not None
            and hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None)
        ):
            return tokenizer.apply_chat_template(  # type: ignore[union-attr]
                messages, tokenize=False, add_generation_prompt=True  # type: ignore[arg-type]
            )
    except Exception:
        pass
    # Plain-text fallback (old behaviour, now with explicit no-extra-turn guard).
    if task == "summarize":
        return build_summarize_prompt(user_input)
    if task == "qa":
        return build_qa_prompt(user_input, context)
    if task == "code":
        return build_code_prompt(user_input, context)
    return build_chat_prompt(f"{context}\n\n{user_input}" if context else user_input)


def truncate_at_stop(text: str, stops: list[str] | None = None) -> str:
    """Cut generation at the first stop sequence (anti 'User:' loop)."""
    for s in stops or STOP_SEQUENCES:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            return text[:idx].rstrip()
    return text


def build_chat_prompt(user_input: str) -> str:
    return (
        "You are a helpful AI assistant running on a memory-constrained "
        "laptop GPU. Answer concisely.\n\n"
        f"User: {user_input}\nAssistant:"
    )


def build_summarize_prompt(document: str, max_sentences: int = 5) -> str:
    return (
        "You are a summarization assistant. The document below may contain "
        "repeated structure (headings, lists, code). Summarize it in at most "
        f"{max_sentences} sentences, preserving key facts.\n\n"
        f"Document:\n{document}\n\nSummary:"
    )


def build_qa_prompt(question: str, context: str) -> str:
    # Context-first layout: intentional. Retrieved passages are repeated
    # verbatim in the prompt, which is exactly the recurrence pattern
    # where Prompt Lookup Decoding (PLD) wins (1.39x in README).
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def build_code_prompt(request: str, context: str = "") -> str:
    ctx = f"\nExisting code/context:\n{context}\n" if context else ""
    return (
        "You are a coding assistant. Respond with a short explanation "
        "followed by clean Python code with comments."
        f"{ctx}\nRequest: {request}\nResponse:"
    )


TASK_DESCRIPTIONS = {
    "chat": "General chat (routed decoding: core router picks AR/PLD/layer-skip).",
    "summarize": "Document summarization (routed decoding).",
    "qa": "Retrieval-augmented QA over user documents (routed decoding).",
    "code": "Code explanation / generation (routed decoding).",
}
