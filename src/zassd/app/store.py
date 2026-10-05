"""Minimal in-memory document store for RAG demo.

No external dependency (no vector DB, no embeddings).
Retrieval = case-insensitive word-overlap score. This is
intentional: the thesis point is the *decoding system* (ZASSD),
not the retriever. A transparent keyword retriever keeps the
demo reproducible on CPU and makes PLD behaviour explainable
(retrieved passages are pasted verbatim into the prompt).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Doc:
    doc_id: str
    text: str


class SimpleDocStore:
    """Tiny keyword document store."""

    def __init__(self) -> None:
        self._docs: list[Doc] = []

    def add(self, doc_id: str, text: str) -> None:
        self._docs = [d for d in self._docs if d.doc_id != doc_id]
        self._docs.append(Doc(doc_id=doc_id, text=text))

    def __len__(self) -> int:
        return len(self._docs)

    def query(self, question: str, top_k: int = 2) -> list[Doc]:
        if not self._docs:
            return []
        q = set(_tokens(question))
        scored: list[tuple[int, Doc]] = []
        for d in self._docs:
            overlap = len(q & set(_tokens(d.text)))
            scored.append((overlap, d))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [d for s, d in scored[:top_k] if s > 0] or [d for _, d in scored[:1]]

    def to_context(self, docs: list[Doc]) -> str:
        return "\n\n".join(f"[{d.doc_id}] {d.text}" for d in docs)
