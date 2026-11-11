"""N-gram matching utilities for Prompt Lookup Decoding (PLD)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NgramMatch:
    candidates: list[int]
    source: str
    ngram_size: int
    repetitions: int
    confidence: float


def find_candidate_match(
    history_tokens: list[int], prompt_tokens: list[int], generated_tokens: list[int], *,
    ngram_size: int = 3, max_candidates: int = 4,
    include_generated: bool = False,
) -> NgramMatch:
    """Find a continuation and report its source and a simple confidence.

    Prompt and generated history are searched separately, so experiments can
    attribute gains and lookup cost to each source. Confidence is a feature
    (not a calibrated probability): longer queries and repeated continuations
    receive higher values.
    """
    if max_candidates <= 0:
        return NgramMatch([], "miss", 0, 0, 0.0)
    sources = [("prompt", prompt_tokens)]
    if include_generated:
        sources.append(("generated", generated_tokens))
    # Prefer prompt matches to preserve classical PLD's input-grounded behavior;
    # use generated text when the prompt contains no usable continuation.
    for source, tokens in sources:
        for size in range(min(ngram_size, len(history_tokens)), 1, -1):
            query = history_tokens[-size:]
            matches: list[list[int]] = []
            for i in range(0, len(tokens) - size):
                if tokens[i:i + size] == query:
                    tail = tokens[i + size:i + size + max_candidates]
                    if tail:
                        matches.append(tail)
            if not matches:
                continue
            # The nearest prior occurrence is the legacy PLD choice.
            candidates = matches[-1]
            comparable = [m[:min(map(len, matches))] for m in matches]
            agreed = 0
            for col in zip(*comparable):
                if len(set(col)) == 1:
                    agreed += 1
                else:
                    break
            confidence = min(1.0, 0.25 * (size - 1) + 0.15 * (len(matches) - 1)
                             + 0.10 * agreed)
            return NgramMatch(candidates, source, size, len(matches), confidence)
    return NgramMatch([], "miss", 0, 0, 0.0)


def find_candidate_tokens(
    tokens: list[int],
    ngram_size: int = 3,
    max_candidates: int = 4,
) -> list[int]:
    """Find speculative candidate tokens via exact n-gram matching in generation history.

    Args:
        tokens: Full sequence of token IDs (prompt + generated tokens).
        ngram_size: Initial n-gram length to search for.
        max_candidates: Maximum number of candidate tokens to return.

    Returns:
        List of candidate token IDs following the most recent n-gram match, or empty list.
    """
    return find_candidate_match(tokens, tokens, [], ngram_size=ngram_size,
                               max_candidates=max_candidates,
                               include_generated=False).candidates
