"""N-gram matching utilities for Prompt Lookup Decoding (PLD)."""

from __future__ import annotations


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
    if len(tokens) <= ngram_size:
        return []

    # Try matching with ngram_size
    query = tokens[-ngram_size:]
    search_end = len(tokens) - ngram_size
    for i in range(search_end - ngram_size, -1, -1):
        if tokens[i : i + ngram_size] == query:
            match_start = i + ngram_size
            candidates = tokens[match_start : match_start + max_candidates]
            if candidates:
                return candidates

    # Fallback: try 2-gram matching
    if ngram_size > 2:
        query_2 = tokens[-2:]
        search_end_2 = len(tokens) - 2
        for i in range(search_end_2 - 2, -1, -1):
            if tokens[i : i + 2] == query_2:
                match_start = i + 2
                candidates = tokens[match_start : match_start + max_candidates]
                if candidates:
                    return candidates

    return []
