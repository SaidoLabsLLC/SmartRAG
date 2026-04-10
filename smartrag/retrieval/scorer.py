"""Scoring and ranking logic for SmartRAG retrieval."""

from __future__ import annotations

import re

STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "same", "so", "than", "too", "very", "just",
    "about", "above", "after", "again", "against", "before", "below",
    "between", "during", "into", "through", "under", "until", "up", "down",
    "out", "off", "over", "then", "once", "here", "there",
})


def tokenize_query(query: str) -> list[str]:
    """Tokenize a query string into keywords, removing stop words."""
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def keyword_overlap_score(keywords: list[str], target: list[str]) -> float:
    """Compute Jaccard-like similarity with boosting between keyword sets."""
    if not keywords or not target:
        return 0.0
    kw_set = set(keywords)
    target_set = set(t.lower() for t in target)
    intersection = kw_set & target_set
    union = kw_set | target_set
    if not union:
        return 0.0
    return len(intersection) / len(union)


def partial_match_score(keywords: list[str], target: list[str]) -> float:
    """Score based on partial/substring matching between keywords and targets."""
    if not keywords or not target:
        return 0.0
    target_lower = [t.lower() for t in target]
    matches = 0
    for kw in keywords:
        for t in target_lower:
            if kw in t or t in kw:
                matches += 1
                break
    return matches / max(len(keywords), 1)


def score_index_entry(
    keywords: list[str],
    fingerprint: list[str],
    categories: list[str],
    synopsis: str,
    title: str,
) -> float:
    """Score a master index entry against query keywords."""
    score = 0.0

    # Fingerprint match (3x weight)
    fp_score = keyword_overlap_score(keywords, fingerprint)
    fp_partial = partial_match_score(keywords, fingerprint)
    score += max(fp_score, fp_partial * 0.7) * 3.0

    # Title match (3x weight)
    title_tokens = re.findall(r"[a-z0-9]+", title.lower())
    title_score = keyword_overlap_score(keywords, title_tokens)
    title_partial = partial_match_score(keywords, title_tokens)
    score += max(title_score, title_partial * 0.7) * 3.0

    # Category match (2x weight)
    cat_score = keyword_overlap_score(keywords, categories)
    cat_partial = partial_match_score(keywords, categories)
    score += max(cat_score, cat_partial * 0.7) * 2.0

    # Synopsis keyword overlap (1.5x weight)
    synopsis_tokens = re.findall(r"[a-z0-9]+", synopsis.lower())
    synopsis_tokens = [t for t in synopsis_tokens if t not in STOP_WORDS and len(t) > 1]
    syn_score = keyword_overlap_score(keywords, synopsis_tokens)
    score += syn_score * 1.5

    return score


def rrf_merge(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    source_names: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion merge of multiple ranked result lists.

    Each input is a list of (slug, score) tuples sorted by score descending.
    If *weights* and *source_names* are provided, each list's contribution is
    scaled by ``weights.get(source_name, 1.0)``.
    Returns merged list of (slug, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for idx, result_list in enumerate(result_lists):
        w = 1.0
        if weights and source_names and idx < len(source_names):
            w = weights.get(source_names[idx], 1.0)
        for rank, (slug, _original_score) in enumerate(result_list):
            if slug not in scores:
                scores[slug] = 0.0
            scores[slug] += w * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def combined_score(
    tier0_score: float,
    tier1_score: float = 0.0,
    tier2_score: float = 0.0,
) -> float:
    """Combine scores from multiple tiers."""
    return tier0_score * 1.0 + tier1_score * 0.8 + tier2_score * 0.6


def classify_query(query: str) -> str:
    """Classify a query as 'structured' or 'semantic'.

    Structured: metadata-answerable (categories, types, counts, lists)
    Semantic: needs content (meaning, how-to, comparison, explanation)
    """
    structured_signals = [
        r"\bwhich files?\b", r"\bhow many\b", r"\blist all\b",
        r"\bwhat categor", r"\bwhat type", r"\bcount\b",
        r"\bshow all\b", r"\bwhat are the\b",
    ]
    query_lower = query.lower()
    for pattern in structured_signals:
        if re.search(pattern, query_lower):
            return "structured"
    return "semantic"
