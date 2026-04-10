"""SignalDetector — implicit feedback signal detection from query patterns."""

from __future__ import annotations

import time
from collections import deque

from smartrag.retrieval.scorer import tokenize_query


class SignalDetector:
    """Detect implicit feedback signals from sequential query patterns."""

    def __init__(self, store):
        self._store = store
        self._recent: deque = deque(maxlen=20)

    def on_query(self, query_id: int, query_result) -> None:
        """Call after every retrieve(). Detects implicit signals."""
        now = time.time()
        keywords = set(tokenize_query(query_result.query))
        entry = {
            "query_id": query_id,
            "query_hash": self._store._hash_query(query_result.query),
            "keywords": keywords,
            "timestamp": now,
            "results": query_result.results,
        }

        # Check against previous queries for implicit signals
        for prev in reversed(self._recent):
            elapsed = now - prev["timestamp"]

            # a. Repeat query: same hash within 60s
            if entry["query_hash"] == prev["query_hash"] and elapsed <= 60:
                self._store.record_feedback(
                    prev["query_id"], 0.3, signal_type="repeat_query"
                )
                break

            # b. Refinement: >60% keyword overlap within 5 min
            if elapsed <= 300 and keywords and prev["keywords"]:
                overlap = len(keywords & prev["keywords"]) / len(
                    keywords | prev["keywords"]
                )
                if overlap > 0.6:
                    self._store.record_feedback(
                        prev["query_id"], 0.5, signal_type="refinement"
                    )
                    break

            # c. Topic switch: <20% overlap within 30s
            if elapsed <= 30 and keywords and prev["keywords"]:
                overlap = len(keywords & prev["keywords"]) / len(
                    keywords | prev["keywords"]
                )
                if overlap < 0.2:
                    self._store.record_feedback(
                        prev["query_id"], 0.8, signal_type="topic_switch"
                    )
                    break

        # d. Tier penalty: any result at tier 3
        for r in query_result.results:
            if r.tier_resolved == 3:
                self._store.record_tier_penalty(query_id, r.slug)

        self._recent.append(entry)
