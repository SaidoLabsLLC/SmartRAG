"""FeedbackStore — SQLite-backed query logging and feedback storage."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time


class FeedbackStore:
    """Persistent store for query logs, feedback scores, and synopsis quality."""

    def __init__(self, db_path: str, anonymize: bool = False):
        self._db_path = db_path
        self._anonymize = anonymize
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                query_hash TEXT,
                timestamp REAL,
                total_ms REAL,
                total_bytes_read INTEGER,
                tier_max INTEGER,
                result_slugs TEXT,
                result_sources TEXT,
                result_count INTEGER,
                feedback_score REAL DEFAULT NULL,
                feedback_slugs TEXT DEFAULT NULL,
                feedback_at REAL DEFAULT NULL,
                signal_type TEXT DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS synopsis_quality (
                slug TEXT PRIMARY KEY,
                tier3_count INTEGER DEFAULT 0,
                total_query_count INTEGER DEFAULT 0,
                last_flagged REAL DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS tuning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                weights TEXT,
                data_points INTEGER,
                trigger TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_query_log_hash
                ON query_log(query_hash);
            CREATE INDEX IF NOT EXISTS idx_query_log_feedback
                ON query_log(feedback_score) WHERE feedback_score IS NOT NULL;
            """
        )
        self._conn.commit()

    @staticmethod
    def _hash_query(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def log_query(self, query_result, anonymize: bool = False) -> int:
        """Log a query and its results. Returns the query_log row id."""
        use_anon = anonymize or self._anonymize
        query_text = (
            self._hash_query(query_result.query) if use_anon else query_result.query
        )
        query_hash = self._hash_query(query_result.query)
        tier_max = (
            max((r.tier_resolved for r in query_result.results), default=0)
            if query_result.results
            else 0
        )
        result_slugs = [r.slug for r in query_result.results]
        # source_map may be attached to the QueryResult by the router
        source_map = getattr(query_result, "_source_map", {})
        result_sources = [source_map.get(r.slug, "unknown") for r in query_result.results]

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO query_log
               (query_text, query_hash, timestamp, total_ms, total_bytes_read,
                tier_max, result_slugs, result_sources, result_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                query_text,
                query_hash,
                time.time(),
                query_result.total_ms,
                query_result.total_bytes_read,
                tier_max,
                json.dumps(result_slugs),
                json.dumps(result_sources),
                len(query_result.results),
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def record_feedback(
        self,
        query_id: int,
        score: float,
        used_slugs: list[str] | None = None,
        signal_type: str = "explicit",
    ) -> None:
        """Record feedback for a previously-logged query."""
        self._conn.execute(
            """UPDATE query_log
               SET feedback_score = ?, feedback_slugs = ?,
                   feedback_at = ?, signal_type = ?
               WHERE id = ?""",
            (
                score,
                json.dumps(used_slugs) if used_slugs else None,
                time.time(),
                signal_type,
                query_id,
            ),
        )
        self._conn.commit()

    def record_tier_penalty(self, query_id: int, slug: str) -> None:
        """Increment tier-3 penalty counter for a document."""
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO synopsis_quality (slug, tier3_count, total_query_count, last_flagged)
               VALUES (?, 1, 1, ?)
               ON CONFLICT(slug) DO UPDATE SET
                   tier3_count = tier3_count + 1,
                   total_query_count = total_query_count + 1,
                   last_flagged = ?""",
            (slug, time.time(), time.time()),
        )
        self._conn.commit()

    def get_stats(self) -> dict:
        """Return aggregate feedback statistics."""
        cur = self._conn.cursor()
        total = cur.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        with_fb = cur.execute(
            "SELECT COUNT(*) FROM query_log WHERE feedback_score IS NOT NULL"
        ).fetchone()[0]
        avg_score = cur.execute(
            "SELECT AVG(feedback_score) FROM query_log WHERE feedback_score IS NOT NULL"
        ).fetchone()[0]

        # Source win rates: fraction of times each source had the top-1 result
        rows = cur.execute(
            "SELECT result_sources FROM query_log WHERE feedback_score IS NOT NULL AND feedback_score >= 0.7"
        ).fetchall()
        source_counts: dict[str, int] = {}
        for row in rows:
            sources = json.loads(row[0]) if row[0] else []
            if sources:
                top = sources[0]
                source_counts[top] = source_counts.get(top, 0) + 1
        total_wins = sum(source_counts.values()) or 1
        source_win_rates = {s: c / total_wins for s, c in source_counts.items()}

        # Tier distribution
        tier_rows = cur.execute(
            "SELECT tier_max, COUNT(*) FROM query_log GROUP BY tier_max"
        ).fetchall()
        tier_distribution = {row[0]: row[1] for row in tier_rows}

        return {
            "total_queries": total,
            "feedback_rate": with_fb / total if total else 0.0,
            "avg_score": avg_score,
            "source_win_rates": source_win_rates,
            "tier_distribution": tier_distribution,
        }

    def get_tuning_data(self, min_queries: int = 50) -> dict | None:
        """Return tuning data if enough feedback is available."""
        cur = self._conn.cursor()
        count = cur.execute(
            "SELECT COUNT(*) FROM query_log WHERE feedback_score IS NOT NULL AND feedback_score >= 0.7"
        ).fetchone()[0]
        if count < min_queries:
            return None

        rows = cur.execute(
            "SELECT result_sources FROM query_log WHERE feedback_score IS NOT NULL AND feedback_score >= 0.7"
        ).fetchall()

        source_counts: dict[str, int] = {}
        total_rows = 0
        for row in rows:
            sources = json.loads(row[0]) if row[0] else []
            if sources:
                top = sources[0]
                source_counts[top] = source_counts.get(top, 0) + 1
                total_rows += 1

        total_wins = sum(source_counts.values()) or 1
        win_rates = {s: c / total_wins for s, c in source_counts.items()}
        return {"win_rates": win_rates, "data_points": count}

    def get_flagged_documents(
        self, min_queries: int = 5, threshold: float = 0.5
    ) -> list[str]:
        """Return slugs of documents with high tier-3 resolution rate."""
        cur = self._conn.cursor()
        rows = cur.execute(
            """SELECT slug, tier3_count, total_query_count
               FROM synopsis_quality
               WHERE total_query_count >= ?""",
            (min_queries,),
        ).fetchall()
        flagged = []
        for row in rows:
            rate = row[1] / row[2] if row[2] else 0.0
            if rate > threshold:
                flagged.append(row[0])
        return flagged

    def close(self) -> None:
        self._conn.close()
