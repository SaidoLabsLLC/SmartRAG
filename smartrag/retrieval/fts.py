"""Full-text search index backed by SQLite FTS5.

Provides BM25-ranked full-text search and structured metadata queries over
the SmartRAG article corpus.  All SQL uses parameterized queries exclusively.

Hook-compatible: exposes ``on_document_upsert`` and ``on_document_delete``
for integration with the ``MarkdownStore`` hook system.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from smartrag.types import FTSResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

def _sanitize_fts_query(query: str) -> str:
    """Sanitize a user query for FTS5 MATCH syntax.

    Strips FTS5-special characters and joins remaining tokens with OR
    so natural-language queries work without syntax errors.
    """
    import re
    # Remove FTS5 special syntax characters
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    tokens = [t for t in cleaned.split() if len(t) > 1]
    if not tokens:
        return ""
    # Join with OR for broad matching
    return " OR ".join(f'"{t}"' for t in tokens)


_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    slug,
    title,
    summary,
    categories,
    concepts,
    fingerprint,
    body_text,
    tokenize='porter unicode61'
)
"""

_CREATE_META = """
CREATE TABLE IF NOT EXISTS articles_meta (
    slug          TEXT PRIMARY KEY,
    title         TEXT,
    categories    TEXT,
    concepts      TEXT,
    synopsis      TEXT,
    fingerprint   TEXT,
    parent_slug   TEXT,
    word_count    INTEGER,
    has_children  INTEGER DEFAULT 0,
    updated_at    TEXT
)
"""

# ---------------------------------------------------------------------------
# FTSIndex
# ---------------------------------------------------------------------------


class FTSIndex:
    """SQLite FTS5 search index with a companion metadata table.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file.  Created automatically
        if it does not exist.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_CREATE_FTS)
        self._conn.execute(_CREATE_META)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    #  Indexing
    # ------------------------------------------------------------------ #

    def index_article(
        self,
        slug: str,
        frontmatter: dict[str, Any],
        body_text: str,
    ) -> None:
        """Index (upsert) a single article into both FTS5 and metadata tables.

        Categories, concepts, and fingerprint are stored as JSON arrays in the
        metadata table and as space-separated tokens in the FTS5 table.
        """
        title = frontmatter.get("title", slug)
        summary = frontmatter.get("summary", "")
        categories: list[str] = frontmatter.get("categories", [])
        concepts: list[str] = frontmatter.get("concepts", [])
        fingerprint: list[str] = frontmatter.get("fingerprint", [])
        parent_slug: str | None = frontmatter.get("parent")
        has_children = 1 if frontmatter.get("children") else 0
        word_count = len(body_text.split())
        now = datetime.now(timezone.utc).isoformat()

        # Space-separated for FTS5 tokenisation
        cats_fts = " ".join(categories)
        concepts_fts = " ".join(concepts)
        fp_fts = " ".join(fingerprint)

        # JSON for metadata table
        cats_json = json.dumps(categories)
        concepts_json = json.dumps(concepts)
        fp_json = json.dumps(fingerprint)

        cur = self._conn.cursor()
        try:
            # --- FTS5 upsert (delete + insert; FTS5 has no UPDATE) ---
            cur.execute("DELETE FROM wiki_fts WHERE slug = ?", (slug,))
            cur.execute(
                "INSERT INTO wiki_fts (slug, title, summary, categories, "
                "concepts, fingerprint, body_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (slug, title, summary, cats_fts, concepts_fts, fp_fts, body_text),
            )

            # --- Metadata upsert ---
            cur.execute(
                "INSERT INTO articles_meta "
                "(slug, title, categories, concepts, synopsis, fingerprint, "
                "parent_slug, word_count, has_children, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(slug) DO UPDATE SET "
                "title=excluded.title, categories=excluded.categories, "
                "concepts=excluded.concepts, synopsis=excluded.synopsis, "
                "fingerprint=excluded.fingerprint, parent_slug=excluded.parent_slug, "
                "word_count=excluded.word_count, has_children=excluded.has_children, "
                "updated_at=excluded.updated_at",
                (
                    slug,
                    title,
                    cats_json,
                    concepts_json,
                    summary,
                    fp_json,
                    parent_slug,
                    word_count,
                    has_children,
                    now,
                ),
            )

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def remove_article(self, slug: str) -> None:
        """Remove an article from both FTS5 and metadata tables."""
        cur = self._conn.cursor()
        try:
            cur.execute("DELETE FROM wiki_fts WHERE slug = ?", (slug,))
            cur.execute("DELETE FROM articles_meta WHERE slug = ?", (slug,))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def rebuild(self, articles: list[tuple[str, dict[str, Any], str]]) -> int:
        """Drop, recreate, and reindex all articles.

        Parameters
        ----------
        articles:
            List of ``(slug, frontmatter_dict, body_text)`` tuples.

        Returns
        -------
        int
            Number of articles indexed.
        """
        cur = self._conn.cursor()
        try:
            cur.execute("DROP TABLE IF EXISTS wiki_fts")
            cur.execute("DROP TABLE IF EXISTS articles_meta")
            cur.execute(_CREATE_FTS)
            cur.execute(_CREATE_META)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        count = 0
        for slug, frontmatter, body_text in articles:
            self.index_article(slug, frontmatter, body_text)
            count += 1

        logger.info("FTSIndex rebuilt with %d articles", count)
        return count

    def clear(self) -> None:
        """Delete all data from both tables without dropping them."""
        cur = self._conn.cursor()
        try:
            cur.execute("DELETE FROM wiki_fts")
            cur.execute("DELETE FROM articles_meta")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------ #
    #  Search — FTS5 (BM25)
    # ------------------------------------------------------------------ #

    def search_fts(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[FTSResult]:
        """Run a BM25-ranked FTS5 MATCH query.

        Parameters
        ----------
        query:
            FTS5 query string (supports AND, OR, NEAR, prefix*, etc.).
        top_k:
            Maximum number of results to return.
        filters:
            Optional metadata filters applied via JOIN on ``articles_meta``.
            Supported keys: ``category``, ``has_children``, ``parent_slug``,
            ``min_word_count``.

        Returns
        -------
        list[FTSResult]
            Results sorted by BM25 relevance (best first).
        """
        if not query or not query.strip():
            return []

        # Sanitize query for FTS5: strip special characters and wrap tokens
        # in quotes to prevent FTS5 syntax errors from user queries.
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []

        where_clauses: list[str] = []
        params: list[Any] = [sanitized]

        if filters:
            if "category" in filters:
                where_clauses.append(
                    "EXISTS ("
                    "  SELECT 1 FROM json_each(m.categories) "
                    "  WHERE json_each.value = ?"
                    ")"
                )
                params.append(filters["category"])

            if "has_children" in filters:
                where_clauses.append("m.has_children = ?")
                params.append(1 if filters["has_children"] else 0)

            if "parent_slug" in filters:
                where_clauses.append("m.parent_slug = ?")
                params.append(filters["parent_slug"])

            if "min_word_count" in filters:
                where_clauses.append("m.word_count >= ?")
                params.append(filters["min_word_count"])

        where_sql = ""
        if where_clauses:
            where_sql = " AND " + " AND ".join(where_clauses)

        params.append(top_k)

        sql = (
            "SELECT "
            "  f.slug, "
            "  f.title, "
            "  COALESCE(m.synopsis, f.summary) AS synopsis, "
            "  bm25(wiki_fts) AS score, "
            "  snippet(wiki_fts, 6, '<b>', '</b>', '...', 48) AS snippet "
            "FROM wiki_fts f "
            "LEFT JOIN articles_meta m ON f.slug = m.slug "
            "WHERE wiki_fts MATCH ?"
            f"{where_sql} "
            "ORDER BY bm25(wiki_fts) "
            "LIMIT ?"
        )

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 query failed: %s (query=%r)", exc, query)
            return []

        return [
            FTSResult(
                slug=row["slug"],
                title=row["title"],
                synopsis=row["synopsis"] or "",
                score=abs(row["score"]),  # BM25 returns negative; lower is better
                snippet=row["snippet"] or "",
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    #  Search — structured metadata
    # ------------------------------------------------------------------ #

    def search_structured(
        self,
        filters: dict[str, Any],
        top_k: int = 10,
    ) -> list[FTSResult]:
        """Query articles by metadata filters only (no full-text matching).

        Parameters
        ----------
        filters:
            Supported keys: ``category``, ``has_children``, ``parent_slug``,
            ``min_word_count``.
        top_k:
            Maximum results.

        Returns
        -------
        list[FTSResult]
            Matching entries ordered by ``updated_at`` descending.
        """
        where_clauses: list[str] = []
        params: list[Any] = []

        if "category" in filters:
            where_clauses.append(
                "EXISTS ("
                "  SELECT 1 FROM json_each(m.categories) "
                "  WHERE json_each.value = ?"
                ")"
            )
            params.append(filters["category"])

        if "has_children" in filters:
            where_clauses.append("m.has_children = ?")
            params.append(1 if filters["has_children"] else 0)

        if "parent_slug" in filters:
            where_clauses.append("m.parent_slug = ?")
            params.append(filters["parent_slug"])

        if "min_word_count" in filters:
            where_clauses.append("m.word_count >= ?")
            params.append(filters["min_word_count"])

        if not where_clauses:
            where_sql = "1=1"
        else:
            where_sql = " AND ".join(where_clauses)

        params.append(top_k)

        sql = (
            "SELECT slug, title, synopsis, word_count "
            "FROM articles_meta m "
            f"WHERE {where_sql} "
            "ORDER BY updated_at DESC "
            "LIMIT ?"
        )

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("Structured query failed: %s", exc)
            return []

        return [
            FTSResult(
                slug=row["slug"],
                title=row["title"],
                synopsis=row["synopsis"] or "",
                score=0.0,
                snippet="",
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    #  Hook-compatible callbacks
    # ------------------------------------------------------------------ #

    def on_document_upsert(
        self,
        slug: str,
        frontmatter: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> None:
        """Hook callback: index or re-index an article.

        Designed to be registered as a ``created`` / ``updated`` hook on
        ``MarkdownStore``.
        """
        self.index_article(slug, frontmatter or {}, body or "")

    def on_document_delete(
        self,
        slug: str,
        frontmatter: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> None:
        """Hook callback: remove an article from the index.

        Designed to be registered as a ``deleted`` hook on ``MarkdownStore``.
        """
        self.remove_article(slug)

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
