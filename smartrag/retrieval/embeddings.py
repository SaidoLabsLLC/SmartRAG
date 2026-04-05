"""Semantic embedding index backed by SQLite and sentence-transformers.

Provides dense vector search over the SmartRAG article corpus using cosine
similarity.  When ``sentence-transformers`` is not installed, the class
degrades gracefully — all public methods become silent no-ops.

Hook-compatible: exposes ``on_document_upsert`` and ``on_document_delete``
for integration with the ``MarkdownStore`` hook system.
"""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
import struct
from typing import Any

logger = logging.getLogger(__name__)

# Embedding dimension for all-MiniLM-L6-v2.
_EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_CREATE_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS wiki_embeddings (
    slug      TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
)
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_sentence_transformers() -> bool:
    """Return True if sentence-transformers is importable."""
    return importlib.util.find_spec("sentence_transformers") is not None


def _serialize_vector(vec: Any) -> bytes:
    """Pack a 1-D float array into a compact little-endian binary blob.

    Accepts any array-like with a ``tolist()`` method (numpy/torch) or a
    plain Python list of floats.
    """
    floats: list[float] = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    return struct.pack(f"<{len(floats)}f", *floats)


def _deserialize_vector(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a Python list of floats."""
    count = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"<{count}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using only stdlib math."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# EmbeddingIndex
# ---------------------------------------------------------------------------


class EmbeddingIndex:
    """SQLite-backed dense vector index using sentence-transformers.

    When ``sentence-transformers`` is not installed the index reports itself
    as unavailable and every mutating or search method silently returns an
    empty / no-op result.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file (typically the same
        ``wiki.db`` used by :class:`~smartrag.retrieval.fts.FTSIndex`).
    model_name:
        HuggingFace model identifier.  Defaults to ``all-MiniLM-L6-v2``
        which produces 384-dimensional embeddings.
    """

    def __init__(
        self,
        db_path: str,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._db_path = db_path
        self._model_name = model_name
        self._model: Any = None  # lazy-loaded SentenceTransformer instance
        self._available = _has_sentence_transformers()
        self._conn: sqlite3.Connection | None = None

        if not self._available:
            logger.info(
                "sentence-transformers is not installed — "
                "EmbeddingIndex will operate as a no-op"
            )
            return

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_CREATE_EMBEDDINGS)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    #  Lazy model loading
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> bool:
        """Lazily import and instantiate the embedding model.

        Returns ``True`` when the model is ready, ``False`` when the
        dependency is missing.
        """
        if not self._available:
            return False
        if self._model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "Loaded embedding model '%s' (dim=%d)",
                self._model_name,
                _EMBEDDING_DIM,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load embedding model: %s", exc)
            self._available = False
            return False

    # ------------------------------------------------------------------ #
    #  Availability
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """Return ``True`` if sentence-transformers is installed and usable."""
        return self._available

    # ------------------------------------------------------------------ #
    #  Indexing
    # ------------------------------------------------------------------ #

    def embed_article(self, slug: str, text: str) -> None:
        """Generate and store an embedding for *text* keyed by *slug*.

        Only the first 512 whitespace-delimited tokens of *text* are
        considered to stay within the model's context window and keep
        embedding time predictable.
        """
        if not self._ensure_model() or self._conn is None:
            return

        truncated = " ".join(text.split()[:512])
        if not truncated:
            return

        vector = self._model.encode(truncated, show_progress_bar=False)
        blob = _serialize_vector(vector)

        try:
            self._conn.execute(
                "INSERT INTO wiki_embeddings (slug, embedding) VALUES (?, ?) "
                "ON CONFLICT(slug) DO UPDATE SET embedding = excluded.embedding",
                (slug, blob),
            )
            self._conn.commit()
        except Exception:
            if self._conn:
                self._conn.rollback()
            raise

    def embed_batch(self, articles: list[tuple[str, str]]) -> None:
        """Batch-embed a list of ``(slug, text)`` pairs.

        Uses the model's built-in batch encoding for significantly better
        throughput than calling :meth:`embed_article` in a loop.
        """
        if not self._ensure_model() or self._conn is None or not articles:
            return

        slugs: list[str] = []
        texts: list[str] = []
        for slug, text in articles:
            truncated = " ".join(text.split()[:512])
            if truncated:
                slugs.append(slug)
                texts.append(truncated)

        if not texts:
            return

        vectors = self._model.encode(texts, show_progress_bar=False, batch_size=64)

        try:
            for slug, vec in zip(slugs, vectors):
                blob = _serialize_vector(vec)
                self._conn.execute(
                    "INSERT INTO wiki_embeddings (slug, embedding) VALUES (?, ?) "
                    "ON CONFLICT(slug) DO UPDATE SET embedding = excluded.embedding",
                    (slug, blob),
                )
            self._conn.commit()
        except Exception:
            if self._conn:
                self._conn.rollback()
            raise

        logger.info("Batch-embedded %d articles", len(slugs))

    def remove_embedding(self, slug: str) -> None:
        """Delete the stored embedding for *slug*."""
        if not self._available or self._conn is None:
            return

        try:
            self._conn.execute(
                "DELETE FROM wiki_embeddings WHERE slug = ?", (slug,)
            )
            self._conn.commit()
        except Exception:
            if self._conn:
                self._conn.rollback()
            raise

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #

    def search_semantic(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Embed *query* and return the *top_k* most similar articles.

        Returns a list of ``(slug, cosine_similarity)`` tuples sorted by
        descending similarity.  Falls back to an empty list when the
        embedding layer is unavailable or the table is empty.
        """
        if not self._ensure_model() or self._conn is None:
            return []

        query_vec = self._model.encode(query, show_progress_bar=False)
        query_list: list[float] = (
            query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
        )

        rows = self._conn.execute(
            "SELECT slug, embedding FROM wiki_embeddings"
        ).fetchall()

        if not rows:
            return []

        scored: list[tuple[str, float]] = []
        for slug, blob in rows:
            doc_vec = _deserialize_vector(blob)
            sim = _cosine_similarity(query_list, doc_vec)
            scored.append((slug, sim))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------ #
    #  Hook-compatible callbacks
    # ------------------------------------------------------------------ #

    def on_document_upsert(
        self,
        slug: str,
        frontmatter: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> None:
        """Hook callback: embed or re-embed an article.

        Designed to be registered as a ``created`` / ``updated`` hook on
        ``MarkdownStore``.
        """
        self.embed_article(slug, body or "")

    def on_document_delete(
        self,
        slug: str,
        frontmatter: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> None:
        """Hook callback: remove the embedding for an article.

        Designed to be registered as a ``deleted`` hook on ``MarkdownStore``.
        """
        self.remove_embedding(slug)

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
