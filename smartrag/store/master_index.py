"""Master index for SmartRAG — in-memory cache backed by _index.md.

Maintains a ``slug → IndexEntry`` mapping in memory with persistence to a
human-readable markdown table.  All file writes are atomic (write to .tmp,
then ``os.replace``).

Hook-compatible: exposes ``on_document_upsert`` and ``on_document_delete``
for integration with the ``MarkdownStore`` hook system.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any

from smartrag.types import IndexEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Markdown table parsing helpers
# ---------------------------------------------------------------------------

_TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<slug>[^|]+?)\s*"
    r"\|\s*(?P<title>[^|]+?)\s*"
    r"\|\s*(?P<categories>[^|]*?)\s*"
    r"\|\s*(?P<synopsis>[^|]*?)\s*"
    r"\|\s*(?P<fingerprint>[^|]*?)\s*\|$"
)

_SEPARATOR_RE = re.compile(r"^\|[-\s|]+\|$")


def _parse_csv_field(value: str) -> list[str]:
    """Split a comma-separated markdown cell into a list of stripped strings."""
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# MasterIndex
# ---------------------------------------------------------------------------


class MasterIndex:
    """In-memory master index backed by a markdown ``_index.md`` file.

    Parameters
    ----------
    index_path:
        Filesystem path to the ``_index.md`` file.
    cache_size:
        Maximum number of entries the in-memory cache will hold.  Purely
        advisory at the moment — used for documentation and future eviction.
    """

    def __init__(self, index_path: str, cache_size: int = 50_000) -> None:
        self._index_path = index_path
        self._cache_size = cache_size
        self._cache: dict[str, IndexEntry] = {}
        self._lock = threading.Lock()

        self._load()

    # ------------------------------------------------------------------ #
    #  Public read API
    # ------------------------------------------------------------------ #

    def get(self, slug: str) -> IndexEntry | None:
        """Return the ``IndexEntry`` for *slug*, or ``None`` if absent."""
        return self._cache.get(slug)

    def search(
        self,
        keywords: list[str],
        top_k: int = 10,
    ) -> list[tuple[IndexEntry, float]]:
        """Score every entry against *keywords* and return the top-K matches.

        Scoring weights:
        - fingerprint token match: **3x**
        - category match:          **2x**
        - synopsis word overlap:   **1.5x**
        - title word match:        **3x**

        Returns a list of ``(IndexEntry, score)`` tuples sorted descending by
        score.  Entries with a score of 0.0 are excluded.
        """
        if not keywords:
            return []

        kw_lower = [kw.lower() for kw in keywords]
        scored: list[tuple[IndexEntry, float]] = []

        for entry in self._cache.values():
            score = self._score_entry(entry, kw_lower)
            if score > 0.0:
                scored.append((entry, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def all_entries(self) -> list[IndexEntry]:
        """Return a list of all cached ``IndexEntry`` objects."""
        return list(self._cache.values())

    def count(self) -> int:
        """Return the number of entries in the index."""
        return len(self._cache)

    # ------------------------------------------------------------------ #
    #  Public write API
    # ------------------------------------------------------------------ #

    def upsert(self, slug: str, entry: IndexEntry) -> None:
        """Insert or update an entry, then persist to ``_index.md``."""
        with self._lock:
            self._cache[slug] = entry
            self._persist()

    def remove(self, slug: str) -> None:
        """Remove an entry by slug and persist.  No-op if slug absent."""
        with self._lock:
            if slug in self._cache:
                del self._cache[slug]
                self._persist()

    def rebuild(self, entries: list[IndexEntry]) -> None:
        """Replace the entire index with *entries* and persist."""
        with self._lock:
            self._cache.clear()
            for entry in entries:
                self._cache[entry.slug] = entry
            self._persist()

    # ------------------------------------------------------------------ #
    #  Hook-compatible callbacks
    # ------------------------------------------------------------------ #

    def on_document_upsert(
        self,
        slug: str,
        frontmatter: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> None:
        """Build an ``IndexEntry`` from *frontmatter* and upsert it.

        Designed to be registered as a ``created`` / ``updated`` hook on
        ``MarkdownStore``.
        """
        fm = frontmatter or {}
        entry = IndexEntry(
            slug=slug,
            title=fm.get("title", slug),
            categories=fm.get("categories", []),
            synopsis=fm.get("summary", ""),
            fingerprint=fm.get("fingerprint", []),
            has_children=bool(fm.get("children")),
            parent=fm.get("parent"),
        )
        self.upsert(slug, entry)

    def on_document_delete(
        self,
        slug: str,
        frontmatter: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> None:
        """Remove an entry from the index.

        Designed to be registered as a ``deleted`` hook on ``MarkdownStore``.
        """
        self.remove(slug)

    # ------------------------------------------------------------------ #
    #  Persistence — loading
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Load entries from ``_index.md`` into ``self._cache``.

        Creates an empty index file if the path does not exist.
        """
        if not os.path.exists(self._index_path):
            self._persist()  # seed empty file
            return

        try:
            with open(self._index_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            logger.warning(
                "Could not read index file %s — starting empty", self._index_path
            )
            return

        in_table = False
        for line in lines:
            stripped = line.rstrip("\n")

            # Skip YAML frontmatter
            if stripped.startswith("---"):
                continue

            # Skip the header row (contains "slug")
            if "| slug" in stripped.lower():
                in_table = True
                continue

            # Skip separator rows
            if _SEPARATOR_RE.match(stripped):
                continue

            if not in_table:
                continue

            match = _TABLE_ROW_RE.match(stripped)
            if match is None:
                continue

            slug = match.group("slug").strip()
            if not slug:
                continue

            entry = IndexEntry(
                slug=slug,
                title=match.group("title").strip(),
                categories=_parse_csv_field(match.group("categories")),
                synopsis=match.group("synopsis").strip(),
                fingerprint=_parse_csv_field(match.group("fingerprint")),
            )
            self._cache[slug] = entry

        logger.info(
            "MasterIndex loaded %d entries from %s",
            len(self._cache),
            self._index_path,
        )

    # ------------------------------------------------------------------ #
    #  Persistence — writing (atomic)
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        """Regenerate ``_index.md`` from the in-memory cache.

        Uses atomic write: creates a temporary file in the same directory,
        writes all content, then renames over the target.
        """
        content = self._render_markdown()
        target = self._index_path
        parent_dir = os.path.dirname(target) or "."
        os.makedirs(parent_dir, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=parent_dir,
            prefix=".tmp_index_",
            suffix=".md",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp_path, target)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _render_markdown(self) -> str:
        """Render the full ``_index.md`` content from the cache."""
        now = datetime.now(timezone.utc).isoformat()
        entry_count = len(self._cache)

        lines: list[str] = [
            "---",
            "type: master_index",
            f"article_count: {entry_count}",
            f"last_updated: {now}",
            "---",
            "",
            "| slug | title | categories | synopsis | fingerprint |",
            "|------|-------|-----------|----------|-------------|",
        ]

        for entry in self._cache.values():
            cats = ", ".join(entry.categories)
            fps = ", ".join(entry.fingerprint)
            # Escape pipe characters in free-text fields
            title = entry.title.replace("|", "\\|")
            synopsis = entry.synopsis.replace("|", "\\|")
            lines.append(f"| {entry.slug} | {title} | {cats} | {synopsis} | {fps} |")

        lines.append("")  # trailing newline
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Scoring
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_entry(entry: IndexEntry, kw_lower: list[str]) -> float:
        """Compute a weighted relevance score for *entry* against keywords."""
        score = 0.0

        # Pre-compute lowercase token sets
        fp_tokens = {tok.lower() for tok in entry.fingerprint}
        cat_tokens = {cat.lower() for cat in entry.categories}
        title_tokens = set(entry.title.lower().split())
        synopsis_tokens = set(entry.synopsis.lower().split())

        for kw in kw_lower:
            # Fingerprint match (3x)
            if kw in fp_tokens:
                score += 3.0

            # Category match (2x)
            if kw in cat_tokens:
                score += 2.0

            # Synopsis word overlap (1.5x)
            if kw in synopsis_tokens:
                score += 1.5

            # Title word match (3x)
            if kw in title_tokens:
                score += 3.0

        return score
