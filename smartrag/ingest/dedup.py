"""Content deduplication index for SmartRAG.

Maintains a persistent SHA-256 hash index that maps normalised content
hashes to document slugs.  Used during ingestion to detect exact
duplicates and content updates (same slug, different hash).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from smartrag.types import DedupResult

logger = logging.getLogger(__name__)


class DedupIndex:
    """Persistent content-hash index backed by a JSON file.

    Storage format (``dedup.json``)::

        {"<sha256-hex>": "<slug>", ...}

    Parameters
    ----------
    index_path:
        Absolute or relative path to the JSON file
        (typically ``.smartrag/dedup.json``).
    """

    def __init__(self, index_path: str) -> None:
        self._path = Path(index_path)
        self._index: dict[str, str] = {}  # hash → slug
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(text: str) -> str:
        """Return the SHA-256 hex digest of *text* after normalisation.

        Normalisation: strip leading/trailing whitespace, collapse to
        lowercase.  This ensures minor formatting changes do not produce
        a different hash.
        """
        normalised = text.strip().lower()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def check(self, text: str) -> DedupResult:
        """Check whether *text* already exists in the index.

        Returns
        -------
        DedupResult
            * ``is_duplicate=True, existing_slug=<slug>`` when the exact
              content is already registered under a **different** slug.
            * ``is_duplicate=False, is_update=True, existing_slug=<slug>``
              is not returned here — update detection requires slug
              context and is handled by comparing hashes externally.
            * ``is_duplicate=False`` when no matching hash is found.
        """
        content_hash = self.compute_hash(text)
        existing_slug = self._index.get(content_hash)

        if existing_slug is not None:
            return DedupResult(
                is_duplicate=True,
                existing_slug=existing_slug,
                is_update=False,
            )

        return DedupResult(is_duplicate=False)

    def register(self, slug: str, text: str) -> None:
        """Register *slug* with the hash of *text*.

        If the slug was previously registered under a different hash the
        old entry is removed first so that stale hashes do not linger.
        """
        content_hash = self.compute_hash(text)

        # Remove any previous hash that pointed to this slug (content update).
        stale_hashes = [h for h, s in self._index.items() if s == slug]
        for h in stale_hashes:
            del self._index[h]

        self._index[content_hash] = slug
        logger.debug("Registered hash %s → %s", content_hash[:12], slug)

    def remove(self, slug: str) -> None:
        """Remove all index entries that map to *slug*."""
        to_remove = [h for h, s in self._index.items() if s == slug]
        for h in to_remove:
            del self._index[h]
        if to_remove:
            logger.debug("Removed %d hash(es) for slug '%s'", len(to_remove), slug)

    def save(self) -> None:
        """Persist the current index to disk as JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(self._index, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            # Atomic-ish rename (best effort on Windows).
            if os.name == "nt" and self._path.exists():
                self._path.unlink()
            tmp_path.rename(self._path)
        except Exception:
            # Clean up partial write.
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        logger.debug("Saved dedup index (%d entries) → %s", len(self._index), self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load existing index from disk, or start empty."""
        if not self._path.exists():
            logger.debug("No existing dedup index at %s; starting empty", self._path)
            return

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning("Dedup index is not a dict; starting empty")
                return
            self._index = data
            logger.debug("Loaded dedup index with %d entries", len(self._index))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load dedup index: %s; starting empty", exc)
