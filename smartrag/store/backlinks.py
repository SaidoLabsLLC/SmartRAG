"""Bidirectional wikilink graph for SmartRAG documents.

Maintains forward links (slug -> [targets]) and reverse links
(target -> [sources]) extracted from ``[[slug]]`` and ``[[slug|display]]``
wikilink syntax inside markdown bodies.

Storage format (``backlinks.json``)::

    {
        "forward": {"slug": ["linked-slug", ...], ...},
        "reverse": {"slug": ["linking-slug", ...], ...}
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Matches fenced code blocks (``` ... ```).  Content inside these is
# excluded from wikilink extraction.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

# Matches [[slug]] or [[slug|display text]].
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]*?)?\]\]")


class BacklinkManager:
    """Bidirectional wikilink graph persisted as JSON.

    Parameters
    ----------
    backlinks_path:
        Absolute or relative path to the JSON file
        (typically ``.smartrag/backlinks.json``).
    """

    def __init__(self, backlinks_path: str) -> None:
        self._path = Path(backlinks_path)
        self._forward: dict[str, list[str]] = {}  # slug → [targets]
        self._reverse: dict[str, list[str]] = {}  # target → [sources]
        self._load()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_wikilinks(body: str) -> list[str]:
        """Return deduplicated slugs referenced via ``[[slug]]`` syntax.

        Links inside fenced code blocks (triple-backtick) are ignored.
        ``[[slug|display text]]`` is supported — only the slug portion
        is returned.
        """
        # Strip fenced code blocks so their content is not scanned.
        cleaned = _CODE_FENCE_RE.sub("", body)

        slugs: list[str] = []
        seen: set[str] = set()
        for match in _WIKILINK_RE.finditer(cleaned):
            slug = match.group(1).strip()
            if slug and slug not in seen:
                seen.add(slug)
                slugs.append(slug)
        return slugs

    # ------------------------------------------------------------------
    # Graph mutations
    # ------------------------------------------------------------------

    def update_links(self, slug: str, body: str) -> None:
        """Rebuild forward and reverse links for *slug* from *body*.

        Any previously recorded forward links for this slug are removed
        from the reverse index before the new links are applied, so
        deleted wikilinks are correctly cleaned up.
        """
        new_targets = self.extract_wikilinks(body)

        # Remove old reverse entries for this slug's previous targets.
        old_targets = self._forward.get(slug, [])
        for target in old_targets:
            sources = self._reverse.get(target, [])
            if slug in sources:
                sources.remove(slug)
                if not sources:
                    del self._reverse[target]
                else:
                    self._reverse[target] = sources

        # Set new forward links.
        if new_targets:
            self._forward[slug] = new_targets
        elif slug in self._forward:
            del self._forward[slug]

        # Build new reverse entries.
        for target in new_targets:
            sources = self._reverse.setdefault(target, [])
            if slug not in sources:
                sources.append(slug)

        logger.debug(
            "Updated links for '%s': %d forward links", slug, len(new_targets)
        )

    def remove_document(self, slug: str) -> None:
        """Remove *slug* from both forward and reverse indexes.

        Cleans up reverse entries for every target this slug pointed to,
        and removes reverse entries where other slugs pointed to this one.
        """
        # Clean reverse entries for targets this slug linked to.
        for target in self._forward.pop(slug, []):
            sources = self._reverse.get(target, [])
            if slug in sources:
                sources.remove(slug)
                if not sources:
                    del self._reverse[target]
                else:
                    self._reverse[target] = sources

        # Remove this slug as a reverse target entirely.
        self._reverse.pop(slug, None)

        logger.debug("Removed document '%s' from backlink graph", slug)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_backlinks(self, slug: str) -> list[str]:
        """Return slugs that link **to** *slug* (reverse links)."""
        return list(self._reverse.get(slug, []))

    def get_forward_links(self, slug: str) -> list[str]:
        """Return slugs that *slug* links **to** (forward links)."""
        return list(self._forward.get(slug, []))

    def get_related(self, slug: str, depth: int = 1) -> set[str]:
        """Return all slugs reachable within *depth* hops.

        A breadth-first traversal follows both forward and reverse edges.
        The starting *slug* is excluded from the result set.
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(slug, 0)])

        while queue:
            current, current_depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current_depth >= depth:
                continue

            # Follow both directions.
            neighbours = set(self._forward.get(current, []))
            neighbours.update(self._reverse.get(current, []))
            for neighbour in neighbours:
                if neighbour not in visited:
                    queue.append((neighbour, current_depth + 1))

        # Remove the origin slug itself.
        visited.discard(slug)
        return visited

    # ------------------------------------------------------------------
    # Hook-compatible interface
    # ------------------------------------------------------------------

    def on_document_change(
        self,
        slug: str,
        frontmatter: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> None:
        """Hook called when a document is created or updated.

        Delegates to :meth:`update_links` if *body* is provided.
        """
        if body is not None:
            self.update_links(slug, body)

    def on_document_delete(
        self,
        slug: str,
        frontmatter: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> None:
        """Hook called when a document is deleted.

        Delegates to :meth:`remove_document`.
        """
        self.remove_document(slug)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the current link graph to disk as JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "forward": self._forward,
            "reverse": self._reverse,
        }
        tmp_path = self._path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if os.name == "nt" and self._path.exists():
                self._path.unlink()
            tmp_path.rename(self._path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        logger.debug(
            "Saved backlinks (%d forward, %d reverse) → %s",
            len(self._forward),
            len(self._reverse),
            self._path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load existing graph from disk, or start empty."""
        if not self._path.exists():
            logger.debug("No existing backlinks at %s; starting empty", self._path)
            return

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning("Backlinks file is not a dict; starting empty")
                return
            self._forward = data.get("forward", {})
            self._reverse = data.get("reverse", {})
            logger.debug(
                "Loaded backlinks: %d forward, %d reverse entries",
                len(self._forward),
                len(self._reverse),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load backlinks: %s; starting empty", exc)
