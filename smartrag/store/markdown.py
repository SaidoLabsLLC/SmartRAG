"""Markdown document store with atomic writes and hook system."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from smartrag.store.frontmatter import (
    parse_frontmatter,
    update_frontmatter,
    write_frontmatter,
)
from smartrag.types import Document

logger = logging.getLogger(__name__)


class DocumentNotFoundError(Exception):
    """Raised when a requested document does not exist."""

    def __init__(self, slug: str) -> None:
        self.slug = slug
        super().__init__(f"Document not found: {slug}")


class MarkdownStore:
    """File-backed markdown document store with CRUD, hooks, and atomic I/O."""

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir).resolve()
        self.documents_dir = self.root / "documents"
        self.smartrag_dir = self.root / ".smartrag"

        self._lock = threading.Lock()
        self._hooks: dict[str, list[Callable]] = {
            "created": [],
            "updated": [],
            "deleted": [],
        }

        self._ensure_directory_structure()

    def _ensure_directory_structure(self) -> None:
        """Create the required directory tree and seed files if absent."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.smartrag_dir.mkdir(parents=True, exist_ok=True)

        # Seed _index.md
        index_path = self.root / "_index.md"
        if not index_path.exists():
            self._atomic_write(
                index_path,
                write_frontmatter(
                    {"title": "Index", "created": _now_iso()},
                    "# SmartRAG Knowledge Base\n",
                ),
            )

        # Seed .smartrag JSON files (db file is NOT created)
        for name, default in (
            ("dedup.json", {}),
            ("config.json", {}),
            ("stats.json", {"documents": 0, "last_updated": _now_iso()}),
        ):
            path = self.smartrag_dir / name
            if not path.exists():
                self._atomic_write(path, json.dumps(default, indent=2))

        # Seed backlinks.json
        backlinks_path = self.root / "backlinks.json"
        if not backlinks_path.exists():
            self._atomic_write(backlinks_path, json.dumps({}, indent=2))

    # ------------------------------------------------------------------ #
    #  CRUD
    # ------------------------------------------------------------------ #

    def create(
        self,
        slug: str,
        body: str,
        frontmatter: dict[str, Any] | None = None,
    ) -> Document:
        """Create a new document. Returns the persisted Document."""
        fm = dict(frontmatter) if frontmatter else {}
        now = _now_iso()
        fm.setdefault("created", now)
        fm.setdefault("updated", now)

        content = write_frontmatter(fm, body)
        path = self._slug_path(slug)

        with self._lock:
            self._atomic_write(path, content)

        self._fire_hooks("created", slug, frontmatter=fm, body=body)

        return self._build_document(slug, fm, body)

    def read(self, slug: str) -> Document:
        """Read a document by slug. Raises DocumentNotFoundError if missing."""
        path = self._slug_path(slug)
        if not path.exists():
            raise DocumentNotFoundError(slug)

        content = self._read_file(path)
        fm, body = parse_frontmatter(content)

        return self._build_document(slug, fm, body)

    def read_frontmatter(self, slug: str) -> dict[str, Any]:
        """Streaming read of YAML frontmatter only — stops at closing ``---``."""
        path = self._slug_path(slug)
        if not path.exists():
            raise DocumentNotFoundError(slug)

        lines: list[str] = []
        found_open = False

        with open(path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if not found_open:
                    if line.strip() == "---":
                        found_open = True
                    continue
                if line.strip() == "---":
                    break
                lines.append(line)

        if not lines:
            return {}

        # Re-assemble and parse via the canonical parser so behaviour is
        # identical to a full read (handles YAML edge-cases in one place).
        synthetic = "---\n" + "\n".join(lines) + "\n---\n"
        fm, _ = parse_frontmatter(synthetic)
        return fm

    def update(
        self,
        slug: str,
        body: str | None = None,
        frontmatter_updates: dict[str, Any] | None = None,
    ) -> Document:
        """Update an existing document's body and/or frontmatter."""
        path = self._slug_path(slug)
        if not path.exists():
            raise DocumentNotFoundError(slug)

        content = self._read_file(path)

        updates = dict(frontmatter_updates) if frontmatter_updates else {}
        updates["updated"] = _now_iso()

        if body is not None:
            # Full rewrite: apply frontmatter updates then write new body.
            fm, _ = parse_frontmatter(content)
            fm.update(updates)
            new_content = write_frontmatter(fm, body)
        else:
            # Frontmatter-only patch — preserve body unchanged.
            new_content = update_frontmatter(content, updates)
            fm, body = parse_frontmatter(new_content)

        with self._lock:
            self._atomic_write(path, new_content)

        final_fm, final_body = parse_frontmatter(new_content)
        self._fire_hooks("updated", slug, frontmatter=final_fm, body=final_body)

        return self._build_document(slug, final_fm, final_body)

    def delete(self, slug: str) -> None:
        """Delete a document and cascade to children if it is a parent."""
        path = self._slug_path(slug)
        if not path.exists():
            raise DocumentNotFoundError(slug)

        # Read frontmatter to check for children before deletion.
        content = self._read_file(path)
        fm, _ = parse_frontmatter(content)

        children: list[str] = fm.get("children") or []

        with self._lock:
            # Remove children first.
            for child_slug in children:
                child_path = self._slug_path(child_slug)
                if child_path.exists():
                    child_path.unlink()
                    self._fire_hooks("deleted", child_slug)

            path.unlink()

        self._fire_hooks("deleted", slug, frontmatter=fm)

    def exists(self, slug: str) -> bool:
        """Return True if the document file exists."""
        return self._slug_path(slug).exists()

    def list_all(self) -> list[tuple[str, str, str]]:
        """Return ``(slug, title, summary)`` for every document."""
        results: list[tuple[str, str, str]] = []
        for md_file in sorted(self.documents_dir.glob("*.md")):
            slug = md_file.stem
            try:
                fm = self.read_frontmatter(slug)
                title = fm.get("title", slug)
                summary = fm.get("summary", "")
                results.append((slug, title, summary))
            except Exception:
                logger.warning("Skipping unreadable document: %s", slug)
        return results

    def count(self) -> int:
        """Return the total number of ``.md`` files in the documents directory."""
        return sum(1 for _ in self.documents_dir.glob("*.md"))

    # ------------------------------------------------------------------ #
    #  Slug management
    # ------------------------------------------------------------------ #

    def generate_slug(self, title: str) -> str:
        """Derive a URL-safe slug from *title*, appending a suffix if needed."""
        slug = title.lower().strip()
        # Replace spaces and underscores with hyphens.
        slug = re.sub(r"[\s_]+", "-", slug)
        # Strip anything that is not alphanumeric or a hyphen.
        slug = re.sub(r"[^a-z0-9-]", "", slug)
        # Collapse consecutive hyphens.
        slug = re.sub(r"-{2,}", "-", slug)
        # Strip leading / trailing hyphens.
        slug = slug.strip("-")
        # Truncate to 60 chars on a hyphen boundary.
        slug = _truncate_at_boundary(slug, 60)

        if not slug:
            slug = "untitled"

        # Deduplicate by appending -2, -3, ...
        base = slug
        counter = 2
        while self.exists(slug):
            candidate = f"{base}-{counter}"
            # Re-truncate in case suffix pushes past 60.
            slug = _truncate_at_boundary(candidate, 60)
            counter += 1

        return slug

    # ------------------------------------------------------------------ #
    #  Hook system
    # ------------------------------------------------------------------ #

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for *event* (``created``, ``updated``, ``deleted``)."""
        if event not in self._hooks:
            raise ValueError(
                f"Unknown event '{event}'. Must be one of: {list(self._hooks)}"
            )
        self._hooks[event].append(callback)

    def _fire_hooks(
        self,
        event: str,
        slug: str,
        frontmatter: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> None:
        """Invoke all registered callbacks for *event*. Failures are logged."""
        for cb in self._hooks.get(event, []):
            try:
                cb(slug=slug, frontmatter=frontmatter, body=body)
            except Exception:
                logger.exception(
                    "Hook %s failed for event '%s' on slug '%s'",
                    cb,
                    event,
                    slug,
                )

    # ------------------------------------------------------------------ #
    #  File I/O helpers
    # ------------------------------------------------------------------ #

    def _slug_path(self, slug: str) -> Path:
        """Return the filesystem path for a given slug."""
        return self.documents_dir / f"{slug}.md"

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read a file with enforced UTF-8 encoding."""
        return path.read_text(encoding="utf-8")

    def _atomic_write(self, target: Path, content: str) -> None:
        """Write *content* to a temp file then atomically rename to *target*.

        On Windows ``os.replace`` is used for an atomic overwrite.  The temp
        file is created in the same directory so the rename stays on a single
        filesystem.
        """
        target.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=".tmp_",
            suffix=".md",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp_path, str(target))
        except BaseException:
            # Clean up the temp file on any failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------ #
    #  Document builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_document(
        slug: str,
        frontmatter: dict[str, Any],
        body: str,
    ) -> Document:
        """Construct a ``Document`` dataclass from raw parts."""
        title = frontmatter.get("title", slug)
        word_count = len(body.split())
        has_children = bool(frontmatter.get("children"))
        return Document(
            slug=slug,
            title=title,
            body=body,
            frontmatter=frontmatter,
            word_count=word_count,
            has_children=has_children,
        )


# ------------------------------------------------------------------ #
#  Module-level helpers
# ------------------------------------------------------------------ #


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _truncate_at_boundary(slug: str, max_len: int) -> str:
    """Truncate *slug* to at most *max_len* characters on a hyphen boundary."""
    if len(slug) <= max_len:
        return slug

    truncated = slug[:max_len]
    # Try to cut at the last hyphen so we don't split a word.
    last_hyphen = truncated.rfind("-")
    if last_hyphen > 0:
        truncated = truncated[:last_hyphen]

    return truncated.rstrip("-")
