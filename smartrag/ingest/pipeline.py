"""Ingest pipeline — wires together extraction, splitting, fingerprinting, and storage."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from smartrag.config import SmartRAGConfig
from smartrag.ingest.dedup import DedupIndex
from smartrag.ingest.extractors import ExtractorRegistry, UnsupportedFormatError
from smartrag.ingest.fingerprint import Fingerprinter
from smartrag.ingest.splitter import SectionSplitter
from smartrag.types import IngestResult

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Orchestrates the full document ingestion flow."""

    def __init__(self, store, config: SmartRAGConfig):
        self._store = store
        self._config = config
        self._extractor = ExtractorRegistry()
        self._splitter = SectionSplitter(threshold=config.split_threshold)
        # Wire up LLM provider for fingerprinting when configured.
        llm = None
        if config.synopsis_mode == "llm" and config.llm_provider:
            try:
                from smartrag.ingest.llm_provider import create_provider

                llm = create_provider(
                    config.llm_provider, api_key=config.llm_api_key
                )
            except (ImportError, ValueError, ConnectionError) as exc:
                logger.warning(
                    "Could not create LLM provider (%s); "
                    "falling back to extractive mode: %s",
                    config.llm_provider,
                    exc,
                )
        self._fingerprinter = Fingerprinter(
            mode=config.synopsis_mode if llm else "extractive",
            llm_provider=llm,
        )
        self._dedup = DedupIndex(
            os.path.join(str(store.root), ".smartrag", "dedup.json")
        )

    def ingest_file(self, path: str) -> IngestResult:
        """Ingest a single file through the full pipeline."""
        try:
            # Step 1: Extract content
            extracted = self._extractor.extract(path)
            text = extracted.text
            metadata = extracted.metadata
            title = metadata.get("title", Path(path).stem)

            return self._ingest_content(text, title, metadata)

        except UnsupportedFormatError as e:
            logger.warning(f"Unsupported format: {path} — {e}")
            return IngestResult(
                slug="", title=Path(path).stem, status="failed", error=str(e)
            )
        except Exception as e:
            logger.error(f"Failed to ingest {path}: {e}")
            return IngestResult(
                slug="", title=Path(path).stem, status="failed", error=str(e)
            )

    def ingest_directory(self, path: str) -> list[IngestResult]:
        """Recursively ingest all supported files in a directory."""
        results = []
        root = Path(path)
        supported = self._extractor.supported_extensions

        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fname in filenames:
                if fname.startswith("."):
                    continue
                ext = Path(fname).suffix.lower()
                if ext in supported:
                    files.append(os.path.join(dirpath, fname))

        for i, fpath in enumerate(files):
            logger.info(f"Ingesting ({i+1}/{len(files)}): {fpath}")
            result = self.ingest_file(fpath)
            results.append(result)

        return results

    def ingest_url(self, url: str) -> IngestResult:
        """Ingest content from a URL through the full pipeline."""
        try:
            from smartrag.ingest.url_fetcher import URLFetcher

            fetcher = URLFetcher()
            extracted = fetcher.fetch(url)
            title = extracted.metadata.get("title", url)
            return self._ingest_content(extracted.text, title, extracted.metadata)
        except Exception as e:
            logger.error(f"Failed to ingest URL {url}: {e}")
            return IngestResult(
                slug="", title=url, status="failed", error=str(e)
            )

    def ingest_text(
        self, text: str, title: str, metadata: dict | None = None
    ) -> IngestResult:
        """Ingest raw text directly."""
        meta = metadata or {}
        meta.setdefault("title", title)
        return self._ingest_content(text, title, meta)

    def _ingest_content(
        self, text: str, title: str, metadata: dict
    ) -> IngestResult:
        """Core ingestion logic shared by file and text ingest."""
        # Step 2: Dedup check
        dedup_result = self._dedup.check(text)
        if dedup_result.is_duplicate:
            return IngestResult(
                slug=dedup_result.existing_slug or "",
                title=title,
                status="duplicate",
            )

        # Generate slug
        slug = self._store.generate_slug(title)

        # Step 3: Split check
        if self._splitter.should_split(text):
            return self._ingest_split(text, slug, title, metadata)
        else:
            return self._ingest_single(text, slug, title, metadata)

    def _ingest_single(
        self, text: str, slug: str, title: str, metadata: dict
    ) -> IngestResult:
        """Ingest a non-split document."""
        # Generate fingerprint
        fp = self._fingerprinter.generate(text, title=title)

        # Build frontmatter
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        frontmatter = {
            "title": title,
            "summary": fp.synopsis,
            "categories": fp.categories,
            "concepts": fp.concepts,
            "fingerprint": fp.fingerprint,
            "created": metadata.get("created", now),
            "updated": now,
        }
        # Merge any existing metadata (from file extraction)
        for key in ("source", "code_structure", "type"):
            if key in metadata:
                frontmatter[key] = metadata[key]

        # Store
        self._store.create(slug, text, frontmatter)

        # Register dedup
        self._dedup.register(slug, text)

        return IngestResult(slug=slug, title=title, status="created")

    def _ingest_split(
        self, text: str, slug: str, title: str, metadata: dict
    ) -> IngestResult:
        """Ingest a document that needs to be split into sections."""
        # Split
        split_result = self._splitter.split(text, slug, metadata)

        if not split_result.is_split and split_result.single:
            # Splitter decided not to split after all
            return self._ingest_single(text, slug, title, metadata)

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        # Fingerprint parent
        parent_doc = split_result.parent
        if not parent_doc:
            return self._ingest_single(text, slug, title, metadata)

        parent_fp = self._fingerprinter.generate(text, title=title)

        # Fingerprint each child and update section_map synopses
        section_map = parent_doc.frontmatter.get("section_map", [])
        for i, child_doc in enumerate(split_result.children):
            child_fp = self._fingerprinter.generate_section_synopsis(
                child_doc.body, child_doc.frontmatter.get("title", "")
            ) if hasattr(self._fingerprinter, "generate_section_synopsis") else ""

            # Update section map synopsis
            if i < len(section_map):
                section_map[i]["synopsis"] = child_fp if isinstance(child_fp, str) else ""

            # Generate full fingerprint for child
            child_full_fp = self._fingerprinter.generate(
                child_doc.body,
                title=child_doc.frontmatter.get("title", ""),
            )
            child_doc.frontmatter.update({
                "summary": child_full_fp.synopsis,
                "categories": child_full_fp.categories,
                "concepts": child_full_fp.concepts,
                "fingerprint": child_full_fp.fingerprint,
                "created": now,
                "updated": now,
            })

        # Update parent frontmatter
        parent_doc.frontmatter.update({
            "title": title,
            "summary": parent_fp.synopsis,
            "categories": parent_fp.categories,
            "concepts": parent_fp.concepts,
            "fingerprint": parent_fp.fingerprint,
            "section_map": section_map,
            "created": metadata.get("created", now),
            "updated": now,
        })

        # Store parent
        self._store.create(parent_doc.slug, parent_doc.body, parent_doc.frontmatter)

        # Store children
        child_slugs = []
        for child_doc in split_result.children:
            self._store.create(child_doc.slug, child_doc.body, child_doc.frontmatter)
            child_slugs.append(child_doc.slug)

        # Register dedup for parent
        self._dedup.register(slug, text)

        return IngestResult(
            slug=slug, title=title, status="split", children=child_slugs
        )
