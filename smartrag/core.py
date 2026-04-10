"""SmartRAG core class — public SDK API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from smartrag.config import SmartRAGConfig
from smartrag.ingest.pipeline import IngestPipeline
from smartrag.retrieval.fts import FTSIndex
from smartrag.retrieval.router import TieredRetriever
from smartrag.store.backlinks import BacklinkManager
from smartrag.store.markdown import MarkdownStore
from smartrag.store.master_index import MasterIndex
from smartrag.types import (
    Document,
    IngestResult,
    QueryResult,
    SearchResult,
)

logger = logging.getLogger(__name__)


class SmartRAG:
    """SmartRAG — drop-in retrieval engine that replaces vector-based RAG."""

    def __init__(self, path: str, config: SmartRAGConfig | None = None):
        self._path = str(Path(path).resolve())
        self._config = config or SmartRAGConfig()

        # Load config from file if exists, merge
        config_file = os.path.join(self._path, ".smartrag", "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                if config is None:
                    for key, value in file_config.items():
                        if hasattr(self._config, key):
                            setattr(self._config, key, value)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")

        # 1. Create store (creates directory structure)
        self._store = MarkdownStore(self._path)

        # 2. Create components
        self._backlinks = BacklinkManager(
            os.path.join(self._path, "backlinks.json")
        )
        self._master_index = MasterIndex(
            os.path.join(self._path, "_index.md"),
            cache_size=self._config.tier0_cache_size,
        )
        self._fts = FTSIndex(
            os.path.join(self._path, ".smartrag", "wiki.db")
        )

        # 3. Register hooks
        self._store.register_hook("created", self._backlinks.on_document_change)
        self._store.register_hook("updated", self._backlinks.on_document_change)
        self._store.register_hook("deleted", self._backlinks.on_document_delete)

        self._store.register_hook("created", self._master_index.on_document_upsert)
        self._store.register_hook("updated", self._master_index.on_document_upsert)
        self._store.register_hook("deleted", self._master_index.on_document_delete)

        self._store.register_hook("created", self._fts.on_document_upsert)
        self._store.register_hook("updated", self._fts.on_document_upsert)
        self._store.register_hook("deleted", self._fts.on_document_delete)

        # 4. Optional embedding index (Phase 2)
        self._embeddings = None
        if self._config.embeddings:
            try:
                from smartrag.retrieval.embeddings import EmbeddingIndex
                self._embeddings = EmbeddingIndex(
                    os.path.join(self._path, ".smartrag", "wiki.db")
                )
                if self._embeddings.is_available():
                    self._store.register_hook("created", self._embeddings.on_document_upsert)
                    self._store.register_hook("updated", self._embeddings.on_document_upsert)
                    self._store.register_hook("deleted", self._embeddings.on_document_delete)
                else:
                    logger.info("Embeddings requested but sentence-transformers not installed")
                    self._embeddings = None
            except Exception as e:
                logger.warning(f"Failed to initialize embedding index: {e}")
                self._embeddings = None

        # 5. Create retriever (with optional embeddings)
        self._retriever = TieredRetriever(
            self._store,
            self._master_index,
            self._fts,
            self._backlinks,
            self._config,
            embedding_index=self._embeddings,
        )

        # 6. Feedback system (optional)
        self._feedback = None
        self._signal_detector = None
        self._tuner = None
        self._query_count = 0
        if self._config.feedback:
            from smartrag.feedback.store import FeedbackStore
            from smartrag.feedback.signals import SignalDetector
            from smartrag.feedback.tuner import RetrievalTuner

            fb_path = os.path.join(self._path, ".smartrag", "feedback.db")
            self._feedback = FeedbackStore(fb_path, anonymize=self._config.feedback_anonymize)
            self._signal_detector = SignalDetector(self._feedback)
            self._tuner = RetrievalTuner(self._config.ranking_weights)

        # 7. Create ingest pipeline
        self._pipeline = IngestPipeline(self._store, self._config)

        # 6. Index state file for incremental reindex
        self._index_state_path = os.path.join(
            self._path, ".smartrag", "index_state.json"
        )

    # ------------------------------------------------------------------
    # Index state helpers
    # ------------------------------------------------------------------

    def _load_index_state(self) -> dict[str, dict[str, str]]:
        """Load the index state file: {slug: {hash, indexed_at}}."""
        if not os.path.exists(self._index_state_path):
            return {}
        try:
            with open(self._index_state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load index state: {e}")
            return {}

    def _save_index_state(self, state: dict[str, dict[str, str]]) -> None:
        """Persist the index state file."""
        os.makedirs(os.path.dirname(self._index_state_path), exist_ok=True)
        with open(self._index_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @staticmethod
    def _content_hash(doc: Document) -> str:
        """Compute SHA-256 hash of a document's body and frontmatter."""
        payload = json.dumps(doc.frontmatter, sort_keys=True) + "\n" + doc.body
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def ingest(self, path: str) -> IngestResult | list[IngestResult]:
        """Ingest a file or directory."""
        p = Path(path)
        if p.is_dir():
            return self._pipeline.ingest_directory(str(p))
        return self._pipeline.ingest_file(str(p))

    def ingest_url(self, url: str) -> IngestResult:
        """Ingest content from a URL."""
        return self._pipeline.ingest_url(url)

    def ingest_text(
        self, text: str, title: str, metadata: dict | None = None
    ) -> IngestResult:
        """Ingest raw text directly."""
        return self._pipeline.ingest_text(text, title, metadata)

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        """Query the knowledge store. Returns ranked, cited results."""
        k = top_k or self._config.max_results
        result = self._retriever.retrieve(question, top_k=k)

        if self._feedback:
            query_id = self._feedback.log_query(result, anonymize=self._config.feedback_anonymize)
            result.query_id = query_id
            self._signal_detector.on_query(query_id, result)
            self._query_count += 1
            if self._config.self_tuning and self._query_count % self._config.tune_interval == 0:
                new_weights = self._tuner.tune(self._feedback)
                if new_weights:
                    self._config.ranking_weights = new_weights
                    self._retriever.update_weights(new_weights)
                    # Fire webhook if available
                    webhook_mgr = getattr(self, "_webhook_manager", None)
                    tenant_id = getattr(self, "_tenant_id", "__global__")
                    if webhook_mgr:
                        webhook_mgr.fire(tenant_id, "weights_updated", {"weights": new_weights})

        return result

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search with optional filters. Lower-level than query()."""
        k = top_k or self._config.max_results
        if filters:
            fts_results = self._fts.search_structured(filters, top_k=k)
        else:
            fts_results = self._fts.search_fts(query, top_k=k)
        return [
            SearchResult(
                slug=r.slug,
                title=r.title,
                summary=r.synopsis,
                score=r.score,
                categories=[],
            )
            for r in fts_results
        ]

    def get(self, slug: str) -> Document | None:
        """Get a specific document by slug."""
        try:
            return self._store.read(slug)
        except Exception:
            return None

    def delete(self, slug: str) -> bool:
        """Delete a document."""
        try:
            self._store.delete(slug)
            return True
        except Exception:
            return False

    def reindex(self, incremental: bool = True) -> int:
        """Rebuild indexes from markdown files. Returns count of processed docs.

        Args:
            incremental: When True (default), only re-process expensive
                operations (backlink extraction) for documents whose content
                hash changed since the last index run. Master index and FTS
                are always fully rebuilt for consistency.
                When False, perform a full rebuild of everything.
        """
        from smartrag.types import IndexEntry

        all_docs = self._store.list_all()
        prev_state = self._load_index_state() if incremental else {}
        new_state: dict[str, dict[str, str]] = {}

        entries: list[IndexEntry] = []
        fts_articles: list[tuple[str, dict, str]] = []
        changed_count = 0

        for slug, _title, _summary in all_docs:
            try:
                doc = self._store.read(slug)
                doc_hash = self._content_hash(doc)

                # Determine whether this document changed
                is_changed = True
                if incremental and slug in prev_state:
                    if prev_state[slug].get("hash") == doc_hash:
                        is_changed = False

                # Always build index entry (needed for full index rebuild)
                entry = IndexEntry(
                    slug=doc.slug,
                    title=doc.title,
                    categories=doc.frontmatter.get("categories", []),
                    synopsis=doc.frontmatter.get("summary", ""),
                    fingerprint=doc.frontmatter.get("fingerprint", []),
                    has_children=doc.has_children,
                    parent=doc.frontmatter.get("parent"),
                )
                entries.append(entry)
                fts_articles.append((doc.slug, doc.frontmatter, doc.body))

                # Only run expensive backlink extraction for changed docs
                if is_changed:
                    self._backlinks.update_links(doc.slug, doc.body)
                    changed_count += 1
                    new_state[slug] = {
                        "hash": doc_hash,
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    # Preserve previous state for unchanged documents
                    new_state[slug] = prev_state[slug]

            except Exception as e:
                logger.warning(f"Skipping {slug} during reindex: {e}")

        skipped = len(entries) - changed_count
        if incremental and skipped > 0:
            logger.info(
                f"Incremental reindex: {skipped} unchanged, "
                f"{changed_count} updated"
            )

        self._master_index.rebuild(entries)
        self._fts.rebuild(fts_articles)
        self._save_index_state(new_state)
        return changed_count if incremental else len(entries)

    def record_feedback(
        self, query_id: int, score: float, used_slugs: list[str] | None = None
    ) -> None:
        """Record explicit feedback for a query result."""
        if not self._feedback:
            raise RuntimeError("Feedback is disabled in this SmartRAG instance.")
        self._feedback.record_feedback(query_id, score, used_slugs)

    def get_retrieval_stats(self) -> dict:
        """Return feedback and retrieval statistics."""
        if not self._feedback:
            return {}
        return self._feedback.get_stats()

    def get_flagged_documents(self) -> list[str]:
        """Return slugs of documents needing synopsis regeneration."""
        if not self._feedback:
            return []
        return self._feedback.get_flagged_documents()

    def tune_now(self) -> dict[str, float] | None:
        """Trigger manual weight tuning. Returns new weights or None."""
        if not self._feedback or not self._tuner:
            return None
        new_weights = self._tuner.tune(self._feedback)
        if new_weights:
            self._config.ranking_weights = new_weights
            self._retriever.update_weights(new_weights)
        return new_weights

    @property
    def stats(self) -> dict:
        """Return knowledge store statistics."""
        db_path = os.path.join(self._path, ".smartrag", "wiki.db")
        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        return {
            "document_count": self._master_index.count(),
            "index_size_bytes": db_size,
            "categories": list(
                {
                    cat
                    for e in self._master_index.all_entries()
                    for cat in e.categories
                }
            ),
        }

    def __repr__(self) -> str:
        return f"SmartRAG('{self._path}', docs={self.stats['document_count']})"
