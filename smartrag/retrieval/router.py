"""Tiered Retrieval Router — the core SmartRAG innovation."""

from __future__ import annotations

import logging
import time

from smartrag.config import SmartRAGConfig
from smartrag.retrieval.scorer import (
    classify_query,
    keyword_overlap_score,
    rrf_merge,
    score_index_entry,
    tokenize_query,
)
from smartrag.types import QueryResult, RetrievalResult

logger = logging.getLogger(__name__)


class TieredRetriever:
    """Routes queries through progressively narrower metadata funnels."""

    def __init__(self, store, master_index, fts_index, backlink_manager, config: SmartRAGConfig, embedding_index=None):
        self._store = store
        self._master_index = master_index
        self._fts = fts_index
        self._backlinks = backlink_manager
        self._config = config
        self._embeddings = embedding_index  # Optional: EmbeddingIndex for semantic search

    def update_weights(self, weights: dict[str, float]) -> None:
        """Update the ranking weights used by RRF merge."""
        self._config.ranking_weights = weights

    def retrieve(self, query: str, top_k: int = 10) -> QueryResult:
        start = time.perf_counter()
        total_bytes = 0
        keywords = tokenize_query(query)
        query_type = classify_query(query)

        if not keywords:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return QueryResult(results=[], query=query, total_ms=elapsed_ms, total_bytes_read=0)

        # --- Step 2: Tier 0 — Master Index scan (<1ms) ---
        mi_results = self._master_index.search(keywords, top_k=top_k)
        mi_ranked = [(entry.slug, score) for entry, score in mi_results]

        # Also use FTS5 if available and store has documents
        fts_ranked = []
        if self._config.fts5:
            try:
                fts_results = self._fts.search_fts(query, top_k=top_k)
                fts_ranked = [(r.slug, r.score) for r in fts_results]
            except Exception as e:
                logger.warning(f"FTS5 search failed: {e}")

        # Semantic embedding search (optional, Phase 2)
        emb_ranked = []
        if self._embeddings and self._config.embeddings:
            try:
                if self._embeddings.is_available():
                    # Use embeddings when FTS5 confidence is low or always in hybrid mode
                    best_fts_score = fts_ranked[0][1] if fts_ranked else 0.0
                    if best_fts_score < 5.0 or len(fts_ranked) < 3:
                        emb_results = self._embeddings.search_semantic(query, top_k=top_k)
                        emb_ranked = list(emb_results)
            except Exception as e:
                logger.warning(f"Embedding search failed: {e}")

        # Merge via RRF (hybrid ranking with configurable weights)
        sources = [mi_ranked]
        source_names = ["master_index"]
        if fts_ranked:
            sources.append(fts_ranked)
            source_names.append("fts5")
        if emb_ranked:
            sources.append(emb_ranked)
            source_names.append("embeddings")
        merged = rrf_merge(
            sources,
            weights=self._config.ranking_weights,
            source_names=source_names,
        )

        # Build source_map: which source contributed each slug's top rank
        source_map: dict[str, str] = {}
        for idx, result_list in enumerate(sources):
            name = source_names[idx]
            for slug, _score in result_list:
                if slug not in source_map:
                    source_map[slug] = name

        # Select top-K candidates
        candidates = [slug for slug, _score in merged[:top_k]]
        candidate_scores = {slug: score for slug, score in merged}

        if not candidates:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return QueryResult(results=[], query=query, total_ms=elapsed_ms, total_bytes_read=0)

        # --- Step 3: Tier 1 — Frontmatter evaluation ---
        results: list[RetrievalResult] = []
        tier1_candidates = candidates[:5]

        for slug in tier1_candidates:
            try:
                fm = self._store.read_frontmatter(slug)
                fm_bytes = len(str(fm).encode("utf-8"))
                total_bytes += fm_bytes

                title = fm.get("title", slug)
                synopsis = fm.get("summary", "")
                categories = fm.get("categories", [])
                concepts = fm.get("concepts", [])
                has_children = bool(fm.get("children"))
                section_map = fm.get("section_map")

                # Score frontmatter
                concept_score = keyword_overlap_score(keywords, concepts)
                tier0_score = candidate_scores.get(slug, 0.0)
                tier1_score = tier0_score + concept_score * 2.0

                # For structured queries, frontmatter may be enough
                if query_type == "structured":
                    results.append(RetrievalResult(
                        slug=slug,
                        title=title,
                        snippet=synopsis,
                        score=tier1_score,
                        tier_resolved=1,
                        categories=categories,
                        source_file=f"documents/{slug}.md",
                    ))
                    continue

                # --- Step 4: Tier 2 — Section map traversal ---
                if has_children and section_map:
                    best_child = None
                    best_child_score = 0.0
                    for section in section_map:
                        sec_synopsis = section.get("synopsis", "")
                        sec_slug = section.get("slug", "")
                        sec_title = section.get("title", "")
                        sec_score = keyword_overlap_score(
                            keywords,
                            tokenize_query(f"{sec_title} {sec_synopsis}"),
                        )
                        if sec_score > best_child_score:
                            best_child_score = sec_score
                            best_child = section

                    if best_child:
                        child_slug = best_child.get("slug", "")
                        try:
                            child_doc = self._store.read(child_slug)
                            child_bytes = len(child_doc.body.encode("utf-8"))
                            total_bytes += child_bytes
                            snippet = child_doc.body[:300].strip()
                            results.append(RetrievalResult(
                                slug=child_slug,
                                title=best_child.get("title", child_slug),
                                snippet=snippet,
                                score=tier1_score + best_child_score,
                                tier_resolved=2,
                                categories=categories,
                                source_file=f"documents/{child_slug}.md",
                            ))
                            continue
                        except Exception:
                            pass  # Fall through to Tier 3

                # --- Step 5: Tier 3 — Full content ---
                try:
                    doc = self._store.read(slug)
                    doc_bytes = len(doc.body.encode("utf-8"))
                    total_bytes += doc_bytes
                    snippet = doc.body[:300].strip()
                    results.append(RetrievalResult(
                        slug=slug,
                        title=title,
                        snippet=snippet,
                        score=tier1_score,
                        tier_resolved=3,
                        categories=categories,
                        source_file=f"documents/{slug}.md",
                    ))
                except Exception as e:
                    logger.warning(f"Failed to read {slug}: {e}")

            except Exception as e:
                logger.warning(f"Failed to read frontmatter for {slug}: {e}")

        # Add remaining candidates at lower scores (Tier 0 only)
        seen_slugs = {r.slug for r in results}
        for slug in candidates[5:]:
            if slug not in seen_slugs:
                entry = self._master_index.get(slug)
                if entry:
                    results.append(RetrievalResult(
                        slug=slug,
                        title=entry.title,
                        snippet=entry.synopsis,
                        score=candidate_scores.get(slug, 0.0) * 0.5,
                        tier_resolved=0,
                        categories=entry.categories,
                        source_file=f"documents/{slug}.md",
                    ))

        # --- Step 6: Backlink expansion ---
        top_result_slugs = [r.slug for r in sorted(results, key=lambda r: r.score, reverse=True)[:3]]
        for slug in top_result_slugs:
            try:
                backlinked = self._backlinks.get_backlinks(slug)
                for bl_slug in backlinked[:2]:
                    if bl_slug not in seen_slugs and bl_slug not in {r.slug for r in results}:
                        try:
                            bl_fm = self._store.read_frontmatter(bl_slug)
                            total_bytes += len(str(bl_fm).encode("utf-8"))
                            results.append(RetrievalResult(
                                slug=bl_slug,
                                title=bl_fm.get("title", bl_slug),
                                snippet=bl_fm.get("summary", ""),
                                score=candidate_scores.get(slug, 0.1) * 0.5,
                                tier_resolved=1,
                                categories=bl_fm.get("categories", []),
                                source_file=f"documents/{bl_slug}.md",
                            ))
                        except Exception:
                            pass
            except Exception:
                pass

        # Track backlink sources
        for r in results:
            if r.slug not in source_map:
                source_map[r.slug] = "backlinks"

        # Sort by score descending, limit to top_k
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]

        elapsed_ms = (time.perf_counter() - start) * 1000
        qr = QueryResult(
            results=results,
            query=query,
            total_ms=elapsed_ms,
            total_bytes_read=total_bytes,
        )
        # Attach source map for feedback system
        qr._source_map = source_map  # type: ignore[attr-defined]
        return qr
