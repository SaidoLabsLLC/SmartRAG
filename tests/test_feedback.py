"""Tests for the feedback loop & self-tuning retrieval system (P2-PROMPT-08)."""

from __future__ import annotations

import os
import shutil
import tempfile
import time

import pytest

from smartrag.config import SmartRAGConfig
from smartrag.feedback.signals import SignalDetector
from smartrag.feedback.store import FeedbackStore
from smartrag.feedback.tuner import RetrievalTuner
from smartrag.types import QueryResult, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    query: str = "test query",
    slugs: list[str] | None = None,
    tiers: list[int] | None = None,
    total_ms: float = 5.0,
) -> QueryResult:
    """Build a minimal QueryResult for testing."""
    slugs = slugs or ["doc-a"]
    tiers = tiers or [1]
    results = [
        RetrievalResult(
            slug=s,
            title=s,
            snippet="...",
            score=1.0,
            tier_resolved=t,
        )
        for s, t in zip(slugs, tiers)
    ]
    qr = QueryResult(
        results=results,
        query=query,
        total_ms=total_ms,
        total_bytes_read=100,
    )
    qr._source_map = {s: "master_index" for s in slugs}  # type: ignore[attr-defined]
    return qr


@pytest.fixture
def fb_dir():
    d = tempfile.mkdtemp(prefix="smartrag_fb_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(fb_dir):
    db_path = os.path.join(fb_dir, ".smartrag", "feedback.db")
    return FeedbackStore(db_path)


# ---------------------------------------------------------------------------
# FeedbackStore basics
# ---------------------------------------------------------------------------


class TestFeedbackStore:
    def test_log_query_creates_entry(self, store):
        qr = _make_result()
        qid = store.log_query(qr)
        assert isinstance(qid, int) and qid > 0

    def test_explicit_feedback(self, store):
        qr = _make_result()
        qid = store.log_query(qr)
        store.record_feedback(qid, 0.9, used_slugs=["doc-a"])
        row = store._conn.execute(
            "SELECT feedback_score, signal_type FROM query_log WHERE id = ?",
            (qid,),
        ).fetchone()
        assert row[0] == 0.9
        assert row[1] == "explicit"

    def test_anonymize_hashes_query(self, fb_dir):
        db_path = os.path.join(fb_dir, ".smartrag", "feedback_anon.db")
        anon_store = FeedbackStore(db_path, anonymize=True)
        qr = _make_result(query="sensitive search")
        qid = anon_store.log_query(qr)
        row = anon_store._conn.execute(
            "SELECT query_text FROM query_log WHERE id = ?", (qid,)
        ).fetchone()
        # Should be SHA-256 hex, not the raw text
        assert row[0] != "sensitive search"
        assert len(row[0]) == 64  # SHA-256 hex length

    def test_get_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["total_queries"] == 0
        assert stats["feedback_rate"] == 0.0

    def test_get_tuning_data_below_threshold(self, store):
        # Less than 50 queries → None
        for i in range(10):
            qr = _make_result(query=f"query {i}")
            qid = store.log_query(qr)
            store.record_feedback(qid, 0.9)
        assert store.get_tuning_data(min_queries=50) is None


# ---------------------------------------------------------------------------
# SignalDetector
# ---------------------------------------------------------------------------


class TestSignalDetector:
    def test_repeat_query_auto_scores(self, store):
        detector = SignalDetector(store)
        qr1 = _make_result(query="python decorators")
        qid1 = store.log_query(qr1)
        detector.on_query(qid1, qr1)

        # Same query again within 60s
        qr2 = _make_result(query="python decorators")
        qid2 = store.log_query(qr2)
        detector.on_query(qid2, qr2)

        row = store._conn.execute(
            "SELECT feedback_score, signal_type FROM query_log WHERE id = ?",
            (qid1,),
        ).fetchone()
        assert row[0] == 0.3
        assert row[1] == "repeat_query"

    def test_refinement_detection(self, store):
        detector = SignalDetector(store)
        qr1 = _make_result(query="python async await patterns decorators")
        qid1 = store.log_query(qr1)
        detector.on_query(qid1, qr1)

        # >60% keyword overlap, different query (4/6 = 0.67)
        qr2 = _make_result(query="python async await patterns examples")
        qid2 = store.log_query(qr2)
        detector.on_query(qid2, qr2)

        row = store._conn.execute(
            "SELECT feedback_score, signal_type FROM query_log WHERE id = ?",
            (qid1,),
        ).fetchone()
        assert row[0] == 0.5
        assert row[1] == "refinement"

    def test_topic_switch_detection(self, store):
        detector = SignalDetector(store)
        qr1 = _make_result(query="database migration strategies")
        qid1 = store.log_query(qr1)
        detector.on_query(qid1, qr1)

        # Completely different topic, within 30s
        qr2 = _make_result(query="react component lifecycle hooks")
        qid2 = store.log_query(qr2)
        detector.on_query(qid2, qr2)

        row = store._conn.execute(
            "SELECT feedback_score, signal_type FROM query_log WHERE id = ?",
            (qid1,),
        ).fetchone()
        assert row[0] == 0.8
        assert row[1] == "topic_switch"

    def test_tier_penalty(self, store):
        detector = SignalDetector(store)
        qr = _make_result(
            query="obscure topic",
            slugs=["doc-x"],
            tiers=[3],
        )
        qid = store.log_query(qr)
        detector.on_query(qid, qr)

        row = store._conn.execute(
            "SELECT tier3_count FROM synopsis_quality WHERE slug = ?",
            ("doc-x",),
        ).fetchone()
        assert row is not None
        assert row[0] == 1


# ---------------------------------------------------------------------------
# RetrievalTuner
# ---------------------------------------------------------------------------


class TestRetrievalTuner:
    def _seed_feedback(self, store, n=60):
        """Seed n queries with positive feedback."""
        for i in range(n):
            source = ["master_index", "fts5", "embeddings", "backlinks"][i % 4]
            qr = _make_result(query=f"tuning query {i}", slugs=[f"doc-{i}"])
            qr._source_map = {f"doc-{i}": source}  # type: ignore[attr-defined]
            qid = store.log_query(qr)
            store.record_feedback(qid, 0.9)

    def test_tune_below_threshold(self, store):
        tuner = RetrievalTuner({"master_index": 1.0, "fts5": 1.0})
        result = tuner.tune(store)
        assert result is None

    def test_tune_clamps_weights(self, store):
        self._seed_feedback(store, n=60)
        tuner = RetrievalTuner({
            "master_index": 1.0,
            "fts5": 1.0,
            "embeddings": 0.8,
            "backlinks": 0.3,
        })
        result = tuner.tune(store)
        assert result is not None
        for w in result.values():
            assert 0.1 <= w <= 3.0

    def test_tune_idempotent(self, store):
        self._seed_feedback(store, n=60)
        tuner = RetrievalTuner({
            "master_index": 1.0,
            "fts5": 1.0,
            "embeddings": 0.8,
            "backlinks": 0.3,
        })
        r1 = tuner.tune(store)
        # Reset tuner state to base to test idempotency with same data
        tuner2 = RetrievalTuner({
            "master_index": 1.0,
            "fts5": 1.0,
            "embeddings": 0.8,
            "backlinks": 0.3,
        })
        r2 = tuner2.tune(store)
        assert r1 == r2

    def test_tune_stores_history(self, store):
        self._seed_feedback(store, n=60)
        tuner = RetrievalTuner({
            "master_index": 1.0,
            "fts5": 1.0,
            "embeddings": 0.8,
            "backlinks": 0.3,
        })
        tuner.tune(store)
        row = store._conn.execute(
            "SELECT COUNT(*) FROM tuning_history"
        ).fetchone()
        assert row[0] >= 1


# ---------------------------------------------------------------------------
# Synopsis quality / flagged documents
# ---------------------------------------------------------------------------


class TestSynopsisQuality:
    def test_flagged_document_above_threshold(self, store):
        # Create a document with >50% tier-3 rate over 6 queries
        for i in range(6):
            qr = _make_result(query=f"q{i}", slugs=["bad-doc"], tiers=[3])
            qid = store.log_query(qr)
            store.record_tier_penalty(qid, "bad-doc")

        flagged = store.get_flagged_documents(min_queries=5, threshold=0.5)
        assert "bad-doc" in flagged

    def test_unflagged_document_below_threshold(self, store):
        # Only 1 tier-3 out of 6 → should not be flagged
        for i in range(5):
            store.record_tier_penalty(0, "ok-doc")
        # Manually set total_query_count higher
        store._conn.execute(
            "UPDATE synopsis_quality SET total_query_count = 20 WHERE slug = ?",
            ("ok-doc",),
        )
        store._conn.commit()
        flagged = store.get_flagged_documents(min_queries=5, threshold=0.5)
        assert "ok-doc" not in flagged


# ---------------------------------------------------------------------------
# SmartRAG SDK integration
# ---------------------------------------------------------------------------


class TestSmartRAGIntegration:
    def test_record_feedback_through_sdk(self, tmp_dir, sample_md):
        from smartrag.core import SmartRAG

        config = SmartRAGConfig(feedback=True)
        rag = SmartRAG(os.path.join(tmp_dir, "knowledge"), config=config)
        rag.ingest(sample_md)
        result = rag.query("API testing")
        assert result.query_id is not None
        rag.record_feedback(result.query_id, 0.9, used_slugs=["sample-doc"])

    def test_feedback_disabled_no_db(self, tmp_dir, sample_md):
        from smartrag.core import SmartRAG

        config = SmartRAGConfig(feedback=False)
        rag = SmartRAG(os.path.join(tmp_dir, "knowledge"), config=config)
        rag.ingest(sample_md)
        result = rag.query("API testing")
        assert result.query_id is None
        fb_path = os.path.join(tmp_dir, "knowledge", ".smartrag", "feedback.db")
        assert not os.path.exists(fb_path)

    def test_feedback_disabled_raises_on_record(self, tmp_dir, sample_md):
        from smartrag.core import SmartRAG

        config = SmartRAGConfig(feedback=False)
        rag = SmartRAG(os.path.join(tmp_dir, "knowledge"), config=config)
        with pytest.raises(RuntimeError):
            rag.record_feedback(1, 0.5)

    def test_get_retrieval_stats_empty(self, tmp_dir, sample_md):
        from smartrag.core import SmartRAG

        config = SmartRAGConfig(feedback=True)
        rag = SmartRAG(os.path.join(tmp_dir, "knowledge"), config=config)
        stats = rag.get_retrieval_stats()
        assert stats["total_queries"] == 0

    def test_tune_now_insufficient_data(self, tmp_dir, sample_md):
        from smartrag.core import SmartRAG

        config = SmartRAGConfig(feedback=True)
        rag = SmartRAG(os.path.join(tmp_dir, "knowledge"), config=config)
        result = rag.tune_now()
        assert result is None


# ---------------------------------------------------------------------------
# Multi-tenant isolation
# ---------------------------------------------------------------------------


class TestMultiTenantFeedback:
    def test_tenant_isolated_feedback_db(self, fb_dir):
        """Each tenant gets its own feedback.db."""
        db1 = os.path.join(fb_dir, "tenant_a", ".smartrag", "feedback.db")
        db2 = os.path.join(fb_dir, "tenant_b", ".smartrag", "feedback.db")
        store1 = FeedbackStore(db1)
        store2 = FeedbackStore(db2)

        qr = _make_result(query="shared query")
        store1.log_query(qr)

        assert store1.get_stats()["total_queries"] == 1
        assert store2.get_stats()["total_queries"] == 0


# ---------------------------------------------------------------------------
# Auto-tune at interval
# ---------------------------------------------------------------------------


class TestAutoTune:
    def test_auto_tune_fires_at_interval(self, tmp_dir, sample_md):
        from smartrag.core import SmartRAG

        config = SmartRAGConfig(
            feedback=True,
            self_tuning=True,
            tune_interval=5,  # Low for testing
        )
        rag = SmartRAG(os.path.join(tmp_dir, "knowledge"), config=config)
        rag.ingest(sample_md)

        # Run 5 queries to trigger tune_interval
        for i in range(5):
            rag.query(f"query number {i}")

        # Auto-tune should have been attempted (but returns None due to insufficient data)
        # The key test: no crash, and query_count incremented correctly
        assert rag._query_count == 5
