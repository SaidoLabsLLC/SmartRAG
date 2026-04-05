import os

from smartrag import SmartRAG, SmartRAGConfig


class TestSmartRAGSDK:
    def test_init_empty_dir(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        assert rag.stats["document_count"] == 0

    def test_init_creates_structure(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        assert os.path.isdir(os.path.join(knowledge_dir, "documents"))
        assert os.path.isdir(os.path.join(knowledge_dir, ".smartrag"))

    def test_ingest_text(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        result = rag.ingest_text("JWT tokens are used for authentication.", title="JWT Guide")
        assert result.status == "created"
        assert result.slug == "jwt-guide"

    def test_ingest_file(self, knowledge_dir, sample_md):
        rag = SmartRAG(knowledge_dir)
        result = rag.ingest(sample_md)
        assert result.status == "created"

    def test_query_returns_results(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        rag.ingest_text(
            "REST APIs use HTTP methods for CRUD operations.", title="REST API Guide"
        )
        qr = rag.query("REST API")
        assert len(qr.results) > 0
        assert qr.total_ms < 100

    def test_query_empty_store(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        qr = rag.query("anything")
        assert len(qr.results) == 0

    def test_search(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        rag.ingest_text(
            "Database indexing improves query performance.", title="DB Indexing"
        )
        results = rag.search("database")
        assert len(results) > 0

    def test_get_document(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        r = rag.ingest_text("Content here.", title="My Doc")
        doc = rag.get(r.slug)
        assert doc is not None
        assert doc.title == "My Doc"

    def test_delete_document(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        r = rag.ingest_text("Temporary content.", title="Temp Doc")
        assert rag.delete(r.slug) is True
        assert rag.get(r.slug) is None

    def test_dedup(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        r1 = rag.ingest_text("Same content here.", title="First")
        r2 = rag.ingest_text("Same content here.", title="Second")
        assert r1.status == "created"
        assert r2.status == "duplicate"

    def test_reindex(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        rag.ingest_text("Content A.", title="Doc A")
        rag.ingest_text("Content B.", title="Doc B")
        count = rag.reindex()
        assert count == 2

    def test_stats(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        rag.ingest_text("Some content.", title="Stats Test")
        stats = rag.stats
        assert stats["document_count"] == 1
        assert "categories" in stats

    def test_repr(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        assert "SmartRAG" in repr(rag)

    def test_ingest_directory(self, knowledge_dir, tmp_dir):
        # Create a small directory of files
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir)
        for i in range(3):
            with open(os.path.join(data_dir, f"doc{i}.md"), "w") as f:
                f.write(
                    f"---\ntitle: Doc {i}\n---\n\n"
                    f"Content for document {i} about topic {i}.\n"
                )
        rag = SmartRAG(knowledge_dir)
        results = rag.ingest(data_dir)
        assert len(results) == 3
        assert all(r.status == "created" for r in results)

    def test_section_splitting(self, knowledge_dir, large_md):
        rag = SmartRAG(knowledge_dir)
        result = rag.ingest(large_md)
        assert result.status == "split"
        assert result.children is not None
        assert len(result.children) >= 2

    def test_three_line_quickstart(self, knowledge_dir):
        rag = SmartRAG(knowledge_dir)
        rag.ingest_text(
            "SmartRAG replaces vector databases with tiered retrieval.",
            title="About SmartRAG",
        )
        result = rag.query("What is SmartRAG?")
        assert len(result.results) > 0
