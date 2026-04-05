import os

import pytest

from smartrag.store.markdown import DocumentNotFoundError, MarkdownStore


class TestMarkdownStore:
    def test_init_creates_structure(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        assert os.path.isdir(os.path.join(knowledge_dir, "documents"))
        assert os.path.isdir(os.path.join(knowledge_dir, ".smartrag"))

    def test_create_and_read(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        doc = store.create("test-doc", "Body content.", {"title": "Test Doc"})
        assert doc.slug == "test-doc"
        read_doc = store.read("test-doc")
        assert read_doc.title == "Test Doc"
        assert "Body content" in read_doc.body

    def test_read_frontmatter_only(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        store.create("fm-test", "Long body...", {"title": "FM Test", "categories": ["api"]})
        fm = store.read_frontmatter("fm-test")
        assert fm["title"] == "FM Test"

    def test_update(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        store.create("upd", "Old body.", {"title": "Old"})
        store.update("upd", body="New body.", frontmatter_updates={"title": "New"})
        doc = store.read("upd")
        assert "New body" in doc.body

    def test_delete(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        store.create("del-me", "Body.", {"title": "Delete Me"})
        store.delete("del-me")
        assert not store.exists("del-me")

    def test_not_found(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        with pytest.raises(DocumentNotFoundError):
            store.read("nonexistent")

    def test_exists(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        store.create("exists-test", "Body.", {"title": "Exists"})
        assert store.exists("exists-test")
        assert not store.exists("nope")

    def test_list_all(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        store.create("a", "Body A.", {"title": "Doc A"})
        store.create("b", "Body B.", {"title": "Doc B"})
        all_docs = store.list_all()
        assert len(all_docs) == 2

    def test_generate_slug(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        assert store.generate_slug("My API Guide") == "my-api-guide"
        assert store.generate_slug("Flask REST API: Auth") == "flask-rest-api-auth"

    def test_slug_dedup(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        store.create("test", "Body.", {"title": "Test"})
        slug2 = store.generate_slug("Test")
        assert slug2 == "test-2"

    def test_hooks_fire(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        events = []
        store.register_hook("created", lambda **kw: events.append("created"))
        store.register_hook("deleted", lambda **kw: events.append("deleted"))
        store.create("hook-test", "Body.", {"title": "Hook"})
        store.delete("hook-test")
        assert "created" in events
        assert "deleted" in events

    def test_count(self, knowledge_dir):
        store = MarkdownStore(knowledge_dir)
        store.create("c1", "B.", {"title": "C1"})
        store.create("c2", "B.", {"title": "C2"})
        assert store.count() == 2
