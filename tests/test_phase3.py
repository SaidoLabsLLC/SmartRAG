"""Phase 3 tests — multi-tenant, webhooks, schemas, export."""

import json
import os
import tempfile
import shutil

import pytest


# ------------------------------------------------------------------ #
# Tenant Manager
# ------------------------------------------------------------------ #


class TestTenantManager:
    def test_create_and_get_instance(self, tmp_dir):
        from smartrag.api.tenants import TenantManager

        tm = TenantManager(tmp_dir)
        rag = tm.get_instance("tenant-a")
        assert rag is not None
        # Same tenant returns same instance
        rag2 = tm.get_instance("tenant-a")
        assert rag is rag2

    def test_tenant_isolation(self, tmp_dir):
        from smartrag.api.tenants import TenantManager

        tm = TenantManager(tmp_dir)
        rag_a = tm.get_instance("tenant-a")
        rag_b = tm.get_instance("tenant-b")
        rag_a.ingest_text("Secret doc for A only.", title="A Secret")
        # Tenant B should not see tenant A's docs
        assert rag_b.stats["document_count"] == 0
        assert rag_a.stats["document_count"] == 1

    def test_list_tenants(self, tmp_dir):
        from smartrag.api.tenants import TenantManager

        tm = TenantManager(tmp_dir)
        tm.get_instance("alpha")
        tm.get_instance("beta")
        tenants = tm.list_tenants()
        assert "alpha" in tenants
        assert "beta" in tenants


# ------------------------------------------------------------------ #
# Webhook Manager
# ------------------------------------------------------------------ #


class TestWebhookManager:
    def test_register_and_list(self, tmp_dir):
        from smartrag.api.webhooks import WebhookManager

        wm = WebhookManager(tmp_dir)
        wh_id = wm.register(
            tenant_id="test",
            url="https://example.com/webhook",
            events=["document.created"],
            secret="my-secret",
        )
        assert wh_id is not None
        hooks = wm.list_webhooks("test")
        assert len(hooks) >= 1
        assert any(h.get("url") == "https://example.com/webhook" for h in hooks)

    def test_remove_webhook(self, tmp_dir):
        from smartrag.api.webhooks import WebhookManager

        wm = WebhookManager(tmp_dir)
        wh_id = wm.register("test", "https://example.com/wh", ["document.created"], "secret")
        wm.remove("test", wh_id)
        hooks = wm.list_webhooks("test")
        assert not any(h.get("id") == wh_id for h in hooks)

    def test_ssrf_blocks_private_url(self, tmp_dir):
        from smartrag.api.webhooks import WebhookManager

        wm = WebhookManager(tmp_dir)
        with pytest.raises((ValueError, Exception)):
            wm.register("test", "http://192.168.1.1/hook", ["document.created"], "secret")


# ------------------------------------------------------------------ #
# Schema Manager
# ------------------------------------------------------------------ #


class TestSchemaManager:
    def test_define_and_list(self, tmp_dir):
        from smartrag.api.schemas import SchemaManager

        sm = SchemaManager(tmp_dir)
        sm.define_field("test", "priority", "int", required=False, default=0, description="Task priority")
        fields = sm.list_fields("test")
        assert len(fields) >= 1
        assert any(f.get("field_name") == "priority" for f in fields)

    def test_remove_field(self, tmp_dir):
        from smartrag.api.schemas import SchemaManager

        sm = SchemaManager(tmp_dir)
        sm.define_field("test", "temp", "string")
        sm.remove_field("test", "temp")
        fields = sm.list_fields("test")
        assert not any(f.get("field_name") == "temp" for f in fields)

    def test_validate_correct_type(self, tmp_dir):
        from smartrag.api.schemas import SchemaManager

        sm = SchemaManager(tmp_dir)
        sm.define_field("test", "count", "int", required=True)
        # Valid
        sm.validate_frontmatter("test", {"count": 5, "title": "Test"})

    def test_validate_wrong_type(self, tmp_dir):
        from smartrag.api.schemas import SchemaManager

        sm = SchemaManager(tmp_dir)
        sm.define_field("test", "count", "int", required=True)
        with pytest.raises(ValueError):
            sm.validate_frontmatter("test", {"count": "not-a-number", "title": "Test"})

    def test_validate_missing_required(self, tmp_dir):
        from smartrag.api.schemas import SchemaManager

        sm = SchemaManager(tmp_dir)
        sm.define_field("test", "priority", "int", required=True)
        with pytest.raises(ValueError):
            sm.validate_frontmatter("test", {"title": "Test"})


# ------------------------------------------------------------------ #
# Knowledge Export / Import
# ------------------------------------------------------------------ #


class TestKnowledgeExport:
    def test_export_creates_bundle(self, knowledge_dir):
        from smartrag import SmartRAG
        from smartrag.export import KnowledgeExporter

        rag = SmartRAG(knowledge_dir)
        rag.ingest_text("Export test content.", title="Export Doc")

        output = os.path.join(os.path.dirname(knowledge_dir), "test.smartrag")
        exporter = KnowledgeExporter(rag)
        path = exporter.export_bundle(output)
        assert os.path.isfile(path)
        assert path.endswith(".smartrag")

    def test_export_import_roundtrip(self, knowledge_dir, tmp_dir):
        from smartrag import SmartRAG
        from smartrag.export import KnowledgeExporter

        # Create and export
        rag = SmartRAG(knowledge_dir)
        rag.ingest_text("Roundtrip test content about databases.", title="Roundtrip Doc")
        original_count = rag.stats["document_count"]

        bundle_path = os.path.join(tmp_dir, "roundtrip.smartrag")
        exporter = KnowledgeExporter(rag)
        exporter.export_bundle(bundle_path)

        # Import into new location
        import_dir = os.path.join(tmp_dir, "imported")
        imported_rag = KnowledgeExporter.import_bundle(bundle_path, import_dir)
        assert imported_rag.stats["document_count"] == original_count

    def test_bundle_contains_manifest(self, knowledge_dir, tmp_dir):
        import zipfile
        from smartrag import SmartRAG
        from smartrag.export import KnowledgeExporter

        rag = SmartRAG(knowledge_dir)
        rag.ingest_text("Manifest test.", title="Manifest Doc")

        bundle_path = os.path.join(tmp_dir, "manifest.smartrag")
        KnowledgeExporter(rag).export_bundle(bundle_path)

        with zipfile.ZipFile(bundle_path, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["document_count"] == 1
            assert "smartrag_version" in manifest
