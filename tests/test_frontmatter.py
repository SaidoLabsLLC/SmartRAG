import pytest

from smartrag.store.frontmatter import parse_frontmatter, update_frontmatter, write_frontmatter
from smartrag.types import DocumentFrontmatter


class TestFrontmatter:
    def test_parse(self):
        content = "---\ntitle: Test\ncategories: [a, b]\n---\n\nBody here."
        fm, body = parse_frontmatter(content)
        assert fm["title"] == "Test"
        assert body.strip() == "Body here."

    def test_parse_no_frontmatter(self):
        fm, body = parse_frontmatter("Just body content.")
        assert fm == {}
        assert "Just body" in body

    def test_write_roundtrip(self):
        metadata = {"title": "Test", "categories": ["a", "b"]}
        body = "Body content."
        output = write_frontmatter(metadata, body)
        fm2, body2 = parse_frontmatter(output)
        assert fm2["title"] == "Test"
        assert body2.strip() == "Body content."

    def test_update(self):
        content = "---\ntitle: Old\n---\n\nBody."
        updated = update_frontmatter(content, {"title": "New"})
        fm, _ = parse_frontmatter(updated)
        assert fm["title"] == "New"

    def test_malformed_yaml(self):
        content = "---\n: invalid: yaml: {{{\n---\n\nBody."
        fm, body = parse_frontmatter(content)
        assert fm == {} or isinstance(fm, dict)

    def test_document_frontmatter_from_dict(self):
        fm = DocumentFrontmatter.from_dict({"title": "Test", "categories": ["a"]})
        assert fm.title == "Test"

    def test_document_frontmatter_missing_title(self):
        with pytest.raises(ValueError):
            DocumentFrontmatter.from_dict({})

    def test_document_frontmatter_to_dict(self):
        fm = DocumentFrontmatter(title="Test")
        d = fm.to_dict()
        assert d["title"] == "Test"
        assert "parent" not in d  # None values stripped
