import os

import pytest

from smartrag.ingest.extractors import (
    ExtractorRegistry,
    UnsupportedFormatError,
    clean_text,
    count_words,
    detect_language,
)


class TestExtractors:
    def test_markdown(self, sample_md):
        reg = ExtractorRegistry()
        result = reg.extract(sample_md)
        assert "Sample" in result.text
        assert result.original_format == "markdown"
        assert result.metadata.get("title") == "Sample Doc"

    def test_txt(self, sample_txt):
        reg = ExtractorRegistry()
        result = reg.extract(sample_txt)
        assert "plain text" in result.text

    def test_python(self, sample_py):
        reg = ExtractorRegistry()
        result = reg.extract(sample_py)
        assert "```python" in result.text or result.original_format == "code"

    def test_unsupported(self, tmp_dir):
        reg = ExtractorRegistry()
        p = os.path.join(tmp_dir, "test.xlsx")
        with open(p, "w") as f:
            f.write("fake")
        with pytest.raises(UnsupportedFormatError):
            reg.extract(p)

    def test_empty_file(self, tmp_dir):
        reg = ExtractorRegistry()
        p = os.path.join(tmp_dir, "empty.md")
        with open(p, "w") as f:
            pass
        result = reg.extract(p)
        assert result.text == "" or result.text.strip() == ""

    def test_count_words(self):
        assert count_words("hello world foo") == 3
        assert count_words("") == 0

    def test_clean_text(self):
        assert "\x00" not in clean_text("hello\x00world")

    def test_detect_language(self):
        assert detect_language(".py") == "python"
        assert detect_language(".js") == "javascript"
        assert detect_language(".ts") == "typescript"
