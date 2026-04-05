"""File type extractors for the SmartRAG ingest pipeline.

Each extractor reads a specific file format and returns an ExtractedContent
dataclass with normalized markdown text, extracted metadata, and the original
format identifier.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from smartrag.types import ExtractedContent


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class UnsupportedFormatError(Exception):
    """Raised when the file extension has no registered extractor."""

    def __init__(self, extension: str) -> None:
        self.extension = extension
        super().__init__(f"No extractor registered for extension: {extension}")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".lua": "lua",
}


def detect_language(extension: str) -> str:
    """Map a file extension (with leading dot) to a language identifier."""
    ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    return _LANGUAGE_MAP.get(ext, "text")


def count_words(text: str) -> int:
    """Count whitespace-delimited words in *text*."""
    return len(text.split())


def clean_text(text: str) -> str:
    """Normalize whitespace, strip null bytes, normalize line endings."""
    # Strip null bytes
    text = text.replace("\0", "")
    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines into at most two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Strip leading/trailing whitespace from the whole document
    return text.strip()


def _read_file(file_path: str) -> str:
    """Read a file with UTF-8 encoding."""
    return Path(file_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Individual extractors
# ---------------------------------------------------------------------------

def _extract_markdown(file_path: str) -> ExtractedContent:
    """Extract content from a Markdown file, parsing YAML frontmatter."""
    raw = _read_file(file_path)
    metadata: dict[str, Any] = {}
    body = raw

    # Parse frontmatter between --- delimiters at the start of the file
    fm_match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)", raw, re.DOTALL)
    if fm_match:
        import yaml

        fm_text = fm_match.group(1)
        body = fm_match.group(2)
        try:
            parsed = yaml.safe_load(fm_text)
            if isinstance(parsed, dict):
                metadata = parsed
        except yaml.YAMLError:
            pass  # Treat unparseable frontmatter as part of body
            body = raw

    text = clean_text(body)
    metadata.setdefault("word_count", count_words(text))
    metadata.setdefault("source_file", os.path.basename(file_path))

    return ExtractedContent(text=text, metadata=metadata, original_format="markdown")


def _extract_text(file_path: str) -> ExtractedContent:
    """Extract content from a plain-text file."""
    raw = _read_file(file_path)
    text = clean_text(raw)
    metadata: dict[str, Any] = {
        "source_file": os.path.basename(file_path),
        "word_count": count_words(text),
    }

    # If the first line looks like a heading, extract it as title
    lines = text.split("\n", 1)
    first_line = lines[0].strip()
    if first_line and (
        first_line.startswith("#")
        or first_line.isupper()
        or (len(first_line) < 120 and not first_line.endswith("."))
    ):
        title = first_line.lstrip("# ").strip()
        metadata["title"] = title

    return ExtractedContent(text=text, metadata=metadata, original_format="text")


def _extract_pdf(file_path: str) -> ExtractedContent:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    metadata: dict[str, Any] = {
        "source_file": os.path.basename(file_path),
        "page_count": len(doc),
    }

    # Extract PDF metadata
    pdf_meta = doc.metadata or {}
    if pdf_meta.get("title"):
        metadata["title"] = pdf_meta["title"]
    if pdf_meta.get("author"):
        metadata["author"] = pdf_meta["author"]
    if pdf_meta.get("subject"):
        metadata["subject"] = pdf_meta["subject"]

    pages: list[str] = []
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            pages.append(page_text.strip())

    doc.close()

    # Join pages with markdown horizontal rules as page breaks
    text = clean_text("\n\n---\n\n".join(pages))
    metadata["word_count"] = count_words(text)

    return ExtractedContent(text=text, metadata=metadata, original_format="pdf")


def _extract_html(file_path: str) -> ExtractedContent:
    """Extract text from HTML, converting heading hierarchy to markdown."""
    from bs4 import BeautifulSoup

    raw = _read_file(file_path)
    soup = BeautifulSoup(raw, "html.parser")

    metadata: dict[str, Any] = {"source_file": os.path.basename(file_path)}

    # Extract <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        metadata["title"] = title_tag.string.strip()

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Convert headings to markdown headers
    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            prefix = "#" * level
            heading.replace_with(f"\n\n{prefix} {heading.get_text().strip()}\n\n")

    # Convert <br> to newlines
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Convert <p> to double-newline separated blocks
    for p in soup.find_all("p"):
        p.replace_with(f"\n\n{p.get_text()}\n\n")

    # Convert list items
    for li in soup.find_all("li"):
        li.replace_with(f"\n- {li.get_text().strip()}")

    text = clean_text(soup.get_text())
    metadata["word_count"] = count_words(text)

    return ExtractedContent(text=text, metadata=metadata, original_format="html")


# Mapping of code file extensions to docstring/comment extraction patterns
_DOCSTRING_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    ".py": [
        re.compile(r'^\s*"""(.*?)"""', re.DOTALL),
        re.compile(r"^\s*'''(.*?)'''", re.DOTALL),
        re.compile(r'^(?:\s*#[^\n]*\n)+', re.MULTILINE),
    ],
    ".js": [
        re.compile(r'^\s*/\*\*(.*?)\*/', re.DOTALL),
        re.compile(r'^(?:\s*//[^\n]*\n)+', re.MULTILINE),
    ],
    ".ts": [
        re.compile(r'^\s*/\*\*(.*?)\*/', re.DOTALL),
        re.compile(r'^(?:\s*//[^\n]*\n)+', re.MULTILINE),
    ],
    ".go": [
        re.compile(r'^(?:\s*//[^\n]*\n)+', re.MULTILINE),
    ],
    ".rs": [
        re.compile(r'^(?:\s*///[^\n]*\n)+', re.MULTILINE),
        re.compile(r'^\s*/\*\!(.*?)\*/', re.DOTALL),
    ],
    ".java": [
        re.compile(r'^\s*/\*\*(.*?)\*/', re.DOTALL),
    ],
    ".cpp": [
        re.compile(r'^\s*/\*\*(.*?)\*/', re.DOTALL),
        re.compile(r'^(?:\s*//[^\n]*\n)+', re.MULTILINE),
    ],
    ".c": [
        re.compile(r'^\s*/\*(.*?)\*/', re.DOTALL),
        re.compile(r'^(?:\s*//[^\n]*\n)+', re.MULTILINE),
    ],
    ".h": [
        re.compile(r'^\s*/\*(.*?)\*/', re.DOTALL),
        re.compile(r'^(?:\s*//[^\n]*\n)+', re.MULTILINE),
    ],
}


def _extract_docstring(content: str, extension: str) -> str | None:
    """Try to extract a leading docstring or comment block from source code."""
    patterns = _DOCSTRING_PATTERNS.get(extension, [])
    for pattern in patterns:
        match = pattern.match(content)
        if match:
            # Get the captured group if available, else the full match
            raw = match.group(1) if match.lastindex else match.group(0)
            # Clean up comment markers
            lines = raw.strip().split("\n")
            cleaned = []
            for line in lines:
                line = line.strip()
                # Strip common comment prefixes
                for prefix in ("///", "//!", "//", "*", "#"):
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                cleaned.append(line)
            summary = " ".join(cleaned).strip()
            if summary:
                return summary
    return None


def _extract_code(file_path: str) -> ExtractedContent:
    """Extract content from a source code file, wrapping in a fenced code block."""
    raw = _read_file(file_path)
    ext = Path(file_path).suffix.lower()
    lang = detect_language(ext)

    metadata: dict[str, Any] = {
        "source_file": os.path.basename(file_path),
        "language": lang,
    }

    # Try to extract a docstring/comment summary
    summary = _extract_docstring(raw, ext)
    if summary:
        metadata["summary"] = summary

    text = f"```{lang}\n{clean_text(raw)}\n```"
    metadata["word_count"] = count_words(raw)

    return ExtractedContent(text=text, metadata=metadata, original_format="code")


def _extract_json(file_path: str) -> ExtractedContent:
    """Extract content from a JSON file."""
    raw = _read_file(file_path)
    metadata: dict[str, Any] = {"source_file": os.path.basename(file_path)}

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for key in ("name", "title", "description"):
                if key in data and isinstance(data[key], str):
                    metadata[key] = data[key]
    except json.JSONDecodeError:
        pass

    text = f"```json\n{clean_text(raw)}\n```"
    metadata["word_count"] = count_words(raw)

    return ExtractedContent(text=text, metadata=metadata, original_format="json")


def _extract_yaml(file_path: str) -> ExtractedContent:
    """Extract content from a YAML file."""
    import yaml

    raw = _read_file(file_path)
    metadata: dict[str, Any] = {"source_file": os.path.basename(file_path)}

    try:
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            for key in ("name", "title"):
                if key in data and isinstance(data[key], str):
                    metadata[key] = data[key]
    except yaml.YAMLError:
        pass

    text = f"```yaml\n{clean_text(raw)}\n```"
    metadata["word_count"] = count_words(raw)

    return ExtractedContent(text=text, metadata=metadata, original_format="yaml")


def _extract_toml(file_path: str) -> ExtractedContent:
    """Extract content from a TOML file."""
    raw = _read_file(file_path)
    metadata: dict[str, Any] = {"source_file": os.path.basename(file_path)}

    try:
        import tomllib
    except ModuleNotFoundError:
        # Python < 3.11 fallback
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            tomllib = None  # type: ignore[assignment]

    if tomllib is not None:
        try:
            data = tomllib.loads(raw)
            if isinstance(data, dict):
                for key in ("name", "title", "description"):
                    if key in data and isinstance(data[key], str):
                        metadata[key] = data[key]
        except Exception:
            pass

    text = f"```toml\n{clean_text(raw)}\n```"
    metadata["word_count"] = count_words(raw)

    return ExtractedContent(text=text, metadata=metadata, original_format="toml")


def _detect_csv_delimiter(sample: str) -> str:
    """Auto-detect CSV delimiter using csv.Sniffer, falling back to comma."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return ","


def _build_md_table(header: list[str], data_rows: list[list[str]]) -> str:
    """Build a markdown table from a header and data rows."""
    lines: list[str] = []
    lines.append("| " + " | ".join(cell.strip() for cell in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in data_rows:
        padded = row + [""] * (len(header) - len(row))
        padded = padded[: len(header)]
        lines.append("| " + " | ".join(cell.strip() for cell in padded) + " |")
    return "\n".join(lines)


def _extract_csv(file_path: str) -> ExtractedContent:
    """Extract content from a CSV, converting to a markdown table.

    Improvements:
    - Auto-detects delimiter (comma, tab, semicolon) via csv.Sniffer.
    - Large CSVs (>100 data rows): stores full table content but prepends a
      summary block with row count, column names, and first 5 rows.
    """
    raw = _read_file(file_path)
    metadata: dict[str, Any] = {"source_file": os.path.basename(file_path)}

    # Auto-detect delimiter from the first 8KB
    delimiter = _detect_csv_delimiter(raw[:8192])
    metadata["delimiter"] = delimiter

    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    rows: list[list[str]] = list(reader)

    if not rows:
        return ExtractedContent(
            text="*Empty CSV file*",
            metadata=metadata,
            original_format="csv",
        )

    header = rows[0]
    data_rows = rows[1:]
    total_data_rows = len(data_rows)

    metadata["columns"] = header
    metadata["column_count"] = len(header)
    metadata["row_count"] = total_data_rows

    is_large = total_data_rows > 100
    sections: list[str] = []

    if is_large:
        # Generate summary block for large CSVs
        summary_lines = [
            f"**CSV Summary**: {total_data_rows} rows, "
            f"{len(header)} columns",
            f"**Columns**: {', '.join(header)}",
            "",
            "**Preview (first 5 rows)**:",
            "",
            _build_md_table(header, data_rows[:5]),
            "",
            "---",
            "",
            "**Full data**:",
            "",
        ]
        sections.append("\n".join(summary_lines))

    # Full markdown table
    sections.append(_build_md_table(header, data_rows))

    text = clean_text("\n".join(sections))
    metadata["word_count"] = count_words(text)

    return ExtractedContent(text=text, metadata=metadata, original_format="csv")


def _extract_docx(file_path: str) -> ExtractedContent:
    """Extract content from a DOCX file, preserving heading hierarchy and tables.

    Requires python-docx. Raises UnsupportedFormatError if not installed.
    Images are skipped with a logged warning.
    """
    try:
        import docx
    except ImportError:
        raise UnsupportedFormatError(".docx (python-docx not installed)")

    document = docx.Document(file_path)
    metadata: dict[str, Any] = {"source_file": os.path.basename(file_path)}

    # Extract core properties as metadata
    props = document.core_properties
    if props.title:
        metadata["title"] = props.title
    if props.author:
        metadata["author"] = props.author
    if props.subject:
        metadata["subject"] = props.subject

    sections: list[str] = []
    image_warned = False

    for element in document.element.body:
        tag = element.tag.split("}")[-1]  # Strip namespace

        if tag == "p":
            # It's a paragraph — check if it contains images
            from docx.oxml.ns import qn

            if element.findall(f".//{qn('wp:inline')}") or element.findall(
                f".//{qn('wp:anchor')}"
            ):
                if not image_warned:
                    logger.warning(
                        "DOCX extractor: skipping embedded images in %s",
                        os.path.basename(file_path),
                    )
                    image_warned = True
                continue

            # Find the corresponding Paragraph object
            from docx.text.paragraph import Paragraph

            para = Paragraph(element, document)
            style_name = (para.style.name or "").lower() if para.style else ""
            text = para.text.strip()

            if not text:
                sections.append("")
                continue

            # Convert heading styles to markdown heading markers
            if style_name.startswith("heading"):
                # Extract heading level: "heading 1" -> 1, "heading 2" -> 2
                try:
                    level = int(style_name.split()[-1])
                    level = min(max(level, 1), 6)
                except (ValueError, IndexError):
                    level = 1
                sections.append(f"{'#' * level} {text}")
            else:
                sections.append(text)

        elif tag == "tbl":
            # It's a table — convert to markdown
            from docx.table import Table

            table = Table(element, document)
            if not table.rows:
                continue

            md_rows: list[str] = []
            for row_idx, row in enumerate(table.rows):
                cells = [cell.text.strip() for cell in row.cells]
                md_rows.append("| " + " | ".join(cells) + " |")
                if row_idx == 0:
                    # Add separator after header row
                    md_rows.append(
                        "| " + " | ".join("---" for _ in cells) + " |"
                    )

            sections.append("\n".join(md_rows))

    text = clean_text("\n\n".join(sections))
    metadata["word_count"] = count_words(text)

    return ExtractedContent(text=text, metadata=metadata, original_format="docx")


# ---------------------------------------------------------------------------
# ExtractorRegistry
# ---------------------------------------------------------------------------

# Type alias for extractor functions
ExtractorFn = Callable[[str], ExtractedContent]


class ExtractorRegistry:
    """Maps file extensions to extractor functions and dispatches extraction."""

    def __init__(self) -> None:
        self._registry: dict[str, ExtractorFn] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all built-in extractors."""
        # Markdown
        self.register(".md", _extract_markdown)

        # Plain text
        self.register(".txt", _extract_text)

        # PDF
        self.register(".pdf", _extract_pdf)

        # HTML
        self.register(".html", _extract_html)
        self.register(".htm", _extract_html)

        # Source code
        for ext in (".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h"):
            self.register(ext, _extract_code)

        # Structured data
        self.register(".json", _extract_json)
        self.register(".yaml", _extract_yaml)
        self.register(".yml", _extract_yaml)
        self.register(".toml", _extract_toml)
        self.register(".csv", _extract_csv)

        # Document formats
        self.register(".docx", _extract_docx)

    def register(self, extension: str, extractor: ExtractorFn) -> None:
        """Register an extractor function for a file extension.

        Args:
            extension: File extension including the leading dot (e.g. ".md").
            extractor: Callable that accepts a file path and returns ExtractedContent.
        """
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        self._registry[ext] = extractor

    def extract(self, file_path: str) -> ExtractedContent:
        """Extract content from a file based on its extension.

        Args:
            file_path: Path to the file to extract.

        Returns:
            ExtractedContent with normalized text, metadata, and format info.

        Raises:
            UnsupportedFormatError: If no extractor is registered for the extension.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        extractor = self._registry.get(ext)
        if extractor is None:
            raise UnsupportedFormatError(ext)

        return extractor(str(path))

    @property
    def supported_extensions(self) -> list[str]:
        """Return a sorted list of all registered extensions."""
        return sorted(self._registry.keys())

    def is_supported(self, extension: str) -> bool:
        """Check if an extension has a registered extractor."""
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        return ext in self._registry
