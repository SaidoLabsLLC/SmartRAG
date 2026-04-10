"""Public type definitions for SmartRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryResult:
    """Result from a SmartRAG query."""

    results: list[RetrievalResult]
    query: str
    total_ms: float
    total_bytes_read: int
    query_id: int | None = None


@dataclass
class RetrievalResult:
    """A single retrieved document/section from a query."""

    slug: str
    title: str
    snippet: str
    score: float
    tier_resolved: int  # 0-3
    categories: list[str] = field(default_factory=list)
    source_file: str = ""


@dataclass
class SearchResult:
    """A single result from a search operation."""

    slug: str
    title: str
    summary: str
    score: float
    categories: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    """Result from ingesting a document."""

    slug: str
    title: str
    status: str  # "created" | "updated" | "duplicate" | "failed" | "split"
    children: list[str] | None = None
    error: str | None = None


@dataclass
class Document:
    """A complete document with frontmatter and body."""

    slug: str
    title: str
    body: str
    frontmatter: dict[str, Any]
    word_count: int
    has_children: bool = False


@dataclass
class DocumentFrontmatter:
    """Structured frontmatter metadata for a document."""

    title: str
    summary: str = ""
    categories: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    fingerprint: list[str] = field(default_factory=list)
    backlinks: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    parent: str | None = None
    children: list[str] | None = None
    section_map: list[dict[str, str]] | None = None
    split_from: str | None = None
    section_index: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DocumentFrontmatter:
        """Create DocumentFrontmatter from a dict, validating and setting defaults."""
        if "title" not in d or not d["title"]:
            raise ValueError("title is required in frontmatter")
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, stripping None values for clean YAML output."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None and value != "" and value != []:
                result[key] = value
        return result

    def validate(self) -> None:
        """Validate required fields."""
        if not self.title:
            raise ValueError("title is required in frontmatter")


@dataclass
class ExtractedContent:
    """Result from extracting content from a file."""

    text: str
    metadata: dict[str, Any]
    original_format: str


@dataclass
class SplitDocument:
    """A document produced by section splitting."""

    slug: str
    body: str
    frontmatter: dict[str, Any]


@dataclass
class SplitResult:
    """Result from section splitting."""

    is_split: bool
    parent: SplitDocument | None = None
    children: list[SplitDocument] = field(default_factory=list)
    single: SplitDocument | None = None


@dataclass
class Fingerprint:
    """Result from fingerprint generation."""

    synopsis: str
    fingerprint: list[str]
    categories: list[str]
    concepts: list[str]


@dataclass
class DedupResult:
    """Result from deduplication check."""

    is_duplicate: bool
    existing_slug: str | None = None
    is_update: bool = False


@dataclass
class IndexEntry:
    """A single entry in the master index."""

    slug: str
    title: str
    categories: list[str]
    synopsis: str
    fingerprint: list[str]
    has_children: bool = False
    parent: str | None = None


@dataclass
class FTSResult:
    """A single result from FTS5 search."""

    slug: str
    title: str
    synopsis: str
    score: float
    snippet: str = ""
