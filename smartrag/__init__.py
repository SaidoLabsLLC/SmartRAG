"""SmartRAG — The retrieval engine that makes vector databases unnecessary."""

from smartrag.config import SmartRAGConfig
from smartrag.core import SmartRAG
from smartrag.types import (
    Document,
    DocumentFrontmatter,
    IngestResult,
    QueryResult,
    RetrievalResult,
    SearchResult,
)

__version__ = "0.1.0"
__all__ = [
    "SmartRAG",
    "SmartRAGConfig",
    "QueryResult",
    "RetrievalResult",
    "SearchResult",
    "IngestResult",
    "Document",
    "DocumentFrontmatter",
]
