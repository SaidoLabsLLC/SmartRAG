"""Server configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for the SmartRAG API server.

    All values are read from environment variables at instantiation time,
    falling back to sensible defaults for local development.
    """

    port: int = int(os.environ.get("SMARTRAG_PORT", "8000"))
    host: str = os.environ.get("SMARTRAG_HOST", "0.0.0.0")
    knowledge_dir: str = os.environ.get("SMARTRAG_KNOWLEDGE_DIR", "./knowledge")
    cors_origins: str = os.environ.get("SMARTRAG_CORS_ORIGINS", "*")
    rate_limit: int = int(os.environ.get("SMARTRAG_RATE_LIMIT", "100"))
    log_level: str = os.environ.get("SMARTRAG_LOG_LEVEL", "INFO")
    embeddings_enabled: bool = (
        os.environ.get("SMARTRAG_EMBEDDINGS_ENABLED", "false").lower() == "true"
    )
