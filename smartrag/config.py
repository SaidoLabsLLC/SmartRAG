"""SmartRAG configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SmartRAGConfig:
    """Configuration for SmartRAG."""

    split_threshold: int = 2000
    synopsis_mode: str = "extractive"  # "extractive" or "llm"
    llm_provider: str | None = None
    llm_api_key: str | None = None
    embeddings: bool = False
    fts5: bool = True
    tier0_cache_size: int = 50_000
    max_results: int = 10
    obsidian_compat: bool = True
    ranking_weights: dict[str, float] = field(default_factory=lambda: {
        "master_index": 1.0,
        "fts5": 1.0,
        "embeddings": 0.8,
        "backlinks": 0.3,
    })
    self_tuning: bool = False
    feedback: bool = True
    feedback_anonymize: bool = False
    tune_interval: int = 100
