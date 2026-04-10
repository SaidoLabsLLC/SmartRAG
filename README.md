# SmartRAG

**The retrieval engine that makes vector databases unnecessary.**

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-under%20active%20development-orange.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()

SmartRAG is a drop-in retrieval engine that replaces traditional RAG pipelines with structured, tiered document routing — delivering faster, more accurate, and cheaper retrieval with zero infrastructure dependencies.

## Quickstart

```bash
pip install smartrag
```

```python
from smartrag import SmartRAG

rag = SmartRAG("./my-docs")
rag.ingest("./data/")
result = rag.query("How do I configure auth?")
```

No vector database. No embedding model. No API keys. No infrastructure. Sub-10ms retrieval.

## How It Works

SmartRAG treats your knowledge base as an intelligently organized library, not a bag of vectors:

- **Tier 0** — Master index scan (in-memory, <1ms)
- **Tier 1** — Frontmatter evaluation (~2ms, metadata only)
- **Tier 2** — Section map traversal (~2ms, targeted reads)
- **Tier 3** — Full content (last resort)

Total: <10ms, <10KB I/O per query.

## Self-Improvement Loop

SmartRAG gets smarter the more you use it. Every query feeds a built-in feedback loop that automatically detects signal from usage patterns and tunes retrieval weights — no manual intervention required.

**How it works:**

1. **Query logging** — every query, its results, timing, and tier resolution are recorded in a lightweight SQLite store (`feedback.db`)
2. **Implicit signal detection** — SmartRAG watches for patterns that reveal result quality without asking the user:
   - **Repeat query** (same search within 60s) → results were unhelpful (score: 0.3)
   - **Query refinement** (>60% keyword overlap within 5 min) → results were close but not quite right (score: 0.5)
   - **Topic switch** (<20% overlap within 30s) → previous results were good enough to move on (score: 0.8)
3. **Self-tuning RRF weights** — after enough signal accumulates (50+ scored queries), SmartRAG adjusts source weights (master index, FTS5, embeddings, backlinks) based on which sources consistently produce top-ranked results
4. **Synopsis quality detection** — documents that repeatedly require Tier 3 (full content read) are flagged for synopsis regeneration

**Explicit feedback** is also supported via the SDK and REST API:

```python
result = rag.query("How do I configure auth?")
rag.record_feedback(result.query_id, score=0.9, used_slugs=["auth-config"])
```

```bash
# CLI
smartrag stats --feedback       # view retrieval statistics
smartrag tune --store ./docs    # trigger manual weight tuning
smartrag flagged --store ./docs # list documents needing better synopses
```

```
# REST API
POST /v1/feedback              # record explicit feedback
GET  /v1/stats/retrieval       # retrieval statistics
GET  /v1/stats/flagged         # documents needing regeneration
POST /v1/tune                  # trigger weight tuning
```

The feedback system is **on by default**, adds zero external dependencies (stdlib `sqlite3` only), and can be disabled with `SmartRAGConfig(feedback=False)` for zero overhead. Query text can be anonymized with `feedback_anonymize=True` (stores SHA-256 hashes instead of raw text). In multi-tenant deployments, each tenant gets an isolated `feedback.db`.

## Features

- Zero infrastructure — runs on SQLite (ships with Python)
- Full offline & mobile capability
- Human-readable storage (markdown + YAML frontmatter)
- Obsidian-compatible knowledge stores
- Sub-10ms retrieval at any scale
- Self-improving retrieval via implicit feedback and auto-tuning
- 3-line SDK integration

## License

Proprietary — see [LICENSE](LICENSE).

- **Attribution required** on all forks and copies
- **No modifications** may be distributed without crediting Saido Labs LLC
- **Commercial use** requires a paid license — contact jesse@saidolabs.com

Built by [Saido Labs LLC](https://saidolabs.com).
