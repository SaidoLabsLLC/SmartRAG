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

## Features

- Zero infrastructure — runs on SQLite (ships with Python)
- Full offline & mobile capability
- Human-readable storage (markdown + YAML frontmatter)
- Obsidian-compatible knowledge stores
- Sub-10ms retrieval at any scale
- 3-line SDK integration

## License

Proprietary — see [LICENSE](LICENSE).

- **Attribution required** on all forks and copies
- **No modifications** may be distributed without crediting Saido Labs LLC
- **Commercial use** requires a paid license — contact jesse@saidolabs.com

Built by [Saido Labs LLC](https://saidolabs.com).
