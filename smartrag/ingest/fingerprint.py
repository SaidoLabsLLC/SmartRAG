"""Fingerprint generation for documents — extractive and LLM modes.

Produces a Fingerprint containing:
  - synopsis:    most informative sentence (≤200 chars)
  - fingerprint: 5-10 TF-IDF keywords
  - categories:  1-5 heuristic categories
  - concepts:    3-10 noun-phrase concepts

Supports two modes:
  - ``extractive`` — zero-LLM, uses TF-IDF and heuristics (default)
  - ``llm`` — uses an LLMProvider for synopsis and keyword generation,
    falling back to extractive on failure
"""

from __future__ import annotations

import json
import logging
import re

from smartrag.types import Fingerprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop words — ~170 common English tokens
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "arent",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "couldn",
        "couldnt",
        "d",
        "did",
        "didn",
        "didnt",
        "do",
        "does",
        "doesn",
        "doesnt",
        "doing",
        "don",
        "dont",
        "down",
        "during",
        "each",
        "even",
        "every",
        "few",
        "for",
        "from",
        "further",
        "get",
        "got",
        "had",
        "hadn",
        "has",
        "hasn",
        "hasnt",
        "have",
        "haven",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "isnt",
        "it",
        "its",
        "itself",
        "just",
        "let",
        "ll",
        "m",
        "may",
        "me",
        "might",
        "more",
        "most",
        "must",
        "mustn",
        "my",
        "myself",
        "need",
        "needn",
        "no",
        "nor",
        "not",
        "now",
        "o",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "s",
        "same",
        "shall",
        "shan",
        "she",
        "should",
        "shouldn",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "us",
        "ve",
        "very",
        "was",
        "wasn",
        "we",
        "were",
        "weren",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "won",
        "would",
        "wouldn",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

# ---------------------------------------------------------------------------
# Static IDF approximation — common software-documentation corpus
# ---------------------------------------------------------------------------

# Tokens that appear in almost every document get low IDF; domain-specific
# tokens get a higher default.  This avoids needing a full corpus scan in
# Phase 1.
_COMMON_TOKENS: frozenset[str] = frozenset(
    {
        "use",
        "using",
        "used",
        "new",
        "make",
        "like",
        "example",
        "set",
        "way",
        "see",
        "run",
        "work",
        "file",
        "code",
        "data",
        "name",
        "type",
        "value",
        "function",
        "class",
        "method",
        "object",
        "list",
        "string",
        "number",
        "return",
        "call",
        "create",
        "add",
        "remove",
        "update",
        "default",
        "first",
        "last",
        "next",
        "time",
        "note",
        "one",
        "two",
        "system",
        "app",
        "application",
        "service",
        "server",
        "client",
        "request",
        "response",
        "error",
        "test",
        "true",
        "false",
        "null",
        "none",
        "import",
        "export",
        "module",
        "package",
        "install",
        "version",
        "config",
        "option",
        "param",
        "parameter",
        "result",
        "output",
        "input",
        "key",
        "path",
        "url",
        "http",
        "https",
        "api",
        "json",
        "html",
        "css",
    }
)

_IDF_COMMON = 1.0  # appears in many docs
_IDF_DEFAULT = 3.5  # average rarity
_IDF_RARE = 5.0  # domain-specific, uncommon


def _static_idf(token: str) -> float:
    """Return a static IDF estimate for *token*."""
    if token in STOP_WORDS:
        return 0.0
    if token in _COMMON_TOKENS:
        return _IDF_COMMON
    if len(token) <= 3:
        return _IDF_DEFAULT * 0.7
    return _IDF_DEFAULT


# ---------------------------------------------------------------------------
# Category taxonomy  (~50 categories with keyword triggers)
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "api": [
        "api",
        "endpoint",
        "rest",
        "graphql",
        "grpc",
        "openapi",
        "swagger",
        "webhook",
        "oauth",
        "jwt",
        "bearer",
        "http",
        "request",
        "response",
        "payload",
        "route",
        "middleware",
    ],
    "architecture": [
        "architecture",
        "microservice",
        "monolith",
        "cqrs",
        "event-sourcing",
        "saga",
        "hexagonal",
        "domain-driven",
        "ddd",
        "soa",
        "serverless",
        "pattern",
        "solid",
        "layered",
    ],
    "database": [
        "database",
        "sql",
        "nosql",
        "postgres",
        "postgresql",
        "mysql",
        "sqlite",
        "mongo",
        "mongodb",
        "redis",
        "dynamodb",
        "migration",
        "schema",
        "index",
        "query",
        "transaction",
        "orm",
    ],
    "devops": [
        "devops",
        "cicd",
        "pipeline",
        "jenkins",
        "github-actions",
        "gitlab",
        "terraform",
        "ansible",
        "helm",
        "argocd",
        "deployment",
        "rollback",
        "canary",
        "blue-green",
    ],
    "frontend": [
        "frontend",
        "react",
        "vue",
        "angular",
        "svelte",
        "nextjs",
        "nuxt",
        "css",
        "tailwind",
        "component",
        "dom",
        "browser",
        "spa",
        "ssr",
        "hydration",
        "jsx",
        "tsx",
        "html",
    ],
    "backend": [
        "backend",
        "server",
        "express",
        "fastapi",
        "django",
        "flask",
        "spring",
        "nestjs",
        "node",
        "golang",
        "rust",
        "java",
        "python",
        "cron",
        "worker",
        "queue",
    ],
    "security": [
        "security",
        "authentication",
        "authorization",
        "encryption",
        "tls",
        "ssl",
        "xss",
        "csrf",
        "injection",
        "vulnerability",
        "firewall",
        "rbac",
        "acl",
        "audit",
        "compliance",
        "owasp",
    ],
    "testing": [
        "testing",
        "test",
        "unittest",
        "pytest",
        "jest",
        "mocha",
        "cypress",
        "playwright",
        "integration",
        "e2e",
        "mock",
        "stub",
        "fixture",
        "coverage",
        "tdd",
        "bdd",
        "assertion",
    ],
    "infrastructure": [
        "infrastructure",
        "aws",
        "azure",
        "gcp",
        "cloud",
        "ec2",
        "s3",
        "lambda",
        "iam",
        "vpc",
        "subnet",
        "load-balancer",
        "cdn",
        "dns",
        "nginx",
        "caddy",
        "traefik",
    ],
    "containers": [
        "docker",
        "container",
        "kubernetes",
        "k8s",
        "pod",
        "helm",
        "dockerfile",
        "compose",
        "registry",
        "image",
        "namespace",
        "ingress",
    ],
    "monitoring": [
        "monitoring",
        "observability",
        "logging",
        "metrics",
        "tracing",
        "prometheus",
        "grafana",
        "datadog",
        "sentry",
        "alerting",
        "healthcheck",
        "uptime",
        "apm",
    ],
    "data-engineering": [
        "etl",
        "pipeline",
        "spark",
        "airflow",
        "kafka",
        "streaming",
        "batch",
        "warehouse",
        "lakehouse",
        "dbt",
        "parquet",
        "avro",
        "data-lake",
    ],
    "machine-learning": [
        "machine-learning",
        "ml",
        "model",
        "training",
        "inference",
        "neural",
        "deep-learning",
        "pytorch",
        "tensorflow",
        "transformer",
        "embedding",
        "llm",
        "gpt",
        "fine-tune",
        "rag",
    ],
    "finance": [
        "finance",
        "payment",
        "stripe",
        "invoice",
        "billing",
        "subscription",
        "ledger",
        "accounting",
        "revenue",
        "tax",
        "currency",
        "transaction",
    ],
    "legal": [
        "legal",
        "compliance",
        "gdpr",
        "hipaa",
        "pci",
        "terms",
        "privacy",
        "policy",
        "license",
        "copyright",
        "regulation",
        "contract",
    ],
    "marketing": [
        "marketing",
        "seo",
        "analytics",
        "campaign",
        "funnel",
        "conversion",
        "acquisition",
        "retention",
        "ab-test",
        "landing-page",
        "brand",
    ],
    "tutorial": [
        "tutorial",
        "step-by-step",
        "walkthrough",
        "how-to",
        "getting-started",
        "quickstart",
        "beginner",
        "learn",
        "lesson",
        "exercise",
    ],
    "reference": [
        "reference",
        "cheatsheet",
        "cheat-sheet",
        "lookup",
        "glossary",
        "appendix",
        "table",
        "enumeration",
        "catalog",
        "registry",
    ],
    "guide": [
        "guide",
        "handbook",
        "manual",
        "playbook",
        "runbook",
        "cookbook",
        "best-practice",
        "recommendation",
    ],
    "specification": [
        "specification",
        "spec",
        "rfc",
        "standard",
        "protocol",
        "schema",
        "format",
        "grammar",
        "bnf",
        "interface",
    ],
    "research": [
        "research",
        "paper",
        "study",
        "experiment",
        "hypothesis",
        "finding",
        "abstract",
        "methodology",
        "literature",
    ],
    "networking": [
        "networking",
        "tcp",
        "udp",
        "ip",
        "socket",
        "websocket",
        "grpc",
        "protocol",
        "latency",
        "bandwidth",
        "proxy",
        "nat",
    ],
    "caching": [
        "cache",
        "caching",
        "redis",
        "memcached",
        "ttl",
        "invalidation",
        "eviction",
        "lru",
        "cdn",
    ],
    "messaging": [
        "messaging",
        "rabbitmq",
        "kafka",
        "sqs",
        "pubsub",
        "event-bus",
        "message-queue",
        "amqp",
        "nats",
        "mqtt",
    ],
    "authentication": [
        "auth",
        "oauth",
        "saml",
        "sso",
        "ldap",
        "openid",
        "identity",
        "session",
        "token",
        "mfa",
        "2fa",
        "passkey",
    ],
    "version-control": [
        "git",
        "github",
        "gitlab",
        "bitbucket",
        "branch",
        "merge",
        "rebase",
        "commit",
        "pull-request",
        "pr",
    ],
    "performance": [
        "performance",
        "optimization",
        "profiling",
        "benchmark",
        "latency",
        "throughput",
        "bottleneck",
        "concurrency",
        "async",
        "parallel",
    ],
    "mobile": [
        "mobile",
        "ios",
        "android",
        "react-native",
        "flutter",
        "swift",
        "kotlin",
        "capacitor",
        "cordova",
    ],
    "design-patterns": [
        "pattern",
        "singleton",
        "factory",
        "observer",
        "strategy",
        "decorator",
        "adapter",
        "proxy",
        "builder",
        "repository",
    ],
    "configuration": [
        "config",
        "configuration",
        "env",
        "dotenv",
        "yaml",
        "toml",
        "ini",
        "settings",
        "feature-flag",
    ],
    "documentation": [
        "documentation",
        "docs",
        "readme",
        "changelog",
        "adr",
        "decision-record",
        "wiki",
        "annotation",
        "docstring",
    ],
    "cli": [
        "cli",
        "command-line",
        "terminal",
        "shell",
        "bash",
        "powershell",
        "argument",
        "flag",
        "prompt",
    ],
    "ai": [
        "ai",
        "artificial-intelligence",
        "chatbot",
        "agent",
        "prompt",
        "completion",
        "embedding",
        "vector",
        "retrieval",
        "semantic",
    ],
    "search": [
        "search",
        "elasticsearch",
        "solr",
        "fts",
        "full-text",
        "ranking",
        "relevance",
        "bm25",
        "inverted-index",
    ],
    "workflow": [
        "workflow",
        "automation",
        "orchestration",
        "state-machine",
        "dag",
        "scheduler",
        "cron",
        "trigger",
    ],
    "storage": [
        "storage",
        "blob",
        "s3",
        "filesystem",
        "upload",
        "download",
        "attachment",
        "media",
        "asset",
    ],
    "email": [
        "email",
        "smtp",
        "imap",
        "sendgrid",
        "ses",
        "template",
        "notification",
        "mailgun",
    ],
    "realtime": [
        "realtime",
        "websocket",
        "sse",
        "server-sent",
        "push",
        "live",
        "streaming",
        "socket",
    ],
    "concurrency": [
        "concurrency",
        "thread",
        "async",
        "await",
        "coroutine",
        "lock",
        "mutex",
        "semaphore",
        "race-condition",
        "deadlock",
    ],
    "logging": [
        "logging",
        "log",
        "structured-logging",
        "logfmt",
        "syslog",
        "rotation",
        "aggregation",
        "elk",
        "loki",
    ],
    "migration": [
        "migration",
        "upgrade",
        "backward-compatible",
        "breaking-change",
        "deprecation",
        "rollback",
        "downtime",
    ],
    "error-handling": [
        "error",
        "exception",
        "retry",
        "circuit-breaker",
        "fallback",
        "timeout",
        "backoff",
        "idempotent",
    ],
    "dependency-management": [
        "dependency",
        "package",
        "npm",
        "pip",
        "cargo",
        "maven",
        "gradle",
        "lockfile",
        "semver",
    ],
    "internationalization": [
        "i18n",
        "l10n",
        "locale",
        "translation",
        "unicode",
        "charset",
        "timezone",
        "tz",
    ],
    "accessibility": [
        "accessibility",
        "a11y",
        "aria",
        "screen-reader",
        "wcag",
        "keyboard-navigation",
        "contrast",
    ],
    "analytics": [
        "analytics",
        "tracking",
        "event",
        "funnel",
        "cohort",
        "dashboard",
        "kpi",
        "metric",
        "telemetry",
    ],
    "project-management": [
        "project",
        "sprint",
        "kanban",
        "scrum",
        "agile",
        "milestone",
        "roadmap",
        "standup",
        "retro",
    ],
}

# Pre-build inverted index: keyword -> set of categories
_KEYWORD_TO_CATEGORIES: dict[str, list[str]] = {}
for _cat, _words in CATEGORY_KEYWORDS.items():
    for _w in _words:
        _KEYWORD_TO_CATEGORIES.setdefault(_w, []).append(_cat)

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]*[a-z0-9]|[a-z]", re.ASCII)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"\*{1,3}([^*\n]+)\*{1,3}|_{1,3}([^_\n]+)_{1,3}")
_CAPITALIZED_RE = re.compile(r"\b([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,2})\b")
_MD_STRIP_RE = re.compile(
    r"^#{1,6}\s+|"           # heading markers
    r"\*{1,3}|_{1,3}|"       # bold/italic markers
    r"`{1,3}|"               # code markers
    r"^\s*[-*+]\s|"          # list bullets
    r"^\s*\d+\.\s|"          # numbered list
    r"\[([^\]]*)\]\([^)]*\)",  # links -> keep text
    re.MULTILINE,
)


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting, keeping plain text content."""
    # Replace links with their display text
    result = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Strip heading markers, bold/italic, code backticks, list bullets
    result = re.sub(r"^#{1,6}\s+", "", result, flags=re.MULTILINE)
    result = re.sub(r"\*{1,3}|_{1,3}", "", result)
    result = re.sub(r"`{1,3}", "", result)
    result = re.sub(r"^\s*[-*+]\s", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*\d+\.\s", "", result, flags=re.MULTILINE)
    return result


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize, remove stop words and non-qualifying tokens."""
    raw = _TOKEN_RE.findall(text.lower())
    return [
        t
        for t in raw
        if t not in STOP_WORDS and len(t) > 1 and len(t) <= 30 and not t.isdigit()
    ]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences — simple heuristic split."""
    # Normalise whitespace first.
    clean = re.sub(r"\s+", " ", text).strip()
    parts = _SENTENCE_RE.split(clean)
    return [s.strip() for s in parts if s.strip()]


def _first_n_words(text: str, n: int) -> str:
    """Return the first *n* words of *text*."""
    words = text.split()
    return " ".join(words[:n])


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars*, breaking at word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated.rstrip(" .,;:!?") + "..."


# ---------------------------------------------------------------------------
# Fingerprinter
# ---------------------------------------------------------------------------


class Fingerprinter:
    """Generate document fingerprints — synopsis, keywords, categories, concepts.

    Supports two modes:
      - ``"extractive"`` — pure heuristic, zero LLM (default)
      - ``"llm"`` — uses an :class:`LLMProvider` for synopsis and keywords,
        falling back to extractive if the LLM call fails
    """

    def __init__(
        self,
        mode: str = "extractive",
        llm_provider: object | None = None,
    ) -> None:
        if mode not in ("extractive", "llm"):
            raise ValueError(
                f"Unsupported fingerprint mode: {mode!r}. "
                "Supported modes: 'extractive', 'llm'."
            )
        if mode == "llm" and llm_provider is None:
            raise ValueError(
                "An llm_provider is required when mode='llm'. "
                "Pass an LLMProvider instance or use mode='extractive'."
            )
        self.mode = mode
        self._llm = llm_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        title: str = "",
        existing_categories: list[str] | None = None,
    ) -> Fingerprint:
        """Produce a full Fingerprint for *text*.

        When ``mode="llm"`` and an LLM provider is available, the synopsis
        and keyword fingerprint are generated via the LLM.  If the LLM call
        fails for any reason, the method silently falls back to extractive.
        """
        if self.mode == "llm" and self._llm is not None:
            synopsis = self._generate_synopsis_llm(text, title)
            fingerprint_kw = self._generate_fingerprint_llm(text)
        else:
            synopsis = None
            fingerprint_kw = None

        # Fall back to extractive for any component that the LLM did not
        # produce (either because mode is extractive or because LLM failed).
        if not synopsis:
            synopsis = self._extractive_synopsis(text, title, max_chars=200)
        if not fingerprint_kw:
            fingerprint_kw = self._keyword_fingerprint(text, title)

        categories = self._detect_categories(fingerprint_kw, existing_categories)
        concepts = self._extract_concepts(text, title, fingerprint_kw)
        return Fingerprint(
            synopsis=synopsis,
            fingerprint=fingerprint_kw,
            categories=categories,
            concepts=concepts,
        )

    def generate_section_synopsis(
        self, section_text: str, section_title: str = ""
    ) -> str:
        """Generate a short synopsis for a single section (max 150 chars)."""
        return self._extractive_synopsis(section_text, section_title, max_chars=150)

    # ------------------------------------------------------------------
    # Synopsis
    # ------------------------------------------------------------------

    def _extractive_synopsis(
        self, text: str, title: str, max_chars: int = 200
    ) -> str:
        """Pick the most informative sentence from the first 500 words."""
        plain = _strip_markdown(text)
        words = plain.split()
        if len(words) < 50:
            return _truncate_at_word_boundary(
                re.sub(r"\s+", " ", plain).strip(), max_chars
            )

        window = " ".join(words[:500])
        sentences = _split_sentences(window)
        if not sentences:
            return _truncate_at_word_boundary(
                re.sub(r"\s+", " ", plain).strip(), max_chars
            )

        title_tokens = set(_tokenize(title)) if title else set()
        best_score = -1.0
        best_sentence = sentences[0]

        for idx, sent in enumerate(sentences):
            score = self._score_sentence(sent, idx, title_tokens)
            if score > best_score:
                best_score = score
                best_sentence = sent

        return _truncate_at_word_boundary(best_sentence.strip(), max_chars)

    @staticmethod
    def _score_sentence(
        sentence: str, position: int, title_tokens: set[str]
    ) -> float:
        """Score a sentence for synopsis candidacy."""
        length = len(sentence)

        # Length penalty — prefer 50-150 chars
        if 50 <= length <= 150:
            length_score = 1.0
        elif length < 50:
            length_score = length / 50.0
        else:
            # Gentle penalty above 150
            length_score = max(0.3, 1.0 - (length - 150) / 300.0)

        # Position bonus — first 3 sentences score higher
        if position == 0:
            position_score = 1.5
        elif position <= 2:
            position_score = 1.2
        else:
            position_score = 1.0

        # Title-word overlap bonus (2x per matching token)
        if title_tokens:
            sent_tokens = set(_tokenize(sentence))
            overlap = len(sent_tokens & title_tokens)
            title_score = 1.0 + overlap * 2.0
        else:
            title_score = 1.0

        return length_score * position_score * title_score

    # ------------------------------------------------------------------
    # TF-IDF keyword fingerprint
    # ------------------------------------------------------------------

    def _keyword_fingerprint(
        self, text: str, title: str, min_kw: int = 5, max_kw: int = 10
    ) -> list[str]:
        """Extract top keywords using TF x static-IDF with positional boosts."""
        tokens = _tokenize(text)
        if not tokens:
            return []

        # Term frequency (raw count)
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1.0

        # Normalise TF by max frequency
        max_tf = max(tf.values()) if tf else 1.0
        for t in tf:
            tf[t] = 0.5 + 0.5 * (tf[t] / max_tf)  # augmented TF

        # Positional boost sets
        title_tokens = set(_tokenize(title)) if title else set()
        first_para_tokens = self._first_paragraph_tokens(text)
        heading_tokens = self._heading_tokens(text)

        # Score: TF x IDF x boosts
        scores: dict[str, float] = {}
        for token, tf_val in tf.items():
            idf = _static_idf(token)
            if idf <= 0:
                continue

            boost = 1.0
            if token in title_tokens:
                boost *= 3.0
            if token in first_para_tokens:
                boost *= 2.0
            if token in heading_tokens:
                boost *= 2.0

            scores[token] = tf_val * idf * boost

        # Sort and take top N
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        keywords = [tok for tok, _ in ranked[:max_kw]]

        # Ensure at least min_kw if available
        return keywords[:max(min_kw, len(keywords))]

    @staticmethod
    def _first_paragraph_tokens(text: str) -> set[str]:
        """Tokenize the first paragraph (up to first blank line)."""
        parts = re.split(r"\n\s*\n", text, maxsplit=1)
        if parts:
            return set(_tokenize(parts[0]))
        return set()

    @staticmethod
    def _heading_tokens(text: str) -> set[str]:
        """Tokenize all markdown headings."""
        headings = _HEADING_RE.findall(text)
        tokens: set[str] = set()
        for h in headings:
            tokens.update(_tokenize(h))
        return tokens

    # ------------------------------------------------------------------
    # Category detection
    # ------------------------------------------------------------------

    def _detect_categories(
        self,
        fingerprint_kw: list[str],
        existing_categories: list[str] | None = None,
    ) -> list[str]:
        """Map fingerprint keywords to taxonomy categories."""
        cat_scores: dict[str, float] = {}

        for kw in fingerprint_kw:
            # Direct keyword match
            if kw in _KEYWORD_TO_CATEGORIES:
                for cat in _KEYWORD_TO_CATEGORIES[kw]:
                    cat_scores[cat] = cat_scores.get(cat, 0) + 1.0

            # Partial/substring match against category keyword lists
            for cat, cat_kws in CATEGORY_KEYWORDS.items():
                for ckw in cat_kws:
                    if kw in ckw or ckw in kw:
                        if kw != ckw:  # avoid double-counting exact match
                            cat_scores[cat] = cat_scores.get(cat, 0) + 0.5

        if not cat_scores:
            return ["general"]

        # Sort by score descending, take top 5 with score >= 1.0
        ranked = sorted(cat_scores.items(), key=lambda kv: kv[1], reverse=True)
        categories = [cat for cat, score in ranked if score >= 1.0][:5]

        if not categories:
            return ["general"]

        # Merge with existing if provided
        if existing_categories:
            merged = list(dict.fromkeys(existing_categories + categories))
            return merged[:5]

        return categories

    # ------------------------------------------------------------------
    # Concept extraction
    # ------------------------------------------------------------------

    def _extract_concepts(
        self,
        text: str,
        title: str,
        fingerprint_kw: list[str],
        min_concepts: int = 3,
        max_concepts: int = 10,
    ) -> list[str]:
        """Extract 1-3 word noun-phrase concepts."""
        concept_scores: dict[str, float] = {}

        # Common sentence-initial words that are not concepts
        _trivial_caps = {
            "the", "this", "that", "these", "those", "there", "here",
            "when", "where", "what", "which", "who", "how", "why",
            "each", "every", "always", "never", "also", "however",
            "since", "because", "although", "while", "after", "before",
            "during", "until", "once", "both", "either", "neither",
            "many", "much", "most", "some", "any", "all", "few",
            "such", "only", "just", "even", "still", "already",
            "often", "sometimes", "usually", "generally", "typically",
            "note", "see", "use", "using", "used",
        }

        # 1. Capitalized sequences (1-3 words) — likely proper nouns / terms
        #    First pass: count occurrences, then filter.
        cap_counts: dict[str, int] = {}
        for match in _CAPITALIZED_RE.finditer(text):
            phrase = match.group(1).strip()
            words = phrase.split()
            if 1 <= len(words) <= 3:
                if len(words) == 1 and (
                    phrase.lower() in STOP_WORDS
                    or phrase.lower() in _trivial_caps
                    or len(phrase) <= 3
                ):
                    continue
                cap_counts[phrase] = cap_counts.get(phrase, 0) + 1
        for phrase, count in cap_counts.items():
            # Single words must appear 2+ times to qualify as concepts
            # (avoids sentence-initial false positives)
            if len(phrase.split()) == 1 and count < 2:
                continue
            concept_scores[phrase] = concept_scores.get(phrase, 0) + count * 1.0

        # 2. Heading words as concepts
        for heading in _HEADING_RE.findall(text):
            cleaned = re.sub(r"[^a-zA-Z0-9 \-]", "", heading).strip()
            if not cleaned:
                continue
            h_words = cleaned.split()
            if len(h_words) <= 3:
                concept_scores[cleaned] = concept_scores.get(cleaned, 0) + 3.0
            else:
                # Extract overlapping 2-3 word windows from longer headings
                for size in (3, 2):
                    for j in range(len(h_words) - size + 1):
                        sub = " ".join(h_words[j : j + size])
                        concept_scores[sub] = (
                            concept_scores.get(sub, 0) + 2.5
                        )

        # 3. Bold/italic text as concepts
        for match in _BOLD_ITALIC_RE.finditer(text):
            phrase = (match.group(1) or match.group(2) or "").strip()
            words = phrase.split()
            if 1 <= len(words) <= 3 and len(phrase) <= 50:
                concept_scores[phrase] = concept_scores.get(phrase, 0) + 2.0

        # 4. High-TF-IDF bigrams
        tokens = _tokenize(text)
        bigram_counts: dict[str, int] = {}
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
        for bigram, count in bigram_counts.items():
            if count >= 2:
                concept_scores[bigram] = concept_scores.get(bigram, 0) + count * 1.5

        # 5. Title words as high-relevance concepts
        if title:
            title_cleaned = re.sub(r"[^a-zA-Z0-9 \-]", "", title).strip()
            title_words = title_cleaned.split()
            if 1 <= len(title_words) <= 3:
                concept_scores[title_cleaned] = (
                    concept_scores.get(title_cleaned, 0) + 5.0
                )

        # Filter: remove single-char, pure numbers, stop words as sole concept
        filtered: dict[str, float] = {}
        for concept, score in concept_scores.items():
            stripped = concept.strip()
            if len(stripped) <= 1:
                continue
            if stripped.replace(" ", "").replace("-", "").isdigit():
                continue
            if stripped.lower() in STOP_WORDS:
                continue
            filtered[stripped] = score

        # Sort and return top N
        ranked = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
        concepts = [c for c, _ in ranked[:max_concepts]]

        # Pad with top fingerprint keywords if under minimum
        if len(concepts) < min_concepts:
            existing = {c.lower() for c in concepts}
            for kw in fingerprint_kw:
                if kw.lower() not in existing and len(concepts) < min_concepts:
                    concepts.append(kw)
                    existing.add(kw.lower())

        return concepts

    # ------------------------------------------------------------------
    # LLM-backed generation (mode="llm")
    # ------------------------------------------------------------------

    def _generate_synopsis_llm(self, text: str, title: str) -> str | None:
        """Ask the LLM for a one-sentence summary (max 200 chars).

        Returns ``None`` on any failure so the caller can fall back to
        extractive mode.
        """
        # Provide a manageable context window — first ~2000 chars.
        excerpt = text[:2000]
        title_hint = f' titled "{title}"' if title else ""

        prompt = (
            f"You are a concise technical summariser.\n"
            f"Given the following document excerpt{title_hint}, write exactly "
            f"ONE sentence that captures the main point.  The sentence MUST be "
            f"200 characters or fewer.  Output ONLY the sentence — no preamble, "
            f"no quotes, no explanation.\n\n"
            f"---\n{excerpt}\n---"
        )
        try:
            result = self._llm.generate(prompt, max_tokens=100)  # type: ignore[union-attr]
            if result:
                # Hard-truncate at 200 chars as a safety net.
                return _truncate_at_word_boundary(result.strip(), 200)
        except Exception:
            logger.warning(
                "LLM synopsis generation failed; falling back to extractive.",
                exc_info=True,
            )
        return None

    def _generate_fingerprint_llm(self, text: str) -> list[str] | None:
        """Ask the LLM for 5-10 keywords as a JSON array.

        Returns ``None`` on any failure so the caller can fall back to
        extractive mode.
        """
        excerpt = text[:2000]

        prompt = (
            "You are a keyword extraction engine.\n"
            "Given the following document excerpt, return a JSON array of "
            "5 to 10 lowercase keywords that best describe its content.  "
            "Output ONLY the JSON array — no markdown fences, no explanation.\n\n"
            f"---\n{excerpt}\n---"
        )
        try:
            raw = self._llm.generate(prompt, max_tokens=200)  # type: ignore[union-attr]
            if not raw:
                return None

            # Strip possible markdown code fences the LLM may emit.
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                # Remove opening fence (with optional language tag) and closing fence.
                cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
                cleaned = cleaned.strip()

            keywords = json.loads(cleaned)
            if isinstance(keywords, list) and all(
                isinstance(k, str) for k in keywords
            ):
                # Normalise: lowercase, strip, deduplicate, limit to 10.
                seen: set[str] = set()
                result: list[str] = []
                for kw in keywords:
                    kw_clean = kw.lower().strip()
                    if kw_clean and kw_clean not in seen:
                        seen.add(kw_clean)
                        result.append(kw_clean)
                    if len(result) >= 10:
                        break
                if len(result) >= 5:
                    return result
        except (json.JSONDecodeError, Exception):
            logger.warning(
                "LLM fingerprint generation failed; falling back to extractive.",
                exc_info=True,
            )
        return None
