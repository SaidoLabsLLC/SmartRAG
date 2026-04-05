"""LLM provider abstraction for enhanced synopsis and fingerprinting.

Supports OpenAI, Anthropic, and local Ollama backends with lazy imports
and graceful degradation when dependencies are missing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base for LLM text generation."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a text completion for *prompt*.

        Returns the generated string.  Implementations must raise on
        unrecoverable errors but should handle transient failures
        (retries, timeouts) internally where possible.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """Chat-completion provider backed by the ``openai`` package."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            import openai  # noqa: F401 — lazy import
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            ) from None

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        choice = response.choices[0]
        return (choice.message.content or "").strip()


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Messages-API provider backed by the ``anthropic`` package."""

    def __init__(
        self, api_key: str, model: str = "claude-sonnet-4-20250514"
    ) -> None:
        try:
            import anthropic  # noqa: F401 — lazy import
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            ) from None

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # The response contains a list of content blocks; concatenate text.
        parts: list[str] = []
        for block in message.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Provider that calls a local Ollama instance over HTTP."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        try:
            import httpx  # noqa: F401 — lazy import
        except ImportError:
            raise ImportError(
                "The 'httpx' package is required for OllamaProvider. "
                "Install it with: pip install httpx"
            ) from None

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._httpx = httpx

        # Validate connectivity at construction time.
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.HTTPStatusError, OSError) as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Ensure the Ollama server is running."
            ) from exc

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }
        resp = self._httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def create_provider(
    provider_name: str,
    api_key: str | None = None,
    **kwargs: object,
) -> LLMProvider:
    """Instantiate an LLM provider by name.

    Parameters
    ----------
    provider_name:
        One of ``"openai"``, ``"anthropic"``, ``"ollama"``.
    api_key:
        Required for ``openai`` and ``anthropic``; ignored for ``ollama``.
    **kwargs:
        Forwarded to the provider constructor (e.g. ``model``, ``base_url``).

    Returns
    -------
    LLMProvider
        A ready-to-use provider instance.

    Raises
    ------
    ValueError
        If *provider_name* is not recognised.
    ImportError
        If the required SDK is not installed.
    ConnectionError
        If Ollama is selected but unreachable.
    """
    name = provider_name.lower().strip()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {provider_name!r}. "
            f"Supported: {', '.join(sorted(_PROVIDERS))}"
        )

    cls = _PROVIDERS[name]

    if name in ("openai", "anthropic"):
        if not api_key:
            raise ValueError(
                f"An API key is required for the {provider_name!r} provider."
            )
        return cls(api_key=api_key, **kwargs)  # type: ignore[arg-type]

    # Ollama — no API key needed.
    return cls(**kwargs)  # type: ignore[arg-type]
