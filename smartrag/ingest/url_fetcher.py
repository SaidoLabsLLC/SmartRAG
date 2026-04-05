"""URL fetcher with SSRF protection for the SmartRAG ingest pipeline.

Fetches web pages, extracts readable text and metadata using BeautifulSoup,
and returns ExtractedContent for downstream processing. All DNS resolutions
are validated against blocked IP ranges before any HTTP request is made.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import logging
import socket
import tempfile
import time
from typing import Any
from urllib.parse import urlparse

from smartrag.ingest.extractors import clean_text, count_words
from smartrag.types import ExtractedContent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

_BLOCKED_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local + cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]

# Configurable domain allowlist — domains listed here bypass SSRF IP checks.
# Useful for internal services that legitimately resolve to private ranges.
DOMAIN_ALLOWLIST: set[str] = set()


def is_safe_url(url: str) -> bool:
    """Validate that *url* does not resolve to a blocked IP range.

    Steps:
      1. Parse URL and ensure scheme is http or https.
      2. Resolve hostname via DNS to one or more IP addresses.
      3. Verify none of the resolved IPs fall within blocked ranges.
      4. Domains in DOMAIN_ALLOWLIST skip the IP check.

    Returns:
        True if the URL is safe to fetch; False otherwise.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    # Scheme validation
    if parsed.scheme not in ("http", "https"):
        logger.warning("Blocked URL with disallowed scheme: %s", parsed.scheme)
        return False

    hostname = parsed.hostname
    if not hostname:
        logger.warning("Blocked URL with no hostname: %s", url)
        return False

    # Allowlist bypass
    if hostname in DOMAIN_ALLOWLIST:
        return True

    # Resolve hostname to IPs
    try:
        addrinfo = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        logger.warning("DNS resolution failed for hostname: %s", hostname)
        return False

    for family, _type, _proto, _canonname, sockaddr in addrinfo:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False

        for blocked in _BLOCKED_RANGES:
            if ip in blocked:
                logger.warning(
                    "Blocked SSRF attempt: %s resolved to %s (in %s)",
                    hostname,
                    ip_str,
                    blocked,
                )
                return False

    return True


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _require_httpx() -> None:
    """Raise a helpful error if httpx is not installed."""
    if importlib.util.find_spec("httpx") is None:
        raise ImportError(
            "httpx is required for URL fetching but is not installed. "
            "Install it with: pip install smartrag[cloud]  "
            "or: pip install httpx"
        )


def _require_bs4() -> None:
    """Raise a helpful error if BeautifulSoup is not installed."""
    if importlib.util.find_spec("bs4") is None:
        raise ImportError(
            "beautifulsoup4 is required for URL fetching but is not installed. "
            "Install it with: pip install beautifulsoup4"
        )


# ---------------------------------------------------------------------------
# HTML → markdown extraction (reuses patterns from extractors._extract_html)
# ---------------------------------------------------------------------------

def _html_to_extracted(
    html: str, source_url: str, response_headers: dict[str, str] | None = None,
) -> ExtractedContent:
    """Parse HTML string into ExtractedContent with markdown-style text.

    Mirrors the logic in ``extractors._extract_html`` but operates on a raw
    HTML string rather than a file path, and extracts richer metadata from
    ``<meta>`` tags.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    metadata: dict[str, Any] = {"source": source_url}

    # --- metadata extraction ------------------------------------------------

    # <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        metadata["title"] = title_tag.string.strip()

    # <meta name="description">
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        metadata["description"] = desc_tag["content"].strip()

    # <link rel="canonical">
    canonical_tag = soup.find("link", attrs={"rel": "canonical"})
    if canonical_tag and canonical_tag.get("href"):
        metadata["canonical_url"] = canonical_tag["href"].strip()

    # publish date — check common meta tag patterns
    for attr_name in ("article:published_time", "datePublished", "date"):
        date_tag = soup.find("meta", attrs={"property": attr_name}) or soup.find(
            "meta", attrs={"name": attr_name}
        )
        if date_tag and date_tag.get("content"):
            metadata["publish_date"] = date_tag["content"].strip()
            break

    # Also check <time> element with datetime attribute
    if "publish_date" not in metadata:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            metadata["publish_date"] = time_tag["datetime"].strip()

    # --- text extraction (mirrors extractors._extract_html) -----------------

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
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


# ---------------------------------------------------------------------------
# URLFetcher
# ---------------------------------------------------------------------------

class URLFetchError(Exception):
    """Raised when a URL fetch fails."""

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to fetch {url}: {reason}")


class URLFetcher:
    """Fetch web pages and extract content with SSRF protection.

    Parameters:
        timeout: HTTP request timeout in seconds.
        rate_limit: Minimum seconds between consecutive requests.
        max_redirects: Maximum number of HTTP redirects to follow.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        rate_limit: float = 1.0,
        max_redirects: int = 5,
    ) -> None:
        _require_httpx()
        _require_bs4()

        self._timeout = timeout
        self._rate_limit = rate_limit
        self._max_redirects = max_redirects
        self._last_request_time: float = 0.0

    def fetch(self, url: str) -> ExtractedContent:
        """Fetch a URL and return extracted content.

        Validates the URL against SSRF blocklists, issues an HTTP GET request,
        and extracts readable text and metadata.  If the response is a PDF,
        delegates to the PDF extractor instead.

        Args:
            url: The URL to fetch (must be http or https).

        Returns:
            ExtractedContent with normalized markdown text and metadata.

        Raises:
            URLFetchError: If the URL is blocked, unreachable, or returns an error.
        """
        import httpx

        # SSRF check
        if not is_safe_url(url):
            raise URLFetchError(url, "URL blocked by SSRF protection")

        try:
            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=True,
                max_redirects=self._max_redirects,
                headers={
                    "User-Agent": "SmartRAG/1.0 (+https://github.com/smartrag)",
                    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
                },
            ) as client:
                response = client.get(url)
                response.raise_for_status()
        except httpx.TimeoutException:
            raise URLFetchError(url, f"Request timed out after {self._timeout}s")
        except httpx.TooManyRedirects:
            raise URLFetchError(
                url, f"Too many redirects (max {self._max_redirects})"
            )
        except httpx.HTTPStatusError as exc:
            raise URLFetchError(
                url, f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}"
            )
        except httpx.HTTPError as exc:
            raise URLFetchError(url, str(exc))

        content_type = response.headers.get("content-type", "")

        # PDF handling — download to temp file, delegate to PDF extractor
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            return self._extract_pdf_from_response(response, url)

        # Non-HTML guard
        if content_type and "text/html" not in content_type and "application/xhtml" not in content_type:
            # Attempt plain-text extraction for text/* types
            if content_type.startswith("text/"):
                text = clean_text(response.text)
                return ExtractedContent(
                    text=text,
                    metadata={
                        "source": url,
                        "content_type": content_type,
                        "word_count": count_words(text),
                    },
                    original_format="text",
                )
            raise URLFetchError(
                url,
                f"Unsupported content type: {content_type}. Expected HTML or PDF.",
            )

        # HTML extraction
        response_headers = dict(response.headers)
        return _html_to_extracted(response.text, url, response_headers)

    def fetch_with_rate_limit(self, url: str) -> ExtractedContent:
        """Fetch a URL, enforcing a minimum delay between requests.

        If called sooner than ``self._rate_limit`` seconds after the previous
        request, sleeps for the remaining time before proceeding.

        Args:
            url: The URL to fetch.

        Returns:
            ExtractedContent with normalized markdown text and metadata.
        """
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

        self._last_request_time = time.monotonic()
        return self.fetch(url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pdf_from_response(response: Any, url: str) -> ExtractedContent:
        """Write PDF response bytes to a temp file and extract with PyMuPDF."""
        from smartrag.ingest.extractors import _extract_pdf

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            extracted = _extract_pdf(tmp_path)
            # Overwrite source metadata with the URL
            extracted.metadata["source"] = url
            extracted.metadata.pop("source_file", None)
            return extracted
        finally:
            import os

            try:
                os.unlink(tmp_path)
            except OSError:
                pass
