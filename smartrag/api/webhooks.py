"""Webhook registration, storage, and delivery for SmartRAG."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Supported webhook event types
WEBHOOK_EVENTS = frozenset(
    {
        "document.created",
        "document.updated",
        "document.deleted",
        "reindex.started",
        "reindex.completed",
    }
)

_RETRY_DELAYS = (1, 5, 25)  # seconds — exponential backoff


class WebhookManager:
    """Register, persist, and fire webhooks per tenant.

    Webhook configurations are stored in
    ``{tenant_dir}/.smartrag/webhooks.json``.
    """

    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _webhooks_path(self, tenant_id: str) -> str:
        return os.path.join(
            self._base_dir, "tenants", tenant_id, ".smartrag", "webhooks.json"
        )

    def _load(self, tenant_id: str) -> list[dict[str, Any]]:
        path = self._webhooks_path(tenant_id)
        if not os.path.isfile(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            logger.exception("Failed to load webhooks for tenant %s", tenant_id)
            return []

    def _save(self, tenant_id: str, webhooks: list[dict[str, Any]]) -> None:
        path = self._webhooks_path(tenant_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(webhooks, f, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(
        self,
        tenant_id: str,
        url: str,
        events: list[str],
        secret: str,
    ) -> dict[str, Any]:
        """Register a new webhook.

        Validates the URL against SSRF rules and checks that all
        requested event types are supported.

        Returns the persisted webhook record (including its generated id).

        Raises
        ------
        ValueError
            If the URL is unsafe, events are invalid, or the secret is empty.
        """
        # Validate events
        invalid = set(events) - WEBHOOK_EVENTS
        if invalid:
            raise ValueError(
                f"Invalid event type(s): {', '.join(sorted(invalid))}. "
                f"Supported: {', '.join(sorted(WEBHOOK_EVENTS))}"
            )
        if not events:
            raise ValueError("At least one event type is required.")

        # Validate secret
        if not secret or not secret.strip():
            raise ValueError("A non-empty webhook secret is required.")

        # SSRF protection
        try:
            from smartrag.ingest.url_fetcher import is_safe_url
        except ImportError:
            # Minimal fallback: only allow http(s)
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError(f"Unsafe or invalid webhook URL: {url}")
        else:
            if not is_safe_url(url):
                raise ValueError(f"Webhook URL blocked by SSRF policy: {url}")

        webhook_id = uuid.uuid4().hex[:16]
        record: dict[str, Any] = {
            "id": webhook_id,
            "url": url,
            "events": sorted(set(events)),
            "secret": secret,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        webhooks = self._load(tenant_id)
        webhooks.append(record)
        self._save(tenant_id, webhooks)
        logger.info(
            "Registered webhook %s for tenant %s → %s", webhook_id, tenant_id, url
        )
        return record

    def list_webhooks(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return all registered webhooks for a tenant (secrets redacted)."""
        webhooks = self._load(tenant_id)
        return [
            {
                "id": w["id"],
                "url": w["url"],
                "events": w["events"],
                "created": w.get("created", "unknown"),
            }
            for w in webhooks
        ]

    def remove(self, tenant_id: str, webhook_id: str) -> bool:
        """Remove a webhook by ID. Returns True if it existed."""
        webhooks = self._load(tenant_id)
        original_len = len(webhooks)
        webhooks = [w for w in webhooks if w["id"] != webhook_id]
        if len(webhooks) == original_len:
            return False
        self._save(tenant_id, webhooks)
        logger.info("Removed webhook %s for tenant %s", webhook_id, tenant_id)
        return True

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    @staticmethod
    def _sign_payload(payload_bytes: bytes, secret: str) -> str:
        """Compute HMAC-SHA256 signature for a payload."""
        return hmac.new(
            secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()

    def fire(
        self,
        tenant_id: str,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Deliver an event to all matching webhooks.

        Sends HTTP POST with JSON body and an ``X-SmartRAG-Signature``
        header. Retries up to 3 times with exponential backoff on
        failure. Never raises — all errors are logged.
        """
        webhooks = self._load(tenant_id)
        matching = [w for w in webhooks if event in w.get("events", [])]

        if not matching:
            return

        payload: dict[str, Any] = {
            "event": event,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tenant_id": tenant_id,
            "data": data or {},
        }
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        for webhook in matching:
            self._deliver(webhook, payload_bytes)

    def _deliver(self, webhook: dict[str, Any], payload_bytes: bytes) -> None:
        """Deliver payload to a single webhook endpoint with retries."""
        try:
            import httpx
        except ImportError:
            logger.error(
                "httpx is not installed — cannot deliver webhook %s. "
                "Install with: pip install httpx",
                webhook["id"],
            )
            return

        url = webhook["url"]
        signature = self._sign_payload(payload_bytes, webhook["secret"])
        headers = {
            "Content-Type": "application/json",
            "X-SmartRAG-Signature": f"sha256={signature}",
            "User-Agent": "SmartRAG-Webhook/1.0",
        }

        last_error: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(url, content=payload_bytes, headers=headers)
                    resp.raise_for_status()
                logger.debug(
                    "Webhook %s delivered to %s (attempt %d)",
                    webhook["id"],
                    url,
                    attempt,
                )
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Webhook %s delivery attempt %d/%d failed for %s: %s",
                    webhook["id"],
                    attempt,
                    len(_RETRY_DELAYS),
                    url,
                    exc,
                )
                if attempt < len(_RETRY_DELAYS):
                    time.sleep(delay)

        logger.error(
            "Webhook %s delivery permanently failed for %s after %d attempts: %s",
            webhook["id"],
            url,
            len(_RETRY_DELAYS),
            last_error,
        )
