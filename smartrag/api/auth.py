"""API key authentication, rate limiting, and tenant resolution."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth result type
# ---------------------------------------------------------------------------

VALID_PERMISSIONS = {"admin", "read-write", "read-only"}

# HTTP methods considered read-only for permission enforcement
_READ_ONLY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@dataclass
class AuthResult:
    """Resolved identity from API key verification."""

    key_name: str | None
    tenant_id: str | None
    permissions: str  # "admin" | "read-write" | "read-only"

# ---------------------------------------------------------------------------
# Key store helpers
# ---------------------------------------------------------------------------

_DEFAULT_KEYS_PATH = os.path.join(".smartrag", "api_keys.json")


def _keys_path(knowledge_dir: str) -> Path:
    return Path(knowledge_dir) / _DEFAULT_KEYS_PATH


def _load_keys(knowledge_dir: str) -> list[dict[str, Any]]:
    path = _keys_path(knowledge_dir)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save_keys(knowledge_dir: str, keys: list[dict[str, Any]]) -> None:
    path = _keys_path(knowledge_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2)


def _hash_key(raw_key: str) -> str:
    """Hash an API key using bcrypt.

    Falls back to SHA-256 when bcrypt is not installed so the module
    remains importable outside the ``[cloud]`` dependency group.
    """
    try:
        import bcrypt

        return bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        return "sha256:" + hashlib.sha256(raw_key.encode()).hexdigest()


def _verify_key(raw_key: str, stored_hash: str) -> bool:
    """Verify a raw API key against a stored hash."""
    if stored_hash.startswith("sha256:"):
        return (
            "sha256:" + hashlib.sha256(raw_key.encode()).hexdigest() == stored_hash
        )
    try:
        import bcrypt

        return bcrypt.checkpw(raw_key.encode(), stored_hash.encode())
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Key management (used by CLI)
# ---------------------------------------------------------------------------


def create_api_key(
    knowledge_dir: str,
    name: str,
    tenant_id: str | None = None,
    permissions: str = "read-write",
) -> str:
    """Create a new API key. Returns the raw key (shown only once).

    Parameters
    ----------
    knowledge_dir:
        Root knowledge directory.
    name:
        Human-readable name for the key.
    tenant_id:
        Tenant this key is scoped to. ``None`` means global / legacy.
    permissions:
        Permission level — one of ``"admin"``, ``"read-write"``,
        ``"read-only"``.
    """
    if permissions not in VALID_PERMISSIONS:
        raise ValueError(
            f"Invalid permissions '{permissions}'. "
            f"Must be one of: {', '.join(sorted(VALID_PERMISSIONS))}"
        )

    keys = _load_keys(knowledge_dir)

    # Prevent duplicate names
    for k in keys:
        if k["name"] == name:
            raise ValueError(f"API key with name '{name}' already exists")

    raw_key = "srag_" + secrets.token_urlsafe(32)
    entry: dict[str, Any] = {
        "name": name,
        "key_hash": _hash_key(raw_key),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "permissions": permissions,
        "tenant_id": tenant_id,
    }
    keys.append(entry)
    _save_keys(knowledge_dir, keys)
    return raw_key


def list_api_keys(knowledge_dir: str) -> list[dict[str, str]]:
    """List key names, creation dates, and tenant associations (never hashes)."""
    keys = _load_keys(knowledge_dir)
    return [
        {
            "name": k["name"],
            "created": k.get("created", "unknown"),
            "tenant_id": k.get("tenant_id", ""),
            "permissions": k.get("permissions", "read-write"),
        }
        for k in keys
    ]


def revoke_api_key(knowledge_dir: str, name: str) -> bool:
    """Revoke (delete) an API key by name. Returns True if found."""
    keys = _load_keys(knowledge_dir)
    original_len = len(keys)
    keys = [k for k in keys if k["name"] != name]
    if len(keys) == original_len:
        return False
    _save_keys(knowledge_dir, keys)
    return True


# ---------------------------------------------------------------------------
# Rate limiter (in-process, per-key, per-minute sliding window)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        # key_identifier -> list of timestamps
        self._buckets: dict[str, list[float]] = {}

    def is_allowed(self, key_id: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets.setdefault(key_id, [])
        # Evict expired entries
        cutoff = now - self._window
        bucket[:] = [t for t in bucket if t > cutoff]
        if len(bucket) >= self._max:
            return False
        bucket.append(now)
        return True


# Module-level limiter instance; reconfigured when the FastAPI app boots.
_limiter = _RateLimiter()


def configure_rate_limiter(max_requests: int = 100, window_seconds: int = 60) -> None:
    """Reconfigure the global rate limiter (called at app startup)."""
    global _limiter
    _limiter = _RateLimiter(max_requests, window_seconds)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_auth_dependency(knowledge_dir: str):
    """Return a FastAPI dependency that validates X-API-Key.

    If no api_keys.json exists or it is empty, authentication is disabled
    and all requests pass through (open access for easy development).

    Returns an ``AuthResult`` with key_name, tenant_id, and permissions.
    """
    from fastapi import HTTPException, Request, Security
    from fastapi.security import APIKeyHeader

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def verify_api_key(
        request: Request,
        api_key: str | None = Security(api_key_header),
    ) -> AuthResult:
        keys = _load_keys(knowledge_dir)

        # Open access when no keys are configured
        if not keys:
            result = AuthResult(key_name=None, tenant_id=None, permissions="admin")
            request.state.auth_result = result
            return result

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Provide X-API-Key header.",
            )

        # Verify against stored hashes
        matched_entry: dict[str, Any] | None = None
        for entry in keys:
            if _verify_key(api_key, entry["key_hash"]):
                matched_entry = entry
                break

        if matched_entry is None:
            raise HTTPException(status_code=403, detail="Invalid API key.")

        matched_name = matched_entry["name"]

        # Rate limiting
        if not _limiter.is_allowed(matched_name):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
            )

        # Resolve permissions — support both old list format and new string
        raw_perms = matched_entry.get("permissions", "read-write")
        if isinstance(raw_perms, list):
            # Legacy format: ["*"] → admin, otherwise read-write
            perms = "admin" if "*" in raw_perms else "read-write"
        else:
            perms = raw_perms if raw_perms in VALID_PERMISSIONS else "read-write"

        # Permission enforcement: read-only keys cannot use mutating methods
        if perms == "read-only" and request.method not in _READ_ONLY_METHODS:
            raise HTTPException(
                status_code=403,
                detail="Read-only API key cannot perform this operation.",
            )

        tenant_id = matched_entry.get("tenant_id")

        result = AuthResult(
            key_name=matched_name,
            tenant_id=tenant_id,
            permissions=perms,
        )

        # Stash on request.state so route handlers can access it
        request.state.auth_result = result
        return result

    return verify_api_key
