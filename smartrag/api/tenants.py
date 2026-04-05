"""Tenant management for multi-tenant SmartRAG deployments."""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)


class TenantManager:
    """Manages per-tenant SmartRAG instances with LRU eviction.

    Each tenant gets an isolated SmartRAG instance backed by its own
    subdirectory under ``{base_dir}/tenants/{tenant_id}/``.
    """

    def __init__(self, base_dir: str, max_cached: int = 100):
        self._base_dir = base_dir
        self._max_cached = max_cached
        self._instances: dict[str, "SmartRAG"] = {}
        self._lock = threading.Lock()

    def get_instance(self, tenant_id: str) -> "SmartRAG":
        """Get or create a SmartRAG instance for a tenant.

        Instances are cached in memory up to ``max_cached``. When the
        cache is full the oldest entry (FIFO insertion order) is evicted.
        """
        with self._lock:
            # Move to end on access for pseudo-LRU behavior
            if tenant_id in self._instances:
                instance = self._instances.pop(tenant_id)
                self._instances[tenant_id] = instance
                return instance

            # Evict oldest if at capacity
            if len(self._instances) >= self._max_cached:
                oldest = next(iter(self._instances))
                logger.info("Evicting cached tenant instance: %s", oldest)
                del self._instances[oldest]

            tenant_dir = self._tenant_dir(tenant_id)
            os.makedirs(tenant_dir, exist_ok=True)

            from smartrag.core import SmartRAG

            instance = SmartRAG(tenant_dir)
            self._instances[tenant_id] = instance
            logger.info("Created SmartRAG instance for tenant: %s", tenant_id)
            return instance

    def _tenant_dir(self, tenant_id: str) -> str:
        """Return the filesystem path for a tenant's data directory."""
        return os.path.join(self._base_dir, "tenants", tenant_id)

    def list_tenants(self) -> list[str]:
        """Return a list of tenant IDs that have data directories on disk."""
        tenants_dir = os.path.join(self._base_dir, "tenants")
        if not os.path.isdir(tenants_dir):
            return []
        return sorted(
            d
            for d in os.listdir(tenants_dir)
            if os.path.isdir(os.path.join(tenants_dir, d))
        )

    def evict(self, tenant_id: str) -> bool:
        """Remove a tenant instance from the cache. Returns True if evicted."""
        with self._lock:
            if tenant_id in self._instances:
                del self._instances[tenant_id]
                return True
            return False

    @property
    def cached_count(self) -> int:
        """Number of currently cached tenant instances."""
        return len(self._instances)
