"""FastAPI application factory for SmartRAG."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def create_app(knowledge_dir: str | None = None):
    """Create and configure the SmartRAG FastAPI application.

    Parameters
    ----------
    knowledge_dir:
        Path to the knowledge store directory. Falls back to the
        ``SMARTRAG_KNOWLEDGE_DIR`` env var, then ``./knowledge``.

    Returns
    -------
    FastAPI
        A fully configured application instance ready to be served.
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError(
            "FastAPI is required but not installed. "
            "Install the cloud extras: pip install smartrag[cloud]"
        )

    from smartrag.api.auth import configure_rate_limiter, get_auth_dependency
    from smartrag.api.config import ServerConfig

    config = ServerConfig()

    app = FastAPI(
        title="SmartRAG API",
        version="0.1.0",
        description="REST API for the SmartRAG retrieval engine.",
    )

    # CORS ------------------------------------------------------------------
    origins = config.cors_origins.split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # SmartRAG instance -----------------------------------------------------
    from smartrag.core import SmartRAG

    kdir = knowledge_dir or config.knowledge_dir
    app.state.rag = SmartRAG(kdir)

    # Auth dependency -------------------------------------------------------
    configure_rate_limiter(max_requests=config.rate_limit)
    auth_dep = get_auth_dependency(kdir)

    # Routes ----------------------------------------------------------------
    from smartrag.api.routes import router

    router_with_auth = router
    # Attach auth as a dependency on every route in the v1 router
    router_with_auth.dependencies = [__import__("fastapi").Depends(auth_dep)]
    app.include_router(router_with_auth)

    # Health (no auth required) ---------------------------------------------
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "documents": app.state.rag.stats["document_count"],
        }

    # Tenant manager --------------------------------------------------------
    from smartrag.api.tenants import TenantManager

    app.state.tenant_manager = TenantManager(kdir)

    # Webhook manager -------------------------------------------------------
    from smartrag.api.webhooks import WebhookManager

    app.state.webhook_manager = WebhookManager(kdir)

    # Schema manager --------------------------------------------------------
    from smartrag.api.schemas import SchemaManager

    app.state.schema_manager = SchemaManager(kdir)

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger.info(
        "SmartRAG API ready  knowledge_dir=%s  port=%s", kdir, config.port
    )

    return app


# Module-level app for ``uvicorn smartrag.api.server:app``
app = create_app()
