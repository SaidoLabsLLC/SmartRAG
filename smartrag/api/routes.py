"""REST API routes for SmartRAG (v1)."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from smartrag.api.auth import AuthResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TextIngestRequest(BaseModel):
    text: str
    title: str
    metadata: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=100)


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1, le=100)
    filters: dict[str, Any] | None = None


class IngestResponse(BaseModel):
    slug: str
    title: str
    status: str
    children: list[str] | None = None
    error: str | None = None


class RetrievalResultResponse(BaseModel):
    slug: str
    title: str
    snippet: str
    score: float
    tier_resolved: int
    categories: list[str] = Field(default_factory=list)
    source_file: str = ""


class QueryResponse(BaseModel):
    results: list[RetrievalResultResponse]
    query: str
    total_ms: float
    total_bytes_read: int


class SearchResultResponse(BaseModel):
    slug: str
    title: str
    summary: str
    score: float
    categories: list[str] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    slug: str
    title: str
    body: str
    frontmatter: dict[str, Any]
    word_count: int
    has_children: bool = False


class DocumentListItem(BaseModel):
    slug: str
    title: str
    summary: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    page: int
    per_page: int
    total: int


class StatsResponse(BaseModel):
    document_count: int
    index_size_bytes: int
    categories: list[str]


class MessageResponse(BaseModel):
    message: str


# --- Webhook models --------------------------------------------------------


class WebhookRegisterRequest(BaseModel):
    url: str
    events: list[str]
    secret: str


class WebhookResponse(BaseModel):
    id: str
    url: str
    events: list[str]
    created: str = ""


# --- Schema models ---------------------------------------------------------


class SchemaFieldRequest(BaseModel):
    field_name: str
    field_type: str
    required: bool = False
    default: Any | None = None
    description: str = ""


class SchemaFieldResponse(BaseModel):
    field_name: str
    field_type: str
    required: bool = False
    default: Any | None = None
    description: str = ""
    updated: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rag(request: Request):
    """Retrieve the SmartRAG instance stored on app.state."""
    return request.app.state.rag


def _get_tenant_rag(request: Request, auth: AuthResult) -> Any:
    """Resolve tenant-scoped SmartRAG instance.

    If the authenticated key has a tenant_id AND the app has a
    TenantManager, use the tenant-scoped instance. Otherwise fall
    back to the global ``app.state.rag``.
    """
    tenant_mgr = getattr(request.app.state, "tenant_manager", None)
    if tenant_mgr and auth.tenant_id:
        return tenant_mgr.get_instance(auth.tenant_id)
    return request.app.state.rag


def _resolve_tenant_id(request: Request, auth: AuthResult) -> str:
    """Return the effective tenant_id, falling back to '__global__'."""
    return auth.tenant_id or "__global__"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(
    request: Request,
    file: UploadFile | None = File(default=None),
    title: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
):
    """Ingest a document via file upload or JSON text body.

    Accepts either:
    - Multipart form with a ``file`` field (and optional ``title``, ``metadata_json``)
    - JSON body with ``{text, title, metadata}`` (detected via Content-Type)
    """
    rag = _rag(request)

    # --- Branch 1: multipart file upload ---
    if file is not None:
        suffix = os.path.splitext(file.filename or "upload.txt")[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            contents = await file.read()
            tmp.write(contents)
            tmp.close()
            result = rag.ingest(tmp.name)
            if isinstance(result, list):
                # Directory ingest returns a list; take the first for the response.
                result = result[0] if result else None
                if result is None:
                    raise HTTPException(status_code=400, detail="No documents ingested.")
            return IngestResponse(
                slug=result.slug,
                title=result.title,
                status=result.status,
                children=result.children,
                error=result.error,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Ingest failed for uploaded file")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # --- Branch 2: JSON text body ---
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file upload or JSON body with {text, title}.",
        )

    parsed = TextIngestRequest(**body)
    try:
        result = rag.ingest_text(
            text=parsed.text,
            title=parsed.title,
            metadata=parsed.metadata,
        )
        return IngestResponse(
            slug=result.slug,
            title=result.title,
            status=result.status,
            children=result.children,
            error=result.error,
        )
    except Exception as exc:
        logger.exception("Ingest failed for text body")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request):
    """Query the knowledge store and return ranked, cited results."""
    rag = _rag(request)
    try:
        qr = rag.query(body.question, top_k=body.top_k)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        results=[
            RetrievalResultResponse(
                slug=r.slug,
                title=r.title,
                snippet=r.snippet,
                score=r.score,
                tier_resolved=r.tier_resolved,
                categories=r.categories,
                source_file=r.source_file,
            )
            for r in qr.results
        ],
        query=qr.query,
        total_ms=qr.total_ms,
        total_bytes_read=qr.total_bytes_read,
    )


@router.post("/search", response_model=list[SearchResultResponse])
async def search(body: SearchRequest, request: Request):
    """Search with optional filters (lower-level than query)."""
    rag = _rag(request)
    try:
        results = rag.search(body.query, top_k=body.top_k, filters=body.filters)
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return [
        SearchResultResponse(
            slug=r.slug,
            title=r.title,
            summary=r.summary,
            score=r.score,
            categories=r.categories,
        )
        for r in results
    ]


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    page: int = 1,
    per_page: int = 20,
):
    """List all documents with pagination."""
    rag = _rag(request)
    all_docs = rag._store.list_all()
    total = len(all_docs)

    # Clamp pagination values
    page = max(1, page)
    per_page = max(1, min(per_page, 100))

    start = (page - 1) * per_page
    end = start + per_page
    page_docs = all_docs[start:end]

    return DocumentListResponse(
        documents=[
            DocumentListItem(slug=slug, title=title, summary=summary)
            for slug, title, summary in page_docs
        ],
        page=page,
        per_page=per_page,
        total=total,
    )


@router.get("/documents/{slug}", response_model=DocumentResponse)
async def get_document(slug: str, request: Request):
    """Get a specific document by slug."""
    rag = _rag(request)
    doc = rag.get(slug)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{slug}' not found.")
    return DocumentResponse(
        slug=doc.slug,
        title=doc.title,
        body=doc.body,
        frontmatter=doc.frontmatter,
        word_count=doc.word_count,
        has_children=doc.has_children,
    )


@router.delete("/documents/{slug}", response_model=MessageResponse)
async def delete_document(slug: str, request: Request):
    """Delete a document by slug."""
    rag = _rag(request)
    success = rag.delete(slug)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{slug}' not found.")
    return MessageResponse(message=f"Document '{slug}' deleted.")


@router.post("/reindex", response_model=MessageResponse)
async def reindex(request: Request):
    """Trigger a full reindex of the knowledge store."""
    rag = _rag(request)
    try:
        count = rag.reindex()
    except Exception as exc:
        logger.exception("Reindex failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return MessageResponse(message=f"Reindexed {count} documents.")


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request):
    """Return knowledge store statistics."""
    rag = _rag(request)
    s = rag.stats
    return StatsResponse(
        document_count=s["document_count"],
        index_size_bytes=s["index_size_bytes"],
        categories=s.get("categories", []),
    )


# ---------------------------------------------------------------------------
# Webhook routes
# ---------------------------------------------------------------------------


@router.post("/webhooks", response_model=WebhookResponse, status_code=201)
async def register_webhook(body: WebhookRegisterRequest, request: Request):
    """Register a webhook for the current tenant."""
    auth: AuthResult = request.state.auth_result
    tenant_id = _resolve_tenant_id(request, auth)

    webhook_mgr = getattr(request.app.state, "webhook_manager", None)
    if webhook_mgr is None:
        raise HTTPException(status_code=501, detail="Webhooks not configured.")

    try:
        record = webhook_mgr.register(
            tenant_id=tenant_id,
            url=body.url,
            events=body.events,
            secret=body.secret,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return WebhookResponse(
        id=record["id"],
        url=record["url"],
        events=record["events"],
        created=record.get("created", ""),
    )


@router.get("/webhooks", response_model=list[WebhookResponse])
async def list_webhooks(request: Request):
    """List webhooks for the current tenant."""
    auth: AuthResult = request.state.auth_result
    tenant_id = _resolve_tenant_id(request, auth)

    webhook_mgr = getattr(request.app.state, "webhook_manager", None)
    if webhook_mgr is None:
        raise HTTPException(status_code=501, detail="Webhooks not configured.")

    webhooks = webhook_mgr.list_webhooks(tenant_id)
    return [
        WebhookResponse(
            id=w["id"],
            url=w["url"],
            events=w["events"],
            created=w.get("created", ""),
        )
        for w in webhooks
    ]


@router.delete("/webhooks/{webhook_id}", response_model=MessageResponse)
async def remove_webhook(webhook_id: str, request: Request):
    """Remove a webhook by ID."""
    auth: AuthResult = request.state.auth_result
    tenant_id = _resolve_tenant_id(request, auth)

    webhook_mgr = getattr(request.app.state, "webhook_manager", None)
    if webhook_mgr is None:
        raise HTTPException(status_code=501, detail="Webhooks not configured.")

    removed = webhook_mgr.remove(tenant_id, webhook_id)
    if not removed:
        raise HTTPException(
            status_code=404, detail=f"Webhook '{webhook_id}' not found."
        )
    return MessageResponse(message=f"Webhook '{webhook_id}' removed.")


# ---------------------------------------------------------------------------
# Custom schema routes
# ---------------------------------------------------------------------------


@router.post("/schema", response_model=SchemaFieldResponse, status_code=201)
async def define_schema_field(body: SchemaFieldRequest, request: Request):
    """Define a custom frontmatter schema field for the current tenant."""
    auth: AuthResult = request.state.auth_result
    tenant_id = _resolve_tenant_id(request, auth)

    schema_mgr = getattr(request.app.state, "schema_manager", None)
    if schema_mgr is None:
        raise HTTPException(
            status_code=501, detail="Custom schemas not configured."
        )

    try:
        result = schema_mgr.define_field(
            tenant_id=tenant_id,
            field_name=body.field_name,
            field_type=body.field_type,
            required=body.required,
            default=body.default,
            description=body.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return SchemaFieldResponse(**result)


@router.get("/schema", response_model=list[SchemaFieldResponse])
async def list_schema_fields(request: Request):
    """List custom frontmatter schema fields for the current tenant."""
    auth: AuthResult = request.state.auth_result
    tenant_id = _resolve_tenant_id(request, auth)

    schema_mgr = getattr(request.app.state, "schema_manager", None)
    if schema_mgr is None:
        raise HTTPException(
            status_code=501, detail="Custom schemas not configured."
        )

    fields = schema_mgr.list_fields(tenant_id)
    return [SchemaFieldResponse(**f) for f in fields]


@router.delete("/schema/{field_name}", response_model=MessageResponse)
async def remove_schema_field(field_name: str, request: Request):
    """Remove a custom frontmatter schema field."""
    auth: AuthResult = request.state.auth_result
    tenant_id = _resolve_tenant_id(request, auth)

    schema_mgr = getattr(request.app.state, "schema_manager", None)
    if schema_mgr is None:
        raise HTTPException(
            status_code=501, detail="Custom schemas not configured."
        )

    removed = schema_mgr.remove_field(tenant_id, field_name)
    if not removed:
        raise HTTPException(
            status_code=404, detail=f"Schema field '{field_name}' not found."
        )
    return MessageResponse(message=f"Schema field '{field_name}' removed.")
