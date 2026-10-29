"""REST endpoints. Routes are sync so blocking work runs in FastAPI's threadpool."""

import hmac
import json
import time

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    SearchRequest,
    SearchResponse,
    UploadItem,
    UploadResponse,
)
from backend.api.state import AppState
from backend.document_loading.validation import PDFValidationError
from backend.indexing.vector_store import DuplicateDocumentError

router = APIRouter()

_STATUS = {"unsupported_type": 415, "file_too_large": 413, "duplicate": 409}


def get_state(request: Request) -> AppState:
    return request.app.state.eaka


def require_token(state: AppState = Depends(get_state), x_api_token: str | None = Header(None)) -> None:
    """If ADMIN_TOKEN is configured, protected endpoints require a matching X-API-Token."""
    expected = state.settings.admin_token
    if expected and not hmac.compare_digest(x_api_token or "", expected):
        raise HTTPException(401, detail={"code": "unauthorized", "message": "Missing or invalid API token."})


@router.get("/health")
def health(state: AppState = Depends(get_state)) -> dict:
    return {
        "status": "ok",
        "indexed_chunks": state.store.chunk_count,
        "embedder": state.embedder.name,
        "auth_enabled": bool(state.settings.admin_token),
        "llm_configured": bool(state.settings.openai_api_key) or state.llm is not None,
    }


@router.post("/documents/upload", response_model=UploadResponse, status_code=201, dependencies=[Depends(require_token)])
def upload(files: list[UploadFile] = File(...), state: AppState = Depends(get_state)):
    limit = state.settings.max_upload_mb * 1024 * 1024
    items: list[UploadItem] = []
    for f in files:
        name = f.filename or "unnamed"
        data = f.file.read(limit + 1)  # never read more than limit + 1 bytes
        try:
            r = state.ingestion.ingest(name, data)
            items.append(UploadItem(status="indexed", **r))
        except PDFValidationError as e:
            items.append(UploadItem(filename=name, status="error", error_code=e.code, error=str(e)))
        except DuplicateDocumentError as e:
            items.append(UploadItem(filename=name, status="error", error_code="duplicate", error=str(e)))
    if all(i.status == "error" for i in items):
        code = items[0].error_code or "invalid_pdf"
        status = _STATUS.get(code, 422)
        return JSONResponse(
            status_code=status,
            content={"error": {"code": code, "message": items[0].error}, "results": [i.model_dump() for i in items]},
        )
    return UploadResponse(results=items)


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(state: AppState = Depends(get_state)):
    return [DocumentOut(**{k: d[k] for k in DocumentOut.model_fields}) for d in state.store.list_documents()]


@router.delete("/documents/{document_id}", dependencies=[Depends(require_token)])
def delete_document(document_id: str, state: AppState = Depends(get_state)) -> dict:
    if not state.ingestion.delete(document_id):
        raise HTTPException(404, detail={"code": "not_found", "message": "Document not found."})
    return {"deleted": document_id}


def _passages(rows: list[dict]) -> list[dict]:
    return [{k: r[k] for k in ("label", "score", "document_id", "filename", "page", "chunk_id", "text")} for r in rows]


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, state: AppState = Depends(get_state)):
    res = state.retriever.search(body.query, body.top_k)
    state.metrics.observe("retrieval_ms", res.latency_ms)
    state.repo.log_query(body.query, "search", len(res.hits), not res.hits, res.latency_ms)
    rows = [{"label": f"S{i}", "score": round(h.score, 4), **h.chunk} for i, h in enumerate(res.hits, 1)]
    return SearchResponse(results=_passages(rows), retrieval_ms=round(res.latency_ms, 2))


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, state: AppState = Depends(get_state)):
    t0 = time.perf_counter()
    chain = state.get_chain()  # ConfigError -> 503 via handler
    r = chain.answer(body.query, body.top_k)  # LLMError -> 502 via handler
    total = (time.perf_counter() - t0) * 1000
    m = state.metrics
    m.observe("retrieval_ms", r.retrieval_ms)
    if r.llm_ms:
        m.observe("llm_ms", r.llm_ms)
    for name, v in (("prompt_tokens", r.prompt_tokens), ("completion_tokens", r.completion_tokens)):
        if v:
            m.inc(name, v)
    if r.insufficient_context:
        m.inc("insufficient_context")
    state.repo.log_query(body.query, "chat", len(r.retrieved), r.insufficient_context, total)
    return ChatResponse(
        answer=r.answer,
        insufficient_context=r.insufficient_context,
        citations=[c.__dict__ for c in r.citations],
        invalid_citations_removed=r.invalid_citations,
        retrieved=_passages(r.retrieved) if body.include_passages else [],
        retrieval_ms=round(r.retrieval_ms, 2),
        llm_ms=round(r.llm_ms, 2),
        total_ms=round(total, 2),
        prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens,
    )


@router.get("/metrics", dependencies=[Depends(require_token)])
def metrics(state: AppState = Depends(get_state)) -> dict:
    snap = state.metrics.snapshot()
    snap["index"] = {"documents": len(state.store.list_documents()), "chunks": state.store.chunk_count}
    snap["queries_logged"] = state.repo.query_count()
    return snap


@router.get("/evaluation")
def evaluation(state: AppState = Depends(get_state)) -> dict:
    p = state.evaluation_path
    if not p.exists():
        raise HTTPException(
            404,
            detail={
                "code": "no_evaluation",
                "message": "No evaluation results yet. Run: python -m backend.evaluation.run_evaluation",
            },
        )
    return json.loads(p.read_text(encoding="utf-8"))
