"""FastAPI application factory. Run: uvicorn backend.api.main:create_app --factory"""

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.api.state import build_state
from backend.config import ConfigError, Settings, get_settings
from backend.embeddings.embedding_service import Embedder, get_embedder
from backend.indexing.vector_store import DimensionMismatchError
from backend.monitoring.logging_config import configure_logging
from backend.rag.chain import LLMClient, LLMError

log = logging.getLogger(__name__)


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(
    settings: Settings | None = None, embedder: Embedder | None = None, llm: LLMClient | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="Enterprise AI Knowledge Assistant",
        version="0.1.0",
        description="RAG-based document question answering with source citations.",
    )
    app.state.eaka = build_state(settings, embedder or get_embedder(settings), llm)
    app.include_router(router)
    if not settings.admin_token:
        log.warning("ADMIN_TOKEN is not set: upload/delete/metrics endpoints are unauthenticated")

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            app.state.eaka.metrics.record_request(
                getattr(route, "path", "unmatched"), status, (time.perf_counter() - t0) * 1000
            )

    @app.exception_handler(HTTPException)
    async def http_exc(_: Request, exc: HTTPException):
        d = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
        return _err(exc.status_code, d["code"], d["message"])

    @app.exception_handler(RequestValidationError)
    async def validation_exc(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return _err(422, "validation_error", f"Invalid request ({loc}): {first.get('msg', 'invalid')}")

    @app.exception_handler(ConfigError)
    async def config_exc(_: Request, exc: ConfigError):
        return _err(503, "not_configured", str(exc))

    @app.exception_handler(LLMError)
    async def llm_exc(_: Request, exc: LLMError):
        return _err(502, "llm_unavailable", str(exc))

    @app.exception_handler(DimensionMismatchError)
    async def dim_exc(_: Request, exc: DimensionMismatchError):
        return _err(500, "index_mismatch", str(exc))

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.error("unhandled_error", extra={"fields": {"error_type": type(exc).__name__}})
        return _err(500, "internal_error", "An internal error occurred.")

    return app
