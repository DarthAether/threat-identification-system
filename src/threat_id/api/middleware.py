"""HTTP middleware and global exception handling.

Provides:
- ``CorrelationIdMiddleware`` — assigns a UUID to every request and threads
  it through structlog via ``correlation_id_var``.
- ``RequestLoggingMiddleware`` — emits a structured log line with method,
  path, status code, and wall-clock duration for every HTTP response.
- ``threat_id_exception_handler`` — converts any :class:`ThreatIdError`
  subclass into a JSON error response with a machine-readable ``code``.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from threat_id.core.exceptions import ThreatIdError
from threat_id.core.logging import correlation_id_var

logger = structlog.get_logger(__name__)


# ── Correlation ID ──────────────────────────────────────────────────────────


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Generates a UUID-4 for each request and propagates it."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        correlation_id_var.set(cid)

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


# ── Request Logging ─────────────────────────────────────────────────────────


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request with method, path, status, and duration."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed_ms, 2),
            correlation_id=correlation_id_var.get(),
        )
        return response


# ── Global Exception Handler ────────────────────────────────────────────────


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the global :class:`ThreatIdError` handler to *app*."""

    @app.exception_handler(ThreatIdError)
    async def _threat_id_exception_handler(
        request: Request,
        exc: ThreatIdError,
    ) -> JSONResponse:
        logger.warning(
            "http.error",
            code=exc.code,
            message=exc.message,
            path=request.url.path,
            correlation_id=correlation_id_var.get(),
        )
        body: dict[str, str | None] = {
            "error": exc.code,
            "message": exc.message,
        }
        if exc.detail:
            body["detail"] = exc.detail
        cid = correlation_id_var.get()
        if cid:
            body["correlation_id"] = cid
        return JSONResponse(status_code=exc.status_code, content=body)
