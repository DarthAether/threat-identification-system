"""Health check system exposing liveness, readiness, and startup probes.

Designed for Kubernetes-style deployments where each probe serves a
different purpose:

- **Liveness** (``/health``): Is the process alive?  Fails only on
  catastrophic internal error.
- **Readiness** (``/ready``): Can the service accept traffic?  Fails when
  critical dependencies (database, model) are unavailable.
- **Startup** (``/startup``): Has the service finished initialising?  Fails
  until all one-time setup (model load, DB migration check) is complete.

All endpoints return a consistent JSON schema::

    {
        "status": "healthy" | "degraded" | "unhealthy",
        "checks": { "<name>": { "status": "...", "detail": "..." } },
        "version": "1.0.0"
    }
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = structlog.get_logger(__name__)

APP_VERSION = "1.0.0"


# ── Types ─────────────────────────────────────────────────────────────────────


class Status(str, Enum):
    """Aggregate or per-check health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(slots=True)
class CheckResult:
    """Outcome of a single health check."""

    status: Status
    detail: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": self.status.value}
        if self.detail:
            result["detail"] = self.detail
        if self.latency_ms > 0:
            result["latency_ms"] = round(self.latency_ms, 2)
        return result


@dataclass(slots=True)
class HealthReport:
    """Aggregated health report returned from every endpoint."""

    status: Status
    checks: dict[str, CheckResult] = field(default_factory=dict)
    version: str = APP_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
            "version": self.version,
        }

    @property
    def http_status(self) -> int:
        if self.status == Status.HEALTHY:
            return 200
        if self.status == Status.DEGRADED:
            return 200
        return 503


def _aggregate_status(checks: dict[str, CheckResult]) -> Status:
    """Derive an aggregate status from individual check results."""
    if not checks:
        return Status.HEALTHY
    statuses = {c.status for c in checks.values()}
    if Status.UNHEALTHY in statuses:
        return Status.UNHEALTHY
    if Status.DEGRADED in statuses:
        return Status.DEGRADED
    return Status.HEALTHY


# ── Health Checker ────────────────────────────────────────────────────────────


class HealthChecker:
    """Runs dependency health checks and builds reports.

    Parameters
    ----------
    session_factory:
        Async session maker for the database check.
    redis_url:
        Redis connection URL for the Redis check.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        redis_url: str | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._redis_url = redis_url

        # Mutable flags set externally as the application boots up.
        self._model_loaded: bool = False
        self._startup_complete: bool = False
        self._camera_statuses: dict[str, str] = {}

    # -- External setters ----------------------------------------------------

    def set_model_loaded(self, loaded: bool) -> None:
        """Mark whether the detection model has been loaded."""
        self._model_loaded = loaded

    def set_startup_complete(self, complete: bool) -> None:
        """Mark overall application startup as finished."""
        self._startup_complete = complete

    def update_camera_status(self, camera_id: str, status: str) -> None:
        """Update the known status for *camera_id*.

        Parameters
        ----------
        status:
            One of ``"connected"``, ``"disconnected"``, ``"error"``.
        """
        self._camera_statuses[camera_id] = status

    def remove_camera(self, camera_id: str) -> None:
        """Remove a camera from the tracked set (e.g. on decommission)."""
        self._camera_statuses.pop(camera_id, None)

    # -- Individual Checks ---------------------------------------------------

    async def check_database(self) -> CheckResult:
        """Execute ``SELECT 1`` against the database."""
        if self._session_factory is None:
            return CheckResult(status=Status.UNHEALTHY, detail="session_factory not configured")

        t0 = time.monotonic()
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            elapsed = (time.monotonic() - t0) * 1_000
            return CheckResult(status=Status.HEALTHY, detail="ok", latency_ms=elapsed)
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1_000
            logger.warning("health.database_check_failed", error=str(exc))
            return CheckResult(
                status=Status.UNHEALTHY,
                detail=f"connection error: {exc}",
                latency_ms=elapsed,
            )

    async def check_redis(self) -> CheckResult:
        """Send ``PING`` to Redis."""
        if not self._redis_url:
            return CheckResult(status=Status.UNHEALTHY, detail="redis_url not configured")

        t0 = time.monotonic()
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(self._redis_url, decode_responses=True)
            try:
                pong: str = await client.ping()  # type: ignore[assignment]
                elapsed = (time.monotonic() - t0) * 1_000
                if pong:
                    return CheckResult(status=Status.HEALTHY, detail="pong", latency_ms=elapsed)
                return CheckResult(status=Status.UNHEALTHY, detail="no response", latency_ms=elapsed)
            finally:
                await client.aclose()
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1_000
            logger.warning("health.redis_check_failed", error=str(exc))
            return CheckResult(
                status=Status.UNHEALTHY,
                detail=f"connection error: {exc}",
                latency_ms=elapsed,
            )

    async def check_model_loaded(self) -> CheckResult:
        """Check whether the detection model is loaded and ready."""
        if self._model_loaded:
            return CheckResult(status=Status.HEALTHY, detail="model loaded")
        return CheckResult(status=Status.UNHEALTHY, detail="model not loaded")

    async def check_camera_status(self) -> CheckResult:
        """Aggregate camera health across all tracked cameras."""
        if not self._camera_statuses:
            return CheckResult(status=Status.HEALTHY, detail="no cameras registered")

        total = len(self._camera_statuses)
        connected = sum(
            1 for s in self._camera_statuses.values() if s == "connected"
        )
        errored = sum(
            1 for s in self._camera_statuses.values() if s in ("error", "disconnected")
        )

        detail = f"{connected}/{total} connected"

        if errored == total:
            return CheckResult(status=Status.UNHEALTHY, detail=detail)
        if errored > 0:
            return CheckResult(status=Status.DEGRADED, detail=detail)
        return CheckResult(status=Status.HEALTHY, detail=detail)

    # -- Probe Builders ------------------------------------------------------

    async def liveness(self) -> HealthReport:
        """Liveness probe: process is alive and not deadlocked.

        Only fails on catastrophic errors.  Keeping this lightweight
        avoids false restarts.
        """
        checks: dict[str, CheckResult] = {
            "process": CheckResult(status=Status.HEALTHY, detail="alive"),
        }
        return HealthReport(status=_aggregate_status(checks), checks=checks)

    async def readiness(self) -> HealthReport:
        """Readiness probe: service can accept traffic.

        Checks critical dependencies: database, Redis, model.
        """
        checks: dict[str, CheckResult] = {}
        checks["database"] = await self.check_database()
        checks["redis"] = await self.check_redis()
        checks["model_loaded"] = await self.check_model_loaded()
        checks["camera_status"] = await self.check_camera_status()

        return HealthReport(status=_aggregate_status(checks), checks=checks)

    async def startup(self) -> HealthReport:
        """Startup probe: application has completed initialisation.

        Returns unhealthy until :meth:`set_startup_complete` is called
        with ``True``.
        """
        checks: dict[str, CheckResult] = {}

        if self._startup_complete:
            checks["startup"] = CheckResult(status=Status.HEALTHY, detail="complete")
        else:
            checks["startup"] = CheckResult(status=Status.UNHEALTHY, detail="initialising")

        checks["database"] = await self.check_database()
        checks["model_loaded"] = await self.check_model_loaded()

        return HealthReport(status=_aggregate_status(checks), checks=checks)

    # -- Starlette Routes ----------------------------------------------------

    def routes(self) -> list[Route]:
        """Return Starlette :class:`Route` objects for all health endpoints.

        Usage::

            from starlette.routing import Mount
            app.mount("/", Mount(routes=health_checker.routes()))
        """

        async def _liveness(request: Request) -> JSONResponse:
            report = await self.liveness()
            return JSONResponse(report.to_dict(), status_code=report.http_status)

        async def _readiness(request: Request) -> JSONResponse:
            report = await self.readiness()
            return JSONResponse(report.to_dict(), status_code=report.http_status)

        async def _startup(request: Request) -> JSONResponse:
            report = await self.startup()
            return JSONResponse(report.to_dict(), status_code=report.http_status)

        return [
            Route("/health", endpoint=_liveness, methods=["GET"]),
            Route("/ready", endpoint=_readiness, methods=["GET"]),
            Route("/startup", endpoint=_startup, methods=["GET"]),
        ]
