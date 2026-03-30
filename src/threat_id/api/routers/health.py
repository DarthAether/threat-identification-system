"""Health-check endpoints for Kubernetes probes.

- ``GET /health``   -- liveness (always 200 if the process is running)
- ``GET /ready``    -- readiness (checks DB connectivity, model loaded)
- ``GET /startup``  -- startup probe (same as readiness)
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    """Return 200 as long as the process is alive."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def readiness(request: Request) -> dict[str, Any]:
    """Check that critical subsystems are operational.

    Returns 200 when all checks pass, 503 otherwise.
    """
    checks: dict[str, bool] = {}

    # Database connectivity
    try:
        from threat_id.db.engine import get_session  # noqa: PLC0415

        async for session in get_session():
            await session.execute(
                __import__("sqlalchemy").text("SELECT 1"),
            )
            checks["database"] = True
    except Exception:
        checks["database"] = False
        logger.warning("health.db_check_failed", exc_info=True)

    # Detection model loaded
    detection_svc = getattr(request.app.state, "detection_service", None)
    checks["detection_model"] = detection_svc is not None

    # Health checker (if available)
    health_checker = getattr(request.app.state, "health_checker", None)
    if health_checker is not None:
        try:
            hc_status = await health_checker.check() if hasattr(health_checker, "check") else True
            checks["health_checker"] = bool(hc_status)
        except Exception:
            checks["health_checker"] = False

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    from starlette.responses import JSONResponse  # noqa: PLC0415

    return JSONResponse(  # type: ignore[return-value]
        status_code=status_code,
        content={
            "status": "ok" if all_healthy else "degraded",
            "checks": checks,
        },
    )


@router.get("/startup", summary="Startup probe")
async def startup(request: Request) -> dict[str, str]:
    """Return 200 once the application has finished initialising.

    Checks that the database engine is set on ``app.state``.
    """
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        from starlette.responses import JSONResponse  # noqa: PLC0415

        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content={"status": "starting"},
        )
    return {"status": "ok"}
