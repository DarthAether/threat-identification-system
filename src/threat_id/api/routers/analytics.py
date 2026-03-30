"""Analytics and audit endpoints.

- ``GET /v1/analytics/dashboard``  — summary stats for the dashboard.
- ``GET /v1/analytics/detections`` — detection counts by label for a time window.
- ``GET /v1/analytics/audit``      — query the audit log with filters.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from threat_id.api.dependencies import (
    get_camera_manager,
    get_db_session,
    require_role,
)
from threat_id.camera.manager import CameraManager
from threat_id.core.security import Role
from threat_id.db.repositories import AlertRepository, AuditRepository, DetectionRepository

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ── Response schemas ────────────────────────────────────────────────────────


class DashboardResponse(BaseModel):
    threats_today: int
    total_detections_today: int
    active_cameras: int
    alerts_by_severity: dict[str, int]
    total_unacknowledged_alerts: int


class DetectionStatsResponse(BaseModel):
    time_min: str
    time_max: str
    counts_by_label: dict[str, int]
    total: int


class AuditEntry(BaseModel):
    id: int
    timestamp: str
    event_type: str
    camera_id: str | None
    payload: dict[str, Any] | None
    correlation_id: str | None


class AuditResponse(BaseModel):
    entries: list[AuditEntry]
    total: int


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Dashboard summary statistics",
    dependencies=[Depends(require_role(Role.VIEWER, Role.OPERATOR, Role.ADMIN))],
)
async def dashboard(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    camera_manager: Annotated[CameraManager, Depends(get_camera_manager)],
) -> DashboardResponse:
    """Aggregate statistics for the main dashboard view."""
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    detection_repo = DetectionRepository(session)
    alert_repo = AlertRepository(session)

    # Detection counts for today
    detection_stats = await detection_repo.get_stats(
        time_min=start_of_day,
        time_max=now,
    )
    total_detections = sum(detection_stats.values())

    # Threat detections only (labels matching threat categories)
    # We count any detection as a "threat" that was recorded with is_threat=True
    from sqlalchemy import func, select  # noqa: PLC0415
    from threat_id.db.models import DetectionRecord  # noqa: PLC0415

    stmt = (
        select(func.count(DetectionRecord.id))
        .where(DetectionRecord.timestamp >= start_of_day)
        .where(DetectionRecord.is_threat.is_(True))
    )
    result = await session.execute(stmt)
    threats_today = result.scalar() or 0

    # Alert stats
    alerts_by_severity = await alert_repo.count_by_severity()
    total_unacked = sum(alerts_by_severity.values())

    # Active cameras
    active_cameras = len(camera_manager.list_cameras())

    return DashboardResponse(
        threats_today=threats_today,
        total_detections_today=total_detections,
        active_cameras=active_cameras,
        alerts_by_severity=alerts_by_severity,
        total_unacknowledged_alerts=total_unacked,
    )


@router.get(
    "/detections",
    response_model=DetectionStatsResponse,
    summary="Detection statistics for a time window",
    dependencies=[Depends(require_role(Role.VIEWER, Role.OPERATOR, Role.ADMIN))],
)
async def detection_stats(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    hours: int = Query(24, ge=1, le=720, description="Lookback window in hours"),
    camera_id: str | None = Query(None, description="Filter by camera"),
) -> DetectionStatsResponse:
    """Return detection counts grouped by label for the specified window."""
    now = datetime.now(timezone.utc)
    time_min = now - timedelta(hours=hours)

    repo = DetectionRepository(session)
    counts = await repo.get_stats(
        time_min=time_min,
        time_max=now,
        camera_id=camera_id,
    )

    return DetectionStatsResponse(
        time_min=time_min.isoformat(),
        time_max=now.isoformat(),
        counts_by_label=counts,
        total=sum(counts.values()),
    )


@router.get(
    "/audit",
    response_model=AuditResponse,
    summary="Query audit log",
    dependencies=[Depends(require_role(Role.ADMIN))],
)
async def audit_log(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    event_type: str | None = Query(None, description="Filter by event type"),
    camera_id: str | None = Query(None, description="Filter by camera ID"),
    hours: int | None = Query(None, ge=1, le=8760, description="Lookback window in hours"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> AuditResponse:
    """Query the audit log with optional time range and event type filters."""
    now = datetime.now(timezone.utc)
    time_min = (now - timedelta(hours=hours)) if hours else None

    repo = AuditRepository(session)
    entries = await repo.query(
        time_min=time_min,
        time_max=now if hours else None,
        event_type=event_type,
        camera_id=camera_id,
        limit=limit,
        offset=offset,
    )

    items = [
        AuditEntry(
            id=e.id,
            timestamp=e.timestamp.isoformat(),
            event_type=e.event_type,
            camera_id=e.camera_id,
            payload=e.payload,
            correlation_id=e.correlation_id,
        )
        for e in entries
    ]

    return AuditResponse(entries=items, total=len(items))
