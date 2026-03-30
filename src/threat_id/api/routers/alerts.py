"""Alert endpoints — CRUD, acknowledgement, stats, and WebSocket stream.

- ``GET    /v1/alerts``              — paginated alert list with filters.
- ``PATCH  /v1/alerts/{id}/acknowledge`` — mark an alert as acknowledged.
- ``GET    /v1/alerts/stats``        — alert counts grouped by severity.
- ``WS     /ws/alerts``              — real-time alert stream (JWT via query).
"""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from threat_id.api.dependencies import (
    get_current_user,
    get_db_session,
    get_settings,
    get_ws_manager,
    require_role,
)
from threat_id.api.websocket_manager import ConnectionManager
from threat_id.core.config import Settings
from threat_id.core.exceptions import AuthError
from threat_id.core.security import Role, TokenPayload, decode_token
from threat_id.db.repositories import AlertRepository

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["alerts"])


# ── Response schemas ────────────────────────────────────────────────────────


class AlertItem(BaseModel):
    id: int
    created_at: str
    camera_id: str
    severity: str
    threat_label: str
    confidence: float
    bbox: dict[str, Any] | None = None
    acknowledged: bool
    acknowledged_by: str | None = None
    acknowledged_at: str | None = None


class AlertListResponse(BaseModel):
    alerts: list[AlertItem]
    total: int


class AlertStatsResponse(BaseModel):
    counts_by_severity: dict[str, int]
    total_unacknowledged: int


# ── REST Endpoints ──────────────────────────────────────────────────────────


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="List recent alerts",
    dependencies=[Depends(require_role(Role.VIEWER, Role.OPERATOR, Role.ADMIN))],
)
async def list_alerts(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    severity: str | None = Query(None, description="Filter by severity level"),
    acknowledged: bool | None = Query(None, description="Filter by acknowledgement status"),
) -> AlertListResponse:
    """Return a paginated list of alerts, optionally filtered by severity."""
    repo = AlertRepository(session)
    records = await repo.list_recent(
        limit=limit,
        offset=offset,
        acknowledged=acknowledged,
    )

    # Apply severity filter in-memory (repo doesn't expose it directly)
    if severity is not None:
        records = [r for r in records if r.severity == severity]

    items = [
        AlertItem(
            id=r.id,
            created_at=r.created_at.isoformat(),
            camera_id=r.camera_id,
            severity=r.severity,
            threat_label=r.threat_label,
            confidence=r.confidence,
            bbox=r.bbox_json,
            acknowledged=r.acknowledged,
            acknowledged_by=r.acknowledged_by,
            acknowledged_at=r.acknowledged_at.isoformat() if r.acknowledged_at else None,
        )
        for r in records
    ]

    return AlertListResponse(alerts=items, total=len(items))


@router.patch(
    "/alerts/{alert_id}/acknowledge",
    summary="Acknowledge an alert",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def acknowledge_alert(
    alert_id: int,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
) -> dict[str, Any]:
    """Mark an alert as acknowledged by the current user."""
    repo = AlertRepository(session)
    record = await repo.acknowledge(alert_id, username=current_user.sub)

    if record is None:
        from threat_id.core.exceptions import ThreatIdError  # noqa: PLC0415

        class AlertNotFoundError(ThreatIdError):
            code = "ALERT_NOT_FOUND"
            status_code = 404

        raise AlertNotFoundError(f"Alert {alert_id} not found")

    logger.info(
        "alerts.acknowledged",
        alert_id=alert_id,
        user=current_user.sub,
    )

    return {
        "message": f"Alert {alert_id} acknowledged",
        "acknowledged_by": current_user.sub,
        "acknowledged_at": record.acknowledged_at.isoformat() if record.acknowledged_at else None,
    }


@router.get(
    "/alerts/stats",
    response_model=AlertStatsResponse,
    summary="Alert statistics by severity",
    dependencies=[Depends(require_role(Role.VIEWER, Role.OPERATOR, Role.ADMIN))],
)
async def alert_stats(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AlertStatsResponse:
    """Return unacknowledged alert counts grouped by severity."""
    repo = AlertRepository(session)
    counts = await repo.count_by_severity()
    total = sum(counts.values())

    return AlertStatsResponse(
        counts_by_severity=counts,
        total_unacknowledged=total,
    )


# ── WebSocket ───────────────────────────────────────────────────────────────


@router.websocket("/ws/alerts")
async def alerts_ws(
    websocket: WebSocket,
    token: str | None = Query(None),
) -> None:
    """Real-time alert stream via WebSocket.

    Authentication is performed via the ``token`` query parameter since
    browsers cannot set Authorization headers on WebSocket connections.
    """
    if not token:
        await websocket.close(code=4001, reason="Missing token query parameter")
        return

    # Validate JWT
    settings: Settings = websocket.app.state.settings
    try:
        payload = decode_token(token, settings.jwt)
        if payload.token_type != "access":
            await websocket.close(code=4001, reason="Invalid token type")
            return
    except AuthError:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    # Connect
    ws_manager: ConnectionManager = websocket.app.state.ws_manager
    await ws_manager.connect(websocket)

    logger.info("alerts.ws_connected", user=payload.sub)

    try:
        # Keep connection alive — wait for client messages (pings)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("alerts.ws_disconnected", user=payload.sub)
    except Exception:
        ws_manager.disconnect(websocket)
        logger.warning("alerts.ws_error", user=payload.sub, exc_info=True)
