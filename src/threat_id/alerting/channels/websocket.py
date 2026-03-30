"""WebSocket alert channel — broadcasts alerts to connected clients.

This channel never raises exceptions.  Delivery errors are logged and
swallowed so that a broken WebSocket connection cannot prevent other
channels from firing.
"""

from __future__ import annotations

from typing import Any, Protocol

import structlog

from threat_id.alerting.models import AlertPayload

logger = structlog.get_logger(__name__)


class ConnectionManager(Protocol):
    """Minimal interface expected from the WebSocket connection manager."""

    async def broadcast(self, data: dict[str, Any]) -> None: ...


class WebSocketAlertChannel:
    """Pushes alert payloads to all connected WebSocket clients."""

    def __init__(self, connection_manager: ConnectionManager) -> None:
        self._manager = connection_manager

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "websocket"

    @property
    def is_available(self) -> bool:
        return self._manager is not None

    # ── Public API ───────────────────────────────────────────────────

    async def send(self, payload: AlertPayload) -> None:
        """Broadcast the alert to all connected WebSocket clients.

        This method never raises.  Any error is logged and silently
        discarded to ensure other channels are not blocked.
        """
        try:
            data = payload.model_dump(mode="json")
            await self._manager.broadcast(data)
            logger.info(
                "websocket_alert.broadcast",
                camera_id=payload.camera_id,
                threat=payload.threat_label,
            )
        except Exception:
            logger.exception(
                "websocket_alert.broadcast_failed",
                camera_id=payload.camera_id,
                threat=payload.threat_label,
            )
