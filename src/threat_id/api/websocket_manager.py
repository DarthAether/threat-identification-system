"""WebSocket connection manager for real-time alert broadcasting.

Maintains a set of active WebSocket connections and provides thread-safe
broadcast / unicast helpers. The ``active_count`` property is exposed for
Prometheus metrics collection.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import structlog
from starlette.websockets import WebSocket, WebSocketState

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections with thread-safe bookkeeping."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock: threading.Lock = threading.Lock()

    # ── Connection lifecycle ────────────────────────────────────────

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        with self._lock:
            self._connections.add(websocket)
        logger.info(
            "ws_manager.connect",
            active=self.active_count,
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from the active set."""
        with self._lock:
            self._connections.discard(websocket)
        logger.info(
            "ws_manager.disconnect",
            active=self.active_count,
        )

    # ── Messaging ───────────────────────────────────────────────────

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send *data* as JSON to every active connection.

        Connections that fail to receive the message are silently removed.
        """
        with self._lock:
            targets = set(self._connections)

        stale: list[WebSocket] = []
        send_tasks = []

        for ws in targets:
            if ws.client_state != WebSocketState.CONNECTED:
                stale.append(ws)
                continue
            send_tasks.append(self._safe_send(ws, data, stale))

        if send_tasks:
            await asyncio.gather(*send_tasks)

        # Prune dead connections
        if stale:
            with self._lock:
                for ws in stale:
                    self._connections.discard(ws)
            logger.debug("ws_manager.pruned", count=len(stale))

    async def send_to(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        """Send *data* as JSON to a single connection."""
        try:
            await websocket.send_json(data)
        except Exception:
            logger.warning("ws_manager.send_to_failed", exc_info=True)
            self.disconnect(websocket)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        """Number of currently tracked connections."""
        with self._lock:
            return len(self._connections)

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    async def _safe_send(
        ws: WebSocket,
        data: dict[str, Any],
        stale: list[WebSocket],
    ) -> None:
        try:
            await ws.send_json(data)
        except Exception:
            stale.append(ws)
