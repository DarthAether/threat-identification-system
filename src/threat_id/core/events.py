"""Typed in-process async event bus.

Decouples detection from alerting: detection services publish events,
alerting services subscribe — neither imports the other.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger(__name__)

Subscriber = Callable[..., Coroutine[Any, Any, None]]


# ── Event Types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BaseEvent:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ThreatDetectedEvent(BaseEvent):
    camera_id: str = ""
    label: str = ""
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class FaceRecognizedEvent(BaseEvent):
    camera_id: str = ""
    identity: str = ""
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class AlertFiredEvent(BaseEvent):
    channel: str = ""
    threat_label: str = ""
    success: bool = True


@dataclass(frozen=True, slots=True)
class CameraStatusEvent(BaseEvent):
    camera_id: str = ""
    status: str = ""  # "connected" | "disconnected" | "error"


# ── Event Bus ────────────────────────────────────────────────────────────────

class EventBus:
    """Simple typed pub/sub. All dispatch is fire-and-forget async."""

    def __init__(self) -> None:
        self._subscribers: dict[type[BaseEvent], list[Subscriber]] = defaultdict(list)

    def subscribe(self, event_type: type[BaseEvent], handler: Subscriber) -> None:
        self._subscribers[event_type].append(handler)
        logger.debug("event_bus.subscribe", event_type=event_type.__name__)

    def unsubscribe(self, event_type: type[BaseEvent], handler: Subscriber) -> None:
        self._subscribers[event_type] = [
            h for h in self._subscribers[event_type] if h is not handler
        ]

    async def publish(self, event: BaseEvent) -> None:
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        if not handlers:
            return

        logger.debug("event_bus.publish", event_type=event_type.__name__, handlers=len(handlers))

        results = await asyncio.gather(
            *(h(event) for h in handlers),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.error("event_bus.handler_error", error=str(result), event=event_type.__name__)
