"""Alerting domain models.

Defines severity levels, alert channels, and the payload structures
used throughout the alerting subsystem.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    """Threat severity levels ordered by escalation priority."""

    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(StrEnum):
    """Supported alert delivery channels."""

    EMAIL = "email"
    SOUND = "sound"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"


class AlertPayload(BaseModel):
    """Canonical alert payload dispatched to every enabled channel."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    camera_id: str
    threat_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    correlation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    model_config = {"frozen": True}


class AlertHistoryItem(BaseModel):
    """Persisted alert record returned by the API."""

    id: str
    timestamp: datetime
    camera_id: str
    threat_label: str
    confidence: float
    severity: Severity
    bbox: tuple[int, int, int, int]
    correlation_id: str
    channels_notified: list[str] = Field(default_factory=list)
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
