"""Tests for threat_id.observability.audit — AuditLogger persistence."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from threat_id.core.config import ObservabilitySettings
from threat_id.core.events import (
    AlertFiredEvent,
    EventBus,
    FaceRecognizedEvent,
    ThreatDetectedEvent,
)
from threat_id.db.models import AuditLogEntry
from threat_id.observability.audit import AuditLogger


def _make_session_factory(session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    """Create a mock session factory that yields the given session."""

    class _FakeSessionFactory:
        def __call__(self) -> Any:
            return self

        async def __aenter__(self) -> AsyncSession:
            return session

        async def __aexit__(self, *args: Any) -> None:
            pass

    return _FakeSessionFactory()  # type: ignore[return-value]


class TestAuditLoggerPersistence:
    """AuditLogger persists events to the database."""

    async def test_threat_event_persisted(self, event_bus: EventBus, db_session: AsyncSession) -> None:
        settings = ObservabilitySettings(audit_log_enabled=True)
        factory = _make_session_factory(db_session)
        logger = AuditLogger(event_bus, factory, settings)
        logger.start()

        event = ThreatDetectedEvent(
            camera_id="cam-1",
            label="knife",
            confidence=0.9,
            correlation_id="corr-123",
        )
        await event_bus.publish(event)

        result = await db_session.execute(select(AuditLogEntry))
        entries = result.scalars().all()

        assert len(entries) >= 1
        entry = entries[0]
        assert entry.event_type == "threat_detected"
        assert entry.camera_id == "cam-1"

        await logger.stop()

    async def test_face_recognized_event_persisted(
        self, event_bus: EventBus, db_session: AsyncSession
    ) -> None:
        settings = ObservabilitySettings(audit_log_enabled=True)
        factory = _make_session_factory(db_session)
        logger = AuditLogger(event_bus, factory, settings)
        logger.start()

        event = FaceRecognizedEvent(
            camera_id="cam-2",
            identity="alice",
            confidence=0.85,
            correlation_id="corr-456",
        )
        await event_bus.publish(event)

        result = await db_session.execute(select(AuditLogEntry))
        entries = result.scalars().all()
        assert any(e.event_type == "face_recognized" for e in entries)

        await logger.stop()

    async def test_alert_fired_event_persisted(
        self, event_bus: EventBus, db_session: AsyncSession
    ) -> None:
        settings = ObservabilitySettings(audit_log_enabled=True)
        factory = _make_session_factory(db_session)
        logger = AuditLogger(event_bus, factory, settings)
        logger.start()

        event = AlertFiredEvent(
            channel="email",
            threat_label="gun",
            success=True,
            correlation_id="corr-789",
        )
        await event_bus.publish(event)

        result = await db_session.execute(select(AuditLogEntry))
        entries = result.scalars().all()
        assert any(e.event_type == "alert_fired" for e in entries)

        await logger.stop()


class TestAuditCorrelationId:
    """Correlation ID is included in audit entries."""

    async def test_correlation_id_persisted(
        self, event_bus: EventBus, db_session: AsyncSession
    ) -> None:
        settings = ObservabilitySettings(audit_log_enabled=True)
        factory = _make_session_factory(db_session)
        logger = AuditLogger(event_bus, factory, settings)
        logger.start()

        event = ThreatDetectedEvent(
            camera_id="cam-1",
            label="knife",
            confidence=0.9,
            correlation_id="my-correlation-id",
        )
        await event_bus.publish(event)

        result = await db_session.execute(select(AuditLogEntry))
        entries = result.scalars().all()
        assert len(entries) >= 1
        assert entries[0].correlation_id == "my-correlation-id"

        await logger.stop()

    async def test_disabled_audit_does_not_subscribe(self, event_bus: EventBus) -> None:
        settings = ObservabilitySettings(audit_log_enabled=False)
        factory = MagicMock()
        logger = AuditLogger(event_bus, factory, settings)
        logger.start()

        # Publish should not trigger any writes
        event = ThreatDetectedEvent(camera_id="cam-1", label="knife", confidence=0.9)
        await event_bus.publish(event)

        # Factory was never called because no subscriptions were made
        factory.assert_not_called()

        await logger.stop()
