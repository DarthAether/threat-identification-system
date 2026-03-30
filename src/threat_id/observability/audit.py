"""Audit logger — persists security-relevant events to the database.

Subscribes to the in-process :class:`EventBus` and writes every
:class:`ThreatDetectedEvent`, :class:`AlertFiredEvent`, and
:class:`FaceRecognizedEvent` as an :class:`AuditLogEntry` via the
:class:`AuditRepository`.

A background retention task periodically purges entries older than the
configured ``audit_retention_days``.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from threat_id.core.config import ObservabilitySettings
from threat_id.core.events import (
    AlertFiredEvent,
    BaseEvent,
    EventBus,
    FaceRecognizedEvent,
    ThreatDetectedEvent,
)
from threat_id.core.logging import correlation_id_var
from threat_id.db.models import AuditLogEntry
from threat_id.db.repositories import AuditRepository

logger = structlog.get_logger(__name__)

# Events that the audit logger cares about, mapped to a human-readable type
# string persisted in the ``event_type`` column.
_AUDITABLE_EVENTS: dict[type[BaseEvent], str] = {
    ThreatDetectedEvent: "threat_detected",
    AlertFiredEvent: "alert_fired",
    FaceRecognizedEvent: "face_recognized",
}


def _event_to_payload(event: BaseEvent) -> dict[str, Any]:
    """Serialise a dataclass event into a JSON-safe dict for the JSONB column."""
    raw = asdict(event)
    # Remove fields stored in dedicated columns to avoid duplication.
    raw.pop("correlation_id", None)
    raw.pop("timestamp", None)
    # Convert non-serialisable values.
    for key, value in raw.items():
        if isinstance(value, datetime):
            raw[key] = value.isoformat()
    return raw


def _extract_camera_id(event: BaseEvent) -> str | None:
    """Extract camera_id if the event carries one."""
    return getattr(event, "camera_id", None) or None


class AuditLogger:
    """Subscribes to the event bus and persists audit entries.

    Parameters
    ----------
    event_bus:
        The application-wide :class:`EventBus`.
    session_factory:
        An ``async_sessionmaker`` used to create short-lived sessions for
        each write.  The audit logger manages its own transaction scope so
        it never interferes with the caller's session.
    settings:
        :class:`ObservabilitySettings` controlling enablement and retention.
    """

    def __init__(
        self,
        event_bus: EventBus,
        session_factory: async_sessionmaker[AsyncSession],
        settings: ObservabilitySettings,
    ) -> None:
        self._event_bus = event_bus
        self._session_factory = session_factory
        self._settings = settings
        self._retention_task: asyncio.Task[None] | None = None

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Subscribe to auditable events and start the retention worker."""
        if not self._settings.audit_log_enabled:
            logger.info("audit_logger.disabled")
            return

        for event_type in _AUDITABLE_EVENTS:
            self._event_bus.subscribe(event_type, self._handle_event)

        self._retention_task = asyncio.create_task(
            self._retention_loop(),
            name="audit-retention",
        )
        logger.info(
            "audit_logger.started",
            retention_days=self._settings.audit_retention_days,
        )

    async def stop(self) -> None:
        """Unsubscribe from events and cancel the retention worker."""
        for event_type in _AUDITABLE_EVENTS:
            self._event_bus.unsubscribe(event_type, self._handle_event)

        if self._retention_task is not None:
            self._retention_task.cancel()
            try:
                await self._retention_task
            except asyncio.CancelledError:
                pass
            self._retention_task = None

        logger.info("audit_logger.stopped")

    # -- Event Handler -------------------------------------------------------

    async def _handle_event(self, event: BaseEvent) -> None:
        """Persist a single event to the audit log."""
        event_type_str = _AUDITABLE_EVENTS.get(type(event))
        if event_type_str is None:
            return

        cid = event.correlation_id or correlation_id_var.get()
        camera_id = _extract_camera_id(event)
        payload = _event_to_payload(event)

        try:
            async with self._session_factory() as session:
                repo = AuditRepository(session)
                await repo.create(
                    event_type=event_type_str,
                    camera_id=camera_id,
                    payload=payload,
                    correlation_id=cid,
                )
                await session.commit()

            logger.debug(
                "audit_logger.persisted",
                event_type=event_type_str,
                camera_id=camera_id,
                correlation_id=cid,
            )
        except Exception:
            logger.exception(
                "audit_logger.persist_failed",
                event_type=event_type_str,
                camera_id=camera_id,
            )

    # -- Retention Cleanup ---------------------------------------------------

    async def _retention_loop(self) -> None:
        """Periodically delete audit entries older than the retention window.

        Runs once every 24 hours.  The first run is delayed by 60 seconds to
        avoid competing with application startup.
        """
        await asyncio.sleep(60)
        while True:
            try:
                deleted = await self._purge_old_entries()
                if deleted > 0:
                    logger.info("audit_logger.retention_purge", deleted=deleted)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("audit_logger.retention_error")

            # Run daily.
            await asyncio.sleep(86_400)

    async def _purge_old_entries(self) -> int:
        """Delete audit log entries older than ``audit_retention_days``.

        Returns
        -------
        int
            Number of rows deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self._settings.audit_retention_days,
        )
        async with self._session_factory() as session:
            stmt = delete(AuditLogEntry).where(AuditLogEntry.timestamp < cutoff)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount  # type: ignore[return-value]
