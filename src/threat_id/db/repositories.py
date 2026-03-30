"""CRUD repository layer.

Each repository receives an ``AsyncSession`` at construction time so the
caller (typically a FastAPI endpoint via ``Depends``) controls the
transaction boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from threat_id.core.security import Role
from threat_id.db.models import (
    AlertRecord,
    AuditLogEntry,
    DetectionRecord,
    User,
)


# ── User Repository ────────────────────────────────────────────────────────


class UserRepository:
    """CRUD operations for :class:`User`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        username: str,
        hashed_password: str,
        role: Role = Role.VIEWER,
    ) -> User:
        """Insert a new user and return the persisted object."""
        user = User(
            username=username,
            hashed_password=hashed_password,
            role=role.value,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def get_by_username(self, username: str) -> User | None:
        """Look up a single user by unique username."""
        stmt = select(User).where(User.username == username)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_users(
        self,
        *,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[User]:
        """Return a page of users ordered by creation date."""
        stmt = select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
        if active_only:
            stmt = stmt.where(User.is_active.is_(True))
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def update_role(self, username: str, role: Role) -> User | None:
        """Change a user's role. Returns the updated user or ``None``."""
        user = await self.get_by_username(username)
        if user is None:
            return None
        user.role = role.value
        await self._session.flush()
        return user

    async def deactivate(self, username: str) -> User | None:
        """Soft-delete a user by setting ``is_active = False``."""
        user = await self.get_by_username(username)
        if user is None:
            return None
        user.is_active = False
        await self._session.flush()
        return user


# ── Alert Repository ───────────────────────────────────────────────────────


class AlertRepository:
    """CRUD operations for :class:`AlertRecord`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        camera_id: str,
        severity: str,
        threat_label: str,
        confidence: float,
        bbox_json: dict[str, Any] | None = None,
    ) -> AlertRecord:
        """Persist a new alert and return it."""
        record = AlertRecord(
            camera_id=camera_id,
            severity=severity,
            threat_label=threat_label,
            confidence=confidence,
            bbox_json=bbox_json,
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def list_recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        acknowledged: bool | None = None,
    ) -> Sequence[AlertRecord]:
        """Return alerts ordered newest-first with optional ack filter."""
        stmt = (
            select(AlertRecord)
            .order_by(AlertRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if acknowledged is not None:
            stmt = stmt.where(AlertRecord.acknowledged.is_(acknowledged))
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def acknowledge(
        self,
        alert_id: int,
        *,
        username: str,
    ) -> AlertRecord | None:
        """Mark an alert as acknowledged. Returns the updated record."""
        stmt = select(AlertRecord).where(AlertRecord.id == alert_id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        record.acknowledged = True
        record.acknowledged_by = username
        record.acknowledged_at = datetime.now(timezone.utc)
        await self._session.flush()
        return record

    async def count_by_severity(self) -> dict[str, int]:
        """Return a mapping of severity -> count for unacknowledged alerts."""
        stmt = (
            select(AlertRecord.severity, func.count(AlertRecord.id))
            .where(AlertRecord.acknowledged.is_(False))
            .group_by(AlertRecord.severity)
        )
        result = await self._session.execute(stmt)
        return {row[0]: row[1] for row in result.all()}


# ── Audit Repository ───────────────────────────────────────────────────────


class AuditRepository:
    """CRUD and query operations for :class:`AuditLogEntry`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        event_type: str,
        camera_id: str | None = None,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> AuditLogEntry:
        """Write a new audit log entry."""
        entry = AuditLogEntry(
            event_type=event_type,
            camera_id=camera_id,
            payload=payload,
            correlation_id=correlation_id,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def query(
        self,
        *,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        event_type: str | None = None,
        camera_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[AuditLogEntry]:
        """Filtered query with optional time range, event type, and camera."""
        stmt = (
            select(AuditLogEntry)
            .order_by(AuditLogEntry.timestamp.desc())
            .limit(limit)
            .offset(offset)
        )
        if time_min is not None:
            stmt = stmt.where(AuditLogEntry.timestamp >= time_min)
        if time_max is not None:
            stmt = stmt.where(AuditLogEntry.timestamp <= time_max)
        if event_type is not None:
            stmt = stmt.where(AuditLogEntry.event_type == event_type)
        if camera_id is not None:
            stmt = stmt.where(AuditLogEntry.camera_id == camera_id)
        result = await self._session.execute(stmt)
        return result.scalars().all()


# ── Detection Repository ───────────────────────────────────────────────────


class DetectionRepository:
    """Batch insert and aggregate queries for :class:`DetectionRecord`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_batch(self, records: Sequence[DetectionRecord]) -> None:
        """Bulk-insert a sequence of detection records."""
        self._session.add_all(records)
        await self._session.flush()

    async def get_stats(
        self,
        *,
        time_min: datetime,
        time_max: datetime,
        camera_id: str | None = None,
    ) -> dict[str, int]:
        """Return detection counts grouped by label within a time window.

        Parameters
        ----------
        time_min / time_max:
            The inclusive time range.
        camera_id:
            Optional filter to a single camera.

        Returns
        -------
        dict[str, int]
            ``{label: count}``
        """
        stmt = (
            select(DetectionRecord.label, func.count(DetectionRecord.id))
            .where(DetectionRecord.timestamp >= time_min)
            .where(DetectionRecord.timestamp <= time_max)
            .group_by(DetectionRecord.label)
        )
        if camera_id is not None:
            stmt = stmt.where(DetectionRecord.camera_id == camera_id)
        result = await self._session.execute(stmt)
        return {row[0]: row[1] for row in result.all()}
