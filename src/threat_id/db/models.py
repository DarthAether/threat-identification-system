"""SQLAlchemy ORM models.

All tables live under a shared ``DeclarativeBase`` so Alembic can
auto-generate migrations from a single metadata object.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)

from threat_id.core.security import Role


# ── Base ────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
    }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── User ────────────────────────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default=Role.VIEWER.value)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=_utcnow,
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} role={self.role}>"


# ── Alert Record ────────────────────────────────────────────────────────────


class AlertRecord(Base):
    __tablename__ = "alert_records"
    __table_args__ = (
        Index("ix_alert_records_created_at", "created_at"),
        Index("ix_alert_records_severity", "severity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    camera_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    threat_label: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<AlertRecord id={self.id} threat={self.threat_label!r} "
            f"severity={self.severity} ack={self.acknowledged}>"
        )


# ── Audit Log Entry ────────────────────────────────────────────────────────


class AuditLogEntry(Base):
    __tablename__ = "audit_log_entries"
    __table_args__ = (
        Index("ix_audit_log_entries_timestamp", "timestamp"),
        Index("ix_audit_log_entries_event_type", "event_type"),
        Index("ix_audit_log_entries_camera_id", "camera_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    camera_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<AuditLogEntry id={self.id} type={self.event_type!r}>"


# ── Detection Record ───────────────────────────────────────────────────────


class DetectionRecord(Base):
    __tablename__ = "detection_records"
    __table_args__ = (
        Index("ix_detection_records_timestamp", "timestamp"),
        Index("ix_detection_records_label", "label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    camera_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_threat: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    bbox: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<DetectionRecord id={self.id} label={self.label!r} threat={self.is_threat}>"


# ── Face Record ─────────────────────────────────────────────────────────────


class FaceRecord(Base):
    __tablename__ = "face_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    embedding_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=_utcnow,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<FaceRecord id={self.id} name={self.name!r} active={self.is_active}>"
