"""Integration tests for database repositories with async SQLite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from threat_id.core.security import Role, hash_password
from threat_id.db.models import AlertRecord, AuditLogEntry
from threat_id.db.repositories import AlertRepository, AuditRepository, UserRepository


# ---------------------------------------------------------------------------
# UserRepository
# ---------------------------------------------------------------------------


class TestUserRepositoryCrud:
    """UserRepository CRUD operations."""

    async def test_create_user(self, db_session: AsyncSession) -> None:
        repo = UserRepository(db_session)
        user = await repo.create("alice", hash_password("pass123"), Role.OPERATOR)

        assert user.id is not None
        assert user.username == "alice"
        assert user.role == "operator"
        assert user.is_active is True

    async def test_get_by_username(self, db_session: AsyncSession) -> None:
        repo = UserRepository(db_session)
        await repo.create("bob", hash_password("pass"), Role.VIEWER)

        found = await repo.get_by_username("bob")
        assert found is not None
        assert found.username == "bob"

    async def test_get_by_username_not_found(self, db_session: AsyncSession) -> None:
        repo = UserRepository(db_session)
        found = await repo.get_by_username("nonexistent")
        assert found is None

    async def test_list_users(self, db_session: AsyncSession) -> None:
        repo = UserRepository(db_session)
        await repo.create("user1", hash_password("p"), Role.VIEWER)
        await repo.create("user2", hash_password("p"), Role.OPERATOR)

        users = await repo.list_users()
        assert len(users) >= 2

    async def test_list_users_active_only(self, db_session: AsyncSession) -> None:
        repo = UserRepository(db_session)
        await repo.create("active_user", hash_password("p"), Role.VIEWER)
        inactive = await repo.create("inactive_user", hash_password("p"), Role.VIEWER)
        await repo.deactivate("inactive_user")

        users = await repo.list_users(active_only=True)
        usernames = [u.username for u in users]
        assert "active_user" in usernames
        assert "inactive_user" not in usernames

    async def test_update_role(self, db_session: AsyncSession) -> None:
        repo = UserRepository(db_session)
        await repo.create("charlie", hash_password("p"), Role.VIEWER)

        updated = await repo.update_role("charlie", Role.ADMIN)
        assert updated is not None
        assert updated.role == "admin"

    async def test_deactivate_user(self, db_session: AsyncSession) -> None:
        repo = UserRepository(db_session)
        await repo.create("dave", hash_password("p"), Role.VIEWER)

        deactivated = await repo.deactivate("dave")
        assert deactivated is not None
        assert deactivated.is_active is False


# ---------------------------------------------------------------------------
# AlertRepository
# ---------------------------------------------------------------------------


class TestAlertRepositoryCrud:
    """AlertRepository create/list/acknowledge operations."""

    async def test_create_alert(self, db_session: AsyncSession) -> None:
        repo = AlertRepository(db_session)
        record = await repo.create(
            camera_id="cam-1",
            severity="critical",
            threat_label="gun",
            confidence=0.95,
        )

        assert record.id is not None
        assert record.camera_id == "cam-1"
        assert record.severity == "critical"
        assert record.acknowledged is False

    async def test_list_recent(self, db_session: AsyncSession) -> None:
        repo = AlertRepository(db_session)
        await repo.create(camera_id="cam-1", severity="high", threat_label="knife", confidence=0.8)
        await repo.create(camera_id="cam-2", severity="critical", threat_label="gun", confidence=0.9)

        records = await repo.list_recent(limit=10)
        assert len(records) >= 2

    async def test_list_recent_with_acknowledged_filter(self, db_session: AsyncSession) -> None:
        repo = AlertRepository(db_session)
        alert = await repo.create(
            camera_id="cam-1", severity="high", threat_label="knife", confidence=0.8
        )
        await repo.acknowledge(alert.id, username="operator1")

        unacked = await repo.list_recent(acknowledged=False)
        acked = await repo.list_recent(acknowledged=True)

        unacked_ids = {r.id for r in unacked}
        acked_ids = {r.id for r in acked}
        assert alert.id in acked_ids
        assert alert.id not in unacked_ids

    async def test_acknowledge_alert(self, db_session: AsyncSession) -> None:
        repo = AlertRepository(db_session)
        alert = await repo.create(
            camera_id="cam-1", severity="warning", threat_label="knife", confidence=0.5
        )

        result = await repo.acknowledge(alert.id, username="admin")
        assert result is not None
        assert result.acknowledged is True
        assert result.acknowledged_by == "admin"
        assert result.acknowledged_at is not None

    async def test_acknowledge_nonexistent_returns_none(self, db_session: AsyncSession) -> None:
        repo = AlertRepository(db_session)
        result = await repo.acknowledge(99999, username="admin")
        assert result is None

    async def test_count_by_severity(self, db_session: AsyncSession) -> None:
        repo = AlertRepository(db_session)
        await repo.create(camera_id="c1", severity="high", threat_label="k", confidence=0.8)
        await repo.create(camera_id="c2", severity="high", threat_label="g", confidence=0.9)
        await repo.create(camera_id="c3", severity="critical", threat_label="r", confidence=0.95)

        counts = await repo.count_by_severity()
        assert counts.get("high", 0) >= 2
        assert counts.get("critical", 0) >= 1


# ---------------------------------------------------------------------------
# AuditRepository
# ---------------------------------------------------------------------------


class TestAuditRepositoryCrud:
    """AuditRepository create and query with filters."""

    async def test_create_audit_entry(self, db_session: AsyncSession) -> None:
        repo = AuditRepository(db_session)
        entry = await repo.create(
            event_type="threat_detected",
            camera_id="cam-1",
            payload={"label": "knife", "confidence": 0.9},
            correlation_id="corr-test-123",
        )

        assert entry.id is not None
        assert entry.event_type == "threat_detected"
        assert entry.camera_id == "cam-1"
        assert entry.correlation_id == "corr-test-123"

    async def test_query_all(self, db_session: AsyncSession) -> None:
        repo = AuditRepository(db_session)
        await repo.create(event_type="threat_detected", camera_id="cam-1")
        await repo.create(event_type="alert_fired", camera_id="cam-2")

        entries = await repo.query()
        assert len(entries) >= 2

    async def test_query_by_event_type(self, db_session: AsyncSession) -> None:
        repo = AuditRepository(db_session)
        await repo.create(event_type="threat_detected", camera_id="cam-1")
        await repo.create(event_type="alert_fired", camera_id="cam-2")

        entries = await repo.query(event_type="threat_detected")
        assert all(e.event_type == "threat_detected" for e in entries)

    async def test_query_by_camera_id(self, db_session: AsyncSession) -> None:
        repo = AuditRepository(db_session)
        await repo.create(event_type="threat_detected", camera_id="cam-filter")
        await repo.create(event_type="threat_detected", camera_id="cam-other")

        entries = await repo.query(camera_id="cam-filter")
        assert all(e.camera_id == "cam-filter" for e in entries)

    async def test_query_with_limit_offset(self, db_session: AsyncSession) -> None:
        repo = AuditRepository(db_session)
        for i in range(5):
            await repo.create(event_type="test_event", camera_id=f"cam-{i}")

        page1 = await repo.query(limit=2, offset=0)
        page2 = await repo.query(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        # Pages should not overlap
        ids1 = {e.id for e in page1}
        ids2 = {e.id for e in page2}
        assert ids1.isdisjoint(ids2)
