"""Integration tests for alert endpoints."""

from __future__ import annotations

import asyncio

import pytest

from threat_id.db.models import AlertRecord


class TestAlertsListEndpoint:
    """GET /api/v1/alerts — paginated alert listing."""

    def test_alerts_returns_paginated_list(self, test_client) -> None:
        response = test_client.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "total" in data
        assert isinstance(data["alerts"], list)

    def test_alerts_with_limit_param(self, test_client) -> None:
        response = test_client.get("/api/v1/alerts", params={"limit": 10})
        assert response.status_code == 200

    def test_alerts_with_offset_param(self, test_client) -> None:
        response = test_client.get("/api/v1/alerts", params={"offset": 5})
        assert response.status_code == 200


class TestAcknowledgeEndpoint:
    """PATCH /api/v1/alerts/{id}/acknowledge — mark alert as acknowledged."""

    def test_acknowledge_existing_alert(self, test_client, db_session) -> None:
        # Seed an alert
        async def seed():
            record = AlertRecord(
                camera_id="cam-1",
                severity="high",
                threat_label="knife",
                confidence=0.85,
                acknowledged=False,
            )
            db_session.add(record)
            await db_session.flush()
            return record.id

        alert_id = asyncio.get_event_loop().run_until_complete(seed())

        response = test_client.patch(f"/api/v1/alerts/{alert_id}/acknowledge")
        assert response.status_code == 200
        data = response.json()
        assert "acknowledged" in data.get("message", "").lower() or "acknowledged_by" in data

    def test_acknowledge_nonexistent_alert_returns_404(self, test_client) -> None:
        response = test_client.patch("/api/v1/alerts/99999/acknowledge")
        assert response.status_code == 404
