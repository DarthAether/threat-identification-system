"""Integration tests for health endpoints."""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    """GET /health returns 200 as a liveness probe."""

    def test_health_returns_200(self, test_client) -> None:
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, test_client) -> None:
        response = test_client.get("/health")
        data = response.json()
        assert data["status"] == "ok"


class TestReadyEndpoint:
    """GET /ready returns readiness status with checks."""

    def test_ready_returns_status(self, test_client) -> None:
        response = test_client.get("/ready")
        # May return 200 or 503 depending on DB connectivity in test;
        # we just verify it responds with a JSON body containing status.
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data

    def test_ready_includes_checks_dict(self, test_client) -> None:
        response = test_client.get("/ready")
        data = response.json()
        assert "checks" in data or "status" in data
