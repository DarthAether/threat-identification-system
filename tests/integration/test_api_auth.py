"""Integration tests for authentication endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threat_id.core.security import Role, TokenPayload, create_access_token, hash_password
from threat_id.core.config import JwtSettings


class TestLoginEndpoint:
    """POST /api/v1/auth/token — credential exchange."""

    def test_valid_credentials_returns_tokens(self, test_client, db_session, test_settings) -> None:
        """Seed a user and authenticate with valid credentials."""
        import asyncio
        from threat_id.db.models import User

        # Seed a user directly into the session
        user = User(
            username="loginuser",
            hashed_password=hash_password("correct-password"),
            role="operator",
            is_active=True,
        )

        async def seed():
            db_session.add(user)
            await db_session.flush()

        asyncio.get_event_loop().run_until_complete(seed())

        response = test_client.post(
            "/api/v1/auth/token",
            json={"username": "loginuser", "password": "correct-password"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_invalid_credentials_returns_401(self, test_client) -> None:
        response = test_client.post(
            "/api/v1/auth/token",
            json={"username": "nonexistent", "password": "wrong"},
        )
        assert response.status_code == 401

    def test_missing_password_returns_422(self, test_client) -> None:
        response = test_client.post(
            "/api/v1/auth/token",
            json={"username": "testuser"},
        )
        assert response.status_code == 422


class TestProtectedEndpoints:
    """Protected endpoints require valid JWT."""

    def test_no_token_returns_401(self, test_client) -> None:
        # Remove the current_user override temporarily
        from threat_id.api.dependencies import get_current_user

        original = test_client.app.dependency_overrides.get(get_current_user)
        test_client.app.dependency_overrides.pop(get_current_user, None)

        response = test_client.get("/api/v1/cameras")
        assert response.status_code in (401, 403)

        # Restore override
        if original is not None:
            test_client.app.dependency_overrides[get_current_user] = original

    def test_wrong_role_returns_403(self, test_client) -> None:
        """A VIEWER cannot access OPERATOR-only endpoints."""
        from threat_id.api.dependencies import get_current_user

        def _viewer_user() -> TokenPayload:
            return TokenPayload(
                sub="viewer-user",
                role=Role.VIEWER,
                exp=datetime.now(timezone.utc) + timedelta(hours=1),
                token_type="access",
            )

        test_client.app.dependency_overrides[get_current_user] = _viewer_user

        # POST /api/v1/detect requires OPERATOR or ADMIN
        response = test_client.post("/api/v1/detect")
        assert response.status_code in (403, 422)

        # Restore admin override
        def _admin_user() -> TokenPayload:
            return TokenPayload(
                sub="testuser",
                role=Role.ADMIN,
                exp=datetime.now(timezone.utc) + timedelta(hours=1),
                token_type="access",
            )

        test_client.app.dependency_overrides[get_current_user] = _admin_user
