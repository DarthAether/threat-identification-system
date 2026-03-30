"""Tests for threat_id.core.security — passwords, JWT, and RBAC."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import SecretStr

from threat_id.core.config import JwtSettings
from threat_id.core.exceptions import (
    AuthError,
    InsufficientPermissionsError,
    TokenExpiredError,
)
from threat_id.core.security import (
    ROLE_PERMISSIONS,
    Role,
    TokenPayload,
    check_permission,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


@pytest.fixture()
def jwt_settings() -> JwtSettings:
    return JwtSettings(
        secret_key="test-jwt-secret-key",
        algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7,
    )


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    """hash_password and verify_password must round-trip correctly."""

    def test_hash_and_verify_correct_password(self) -> None:
        hashed = hash_password("my-secure-password")
        assert verify_password("my-secure-password", hashed) is True

    def test_verify_wrong_password_returns_false(self) -> None:
        hashed = hash_password("correct-password")
        assert verify_password("wrong-password", hashed) is False

    def test_hash_is_not_plaintext(self) -> None:
        hashed = hash_password("plaintext")
        assert hashed != "plaintext"
        assert len(hashed) > 20

    def test_two_hashes_differ(self) -> None:
        hash1 = hash_password("same-password")
        hash2 = hash_password("same-password")
        # bcrypt uses different salts
        assert hash1 != hash2

    def test_both_hashes_verify_correctly(self) -> None:
        hash1 = hash_password("pwd")
        hash2 = hash_password("pwd")
        assert verify_password("pwd", hash1) is True
        assert verify_password("pwd", hash2) is True


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------


class TestJwtTokens:
    """create_access_token / decode_token round-trip and validation."""

    def test_create_and_decode_access_token(self, jwt_settings: JwtSettings) -> None:
        token = create_access_token("alice", Role.ADMIN, jwt_settings)
        payload = decode_token(token, jwt_settings)

        assert payload.sub == "alice"
        assert payload.role == Role.ADMIN
        assert payload.token_type == "access"
        assert payload.exp > datetime.now(timezone.utc)

    def test_create_and_decode_refresh_token(self, jwt_settings: JwtSettings) -> None:
        token = create_refresh_token("bob", Role.OPERATOR, jwt_settings)
        payload = decode_token(token, jwt_settings)

        assert payload.sub == "bob"
        assert payload.role == Role.OPERATOR
        assert payload.token_type == "refresh"

    def test_expired_token_raises_token_expired_error(self, jwt_settings: JwtSettings) -> None:
        expired_settings = JwtSettings(
            secret_key=jwt_settings.secret_key.get_secret_value(),
            algorithm="HS256",
            access_token_expire_minutes=-1,
        )
        token = create_access_token("alice", Role.ADMIN, expired_settings)

        with pytest.raises(TokenExpiredError, match="expired"):
            decode_token(token, jwt_settings)

    def test_tampered_token_raises_auth_error(self, jwt_settings: JwtSettings) -> None:
        token = create_access_token("alice", Role.ADMIN, jwt_settings)
        tampered = token + "tampered"

        with pytest.raises(AuthError):
            decode_token(tampered, jwt_settings)

    def test_wrong_secret_raises_auth_error(self, jwt_settings: JwtSettings) -> None:
        token = create_access_token("alice", Role.ADMIN, jwt_settings)
        other_settings = JwtSettings(secret_key="wrong-secret")

        with pytest.raises(AuthError):
            decode_token(token, other_settings)


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------


class TestCheckPermission:
    """check_permission enforces the role-permission matrix."""

    def test_admin_has_all_permissions(self) -> None:
        for perm in ROLE_PERMISSIONS[Role.ADMIN]:
            check_permission(Role.ADMIN, perm)  # Should not raise

    def test_viewer_has_read_permissions(self) -> None:
        check_permission(Role.VIEWER, "camera:read")
        check_permission(Role.VIEWER, "detection:read")
        check_permission(Role.VIEWER, "alerts:read")

    def test_viewer_cannot_write_camera(self) -> None:
        with pytest.raises(InsufficientPermissionsError):
            check_permission(Role.VIEWER, "camera:write")

    def test_viewer_cannot_access_admin(self) -> None:
        with pytest.raises(InsufficientPermissionsError):
            check_permission(Role.VIEWER, "admin:access")

    def test_operator_can_write_detection(self) -> None:
        check_permission(Role.OPERATOR, "detection:write")

    def test_operator_cannot_access_admin(self) -> None:
        with pytest.raises(InsufficientPermissionsError):
            check_permission(Role.OPERATOR, "admin:access")

    def test_operator_cannot_manage_users(self) -> None:
        with pytest.raises(InsufficientPermissionsError):
            check_permission(Role.OPERATOR, "users:write")

    def test_admin_can_manage_users(self) -> None:
        check_permission(Role.ADMIN, "users:write")
        check_permission(Role.ADMIN, "users:delete")

    def test_nonexistent_permission_raises(self) -> None:
        with pytest.raises(InsufficientPermissionsError):
            check_permission(Role.ADMIN, "nonexistent:permission")


class TestRbacPermissionMatrix:
    """Comprehensive checks on the permission matrix."""

    def test_admin_is_superset_of_operator(self) -> None:
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        operator_perms = ROLE_PERMISSIONS[Role.OPERATOR]
        assert operator_perms.issubset(admin_perms)

    def test_operator_is_superset_of_viewer(self) -> None:
        operator_perms = ROLE_PERMISSIONS[Role.OPERATOR]
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert viewer_perms.issubset(operator_perms)

    def test_viewer_has_no_write_permissions(self) -> None:
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        write_perms = {p for p in viewer_perms if ":write" in p or ":delete" in p}
        assert len(write_perms) == 0

    def test_all_roles_defined(self) -> None:
        for role in Role:
            assert role in ROLE_PERMISSIONS
