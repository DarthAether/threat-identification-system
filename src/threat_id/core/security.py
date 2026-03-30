"""JWT authentication, password hashing, and RBAC.

Uses python-jose for JWT and passlib[bcrypt] for password hashing.
All free, no external auth provider needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from threat_id.core.config import JwtSettings
from threat_id.core.exceptions import (
    AuthError,
    InsufficientPermissionsError,
    InvalidCredentialsError,
    TokenExpiredError,
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Roles ────────────────────────────────────────────────────────────────────

class Role(StrEnum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"

# Permission matrix
ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {
        "camera:read", "camera:write", "camera:delete",
        "detection:read", "detection:write",
        "recognition:read", "recognition:write",
        "alerts:read", "alerts:write",
        "users:read", "users:write", "users:delete",
        "analytics:read", "admin:access",
    },
    Role.OPERATOR: {
        "camera:read", "camera:write",
        "detection:read", "detection:write",
        "recognition:read", "recognition:write",
        "alerts:read", "alerts:write",
        "analytics:read",
    },
    Role.VIEWER: {
        "camera:read",
        "detection:read",
        "recognition:read",
        "alerts:read",
        "analytics:read",
    },
}


# ── Token Models ─────────────────────────────────────────────────────────────

class TokenPayload(BaseModel):
    sub: str  # username
    role: Role
    exp: datetime
    token_type: str = "access"


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# ── Password Hashing ────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT ──────────────────────────────────────────────────────────────────────

def create_access_token(username: str, role: Role, settings: JwtSettings) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": username,
        "role": role.value,
        "exp": expire,
        "token_type": "access",
    }
    return jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=settings.algorithm)


def create_refresh_token(username: str, role: Role, settings: JwtSettings) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": username,
        "role": role.value,
        "exp": expire,
        "token_type": "refresh",
    }
    return jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=settings.algorithm)


def create_token_pair(username: str, role: Role, settings: JwtSettings) -> TokenPair:
    return TokenPair(
        access_token=create_access_token(username, role, settings),
        refresh_token=create_refresh_token(username, role, settings),
    )


def decode_token(token: str, settings: JwtSettings) -> TokenPayload:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.algorithm],
        )
        return TokenPayload(**payload)
    except JWTError as exc:
        if "expired" in str(exc).lower():
            raise TokenExpiredError("Token has expired") from exc
        raise AuthError(f"Invalid token: {exc}") from exc


def check_permission(role: Role, permission: str) -> None:
    """Raise InsufficientPermissionsError if the role lacks the permission."""
    allowed = ROLE_PERMISSIONS.get(role, set())
    if permission not in allowed:
        raise InsufficientPermissionsError(
            f"Role '{role}' does not have permission '{permission}'"
        )
