"""Authentication endpoints — login and token refresh.

- ``POST /auth/token``   — exchange credentials for a token pair.
- ``POST /auth/refresh`` — exchange a valid refresh token for a new
  access token.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from threat_id.api.dependencies import get_db_session, get_settings
from threat_id.core.config import Settings
from threat_id.core.exceptions import AuthError, InvalidCredentialsError
from threat_id.core.security import (
    Role,
    TokenPair,
    TokenPayload,
    create_token_pair,
    decode_token,
    verify_password,
)
from threat_id.db.repositories import UserRepository

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Request / Response schemas ──────────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post(
    "/token",
    response_model=TokenPair,
    summary="Authenticate and obtain tokens",
)
async def login(
    body: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenPair:
    """Validate username/password and return an access + refresh token pair."""
    repo = UserRepository(session)
    user = await repo.get_by_username(body.username)

    if user is None or not user.is_active:
        raise InvalidCredentialsError("Invalid username or password")

    if not verify_password(body.password, user.hashed_password):
        raise InvalidCredentialsError("Invalid username or password")

    role = Role(user.role)
    pair = create_token_pair(user.username, role, settings.jwt)

    logger.info("auth.login_success", username=user.username, role=role.value)
    return pair


@router.post(
    "/refresh",
    response_model=TokenPair,
    summary="Refresh access token",
)
async def refresh(
    body: RefreshRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TokenPair:
    """Exchange a valid refresh token for a fresh token pair.

    The refresh token's ``token_type`` claim must be ``"refresh"``.
    The user is re-validated against the database to ensure the account
    is still active.
    """
    payload: TokenPayload = decode_token(body.refresh_token, settings.jwt)

    if payload.token_type != "refresh":
        raise AuthError("Expected a refresh token")

    # Re-validate user is still active
    repo = UserRepository(session)
    user = await repo.get_by_username(payload.sub)
    if user is None or not user.is_active:
        raise AuthError("User account is inactive or deleted")

    role = Role(user.role)
    pair = create_token_pair(user.username, role, settings.jwt)

    logger.info("auth.token_refreshed", username=user.username)
    return pair
