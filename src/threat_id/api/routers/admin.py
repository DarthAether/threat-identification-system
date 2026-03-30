"""Admin endpoints for user management.

- ``GET    /v1/admin/users``          — list all users.
- ``POST   /v1/admin/users``          — create a new user.
- ``PATCH  /v1/admin/users/{id}/role`` — change a user's role.
- ``DELETE /v1/admin/users/{id}``     — deactivate a user.

All endpoints require the ``ADMIN`` role.
"""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from threat_id.api.dependencies import (
    get_current_user,
    get_db_session,
    require_role,
)
from threat_id.core.exceptions import AuthError, ThreatIdError
from threat_id.core.security import Role, TokenPayload, hash_password
from threat_id.db.repositories import UserRepository

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_role(Role.ADMIN))],
)


# ── Request / Response schemas ──────────────────────────────────────────────


class UserInfo(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: str


class UserListResponse(BaseModel):
    users: list[UserInfo]
    total: int


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=8, max_length=128)
    role: Role = Field(default=Role.VIEWER)


class UserCreateResponse(BaseModel):
    id: int
    username: str
    role: str
    message: str


class RoleUpdateRequest(BaseModel):
    role: Role


class UserNotFoundError(ThreatIdError):
    code = "USER_NOT_FOUND"
    status_code = 404


class UserConflictError(ThreatIdError):
    code = "USER_CONFLICT"
    status_code = 409


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List all users",
)
async def list_users(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserListResponse:
    """Return all active users ordered by creation date."""
    repo = UserRepository(session)
    users = await repo.list_users(active_only=False)

    items = [
        UserInfo(
            id=u.id,
            username=u.username,
            role=u.role,
            is_active=u.is_active,
            created_at=u.created_at.isoformat(),
        )
        for u in users
    ]

    return UserListResponse(users=items, total=len(items))


@router.post(
    "/users",
    response_model=UserCreateResponse,
    status_code=201,
    summary="Create a new user",
)
async def create_user(
    body: UserCreateRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
) -> UserCreateResponse:
    """Create a new user account with hashed password."""
    repo = UserRepository(session)

    # Check for existing user
    existing = await repo.get_by_username(body.username)
    if existing is not None:
        raise UserConflictError(f"Username '{body.username}' already exists")

    hashed = hash_password(body.password)
    user = await repo.create(
        username=body.username,
        hashed_password=hashed,
        role=body.role,
    )

    logger.info(
        "admin.user_created",
        username=body.username,
        role=body.role.value,
        created_by=current_user.sub,
    )

    return UserCreateResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        message=f"User '{body.username}' created successfully",
    )


@router.patch(
    "/users/{user_id}/role",
    summary="Update a user's role",
)
async def update_role(
    user_id: int,
    body: RoleUpdateRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
) -> dict[str, Any]:
    """Change the role of an existing user by user ID.

    The user is looked up by ID. Because ``UserRepository.update_role``
    operates by username, we first resolve the username from the ID.
    """
    from sqlalchemy import select  # noqa: PLC0415
    from threat_id.db.models import User  # noqa: PLC0415

    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        raise UserNotFoundError(f"User {user_id} not found")

    repo = UserRepository(session)
    updated = await repo.update_role(user.username, body.role)

    logger.info(
        "admin.role_updated",
        user_id=user_id,
        new_role=body.role.value,
        updated_by=current_user.sub,
    )

    return {
        "message": f"Role updated to '{body.role.value}'",
        "user_id": user_id,
        "username": user.username,
        "role": body.role.value,
    }


@router.delete(
    "/users/{user_id}",
    summary="Deactivate a user",
)
async def deactivate_user(
    user_id: int,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
) -> dict[str, str]:
    """Soft-delete a user by setting their account to inactive."""
    from sqlalchemy import select  # noqa: PLC0415
    from threat_id.db.models import User  # noqa: PLC0415

    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        raise UserNotFoundError(f"User {user_id} not found")

    # Prevent self-deactivation
    if user.username == current_user.sub:
        raise AuthError("Cannot deactivate your own account")

    repo = UserRepository(session)
    await repo.deactivate(user.username)

    logger.info(
        "admin.user_deactivated",
        user_id=user_id,
        username=user.username,
        deactivated_by=current_user.sub,
    )

    return {"message": f"User '{user.username}' deactivated"}
