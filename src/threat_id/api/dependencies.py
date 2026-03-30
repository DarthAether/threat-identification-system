"""FastAPI dependency injection functions.

All service instances live on ``app.state`` and are exposed here as
thin ``Depends()``-compatible callables. Authentication dependencies
validate JWTs and enforce role-based access control.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

import structlog
from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from threat_id.alerting.service import AlertService
from threat_id.api.websocket_manager import ConnectionManager
from threat_id.camera.manager import CameraManager
from threat_id.core.config import Settings
from threat_id.core.events import EventBus
from threat_id.core.exceptions import AuthError, InsufficientPermissionsError
from threat_id.core.security import Role, TokenPayload, decode_token
from threat_id.db.engine import get_session as _db_get_session
from threat_id.detection.service import DetectionService
from threat_id.recognition.service import RecognitionService

logger = structlog.get_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


# ── Settings ────────────────────────────────────────────────────────────────


def get_settings(request: Request) -> Settings:
    """Return the application-wide :class:`Settings` instance."""
    return request.app.state.settings  # type: ignore[no-any-return]


# ── Database Session ────────────────────────────────────────────────────────


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a request-scoped async database session.

    Delegates to :func:`threat_id.db.engine.get_session` which commits on
    success and rolls back on exception.
    """
    async for session in _db_get_session():
        yield session


# ── Authentication ──────────────────────────────────────────────────────────


def get_current_user(
    request: Request,
    token: Annotated[str, Depends(oauth2_scheme)],
) -> TokenPayload:
    """Validate the bearer JWT and return the decoded payload.

    Raises
    ------
    AuthError
        If the token is missing, malformed, or expired.
    """
    settings: Settings = request.app.state.settings
    payload = decode_token(token, settings.jwt)
    if payload.token_type != "access":
        raise AuthError("Expected an access token")
    return payload


# ── Role-Based Access Control ───────────────────────────────────────────────


def require_role(*roles: Role):
    """Dependency factory that enforces role membership.

    Usage::

        @router.get("/admin-only", dependencies=[Depends(require_role(Role.ADMIN))])
        async def admin_view(): ...

    Parameters
    ----------
    *roles:
        One or more :class:`Role` values. The current user must hold one
        of these roles to proceed.

    Returns
    -------
    Callable
        A FastAPI-compatible dependency.
    """

    def _checker(
        current_user: Annotated[TokenPayload, Depends(get_current_user)],
    ) -> TokenPayload:
        if current_user.role not in roles:
            raise InsufficientPermissionsError(
                f"Role '{current_user.role}' is not in {[r.value for r in roles]}"
            )
        return current_user

    return _checker


# ── Service Dependencies ───────────────────────────────────────────────────


def get_detection_service(request: Request) -> DetectionService:
    """Return the :class:`DetectionService` from application state."""
    return request.app.state.detection_service  # type: ignore[no-any-return]


def get_recognition_service(request: Request) -> RecognitionService:
    """Return the :class:`RecognitionService` from application state."""
    return request.app.state.recognition_service  # type: ignore[no-any-return]


def get_alert_service(request: Request) -> AlertService:
    """Return the :class:`AlertService` from application state."""
    return request.app.state.alert_service  # type: ignore[no-any-return]


def get_camera_manager(request: Request) -> CameraManager:
    """Return the :class:`CameraManager` from application state."""
    return request.app.state.camera_manager  # type: ignore[no-any-return]


def get_event_bus(request: Request) -> EventBus:
    """Return the :class:`EventBus` from application state."""
    return request.app.state.event_bus  # type: ignore[no-any-return]


def get_metrics(request: Request) -> object:
    """Return the :class:`MetricsCollector` from application state."""
    return request.app.state.metrics_collector  # type: ignore[no-any-return]


def get_ws_manager(request: Request) -> ConnectionManager:
    """Return the :class:`ConnectionManager` from application state."""
    return request.app.state.ws_manager  # type: ignore[no-any-return]
