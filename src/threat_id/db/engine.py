"""Async SQLAlchemy engine, session factory, and FastAPI dependency.

All database access flows through the async session produced here.
Engine tuning knobs come from ``DatabaseSettings``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine as _sa_create_async_engine,
)

if TYPE_CHECKING:
    from threat_id.core.config import DatabaseSettings

# Module-level references populated at startup by ``init_db``.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


# ── Factory Functions ───────────────────────────────────────────────────────


def create_async_engine(settings: DatabaseSettings) -> AsyncEngine:
    """Build an :class:`AsyncEngine` from application settings.

    Parameters
    ----------
    settings:
        ``DatabaseSettings`` with connection URL, pool, and echo options.

    Returns
    -------
    AsyncEngine
        A configured async engine ready for session binding.
    """
    return _sa_create_async_engine(
        settings.async_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a reusable session factory bound to *engine*.

    Parameters
    ----------
    engine:
        The :class:`AsyncEngine` to bind sessions to.

    Returns
    -------
    async_sessionmaker[AsyncSession]
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


# ── Initialisation / Teardown ──────────────────────────────────────────────


def init_db(settings: DatabaseSettings) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """One-shot initialiser called during application startup.

    Sets the module-level engine and factory so ``get_session`` works as a
    FastAPI dependency.

    Returns
    -------
    tuple[AsyncEngine, async_sessionmaker[AsyncSession]]
    """
    global _engine, _session_factory  # noqa: PLW0603
    _engine = create_async_engine(settings)
    _session_factory = create_session_factory(_engine)
    return _engine, _session_factory


async def close_db() -> None:
    """Dispose of the engine connection pool — call on shutdown."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


# ── FastAPI Dependency ──────────────────────────────────────────────────────


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an ``AsyncSession`` for request-scoped use.

    Intended for ``fastapi.Depends``::

        @router.get("/items")
        async def list_items(session: AsyncSession = Depends(get_session)):
            ...

    The session is committed on success and rolled back on exception.
    """
    if _session_factory is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() during application startup."
        )

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
