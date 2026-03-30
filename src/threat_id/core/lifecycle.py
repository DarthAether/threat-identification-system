"""Application lifecycle management — startup/shutdown orchestration."""

from __future__ import annotations

import signal
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_shutdown_handlers: list = []


def register_shutdown(handler: Any) -> None:
    """Register a coroutine or callable to run on shutdown."""
    _shutdown_handlers.append(handler)


async def graceful_shutdown() -> None:
    """Execute all registered shutdown handlers in reverse order."""
    logger.info("lifecycle.shutdown_start", handlers=len(_shutdown_handlers))
    for handler in reversed(_shutdown_handlers):
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()
        except Exception as exc:
            logger.error("lifecycle.shutdown_error", handler=str(handler), error=str(exc))
    _shutdown_handlers.clear()
    logger.info("lifecycle.shutdown_complete")


def install_signal_handlers(loop: Any) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""
    import asyncio

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(graceful_shutdown()))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: asyncio.ensure_future(graceful_shutdown()))


import asyncio  # noqa: E402 — needed for the function above
