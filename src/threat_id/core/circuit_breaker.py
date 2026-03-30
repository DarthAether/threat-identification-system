"""Circuit breaker pattern for external service calls.

Prevents cascade failures when SMTP, webhooks, or other external
services are down. Three states: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and calls are rejected."""


class CircuitBreaker:
    """Async circuit breaker wrapping external calls.

    Args:
        name: Identifier for logging.
        failure_threshold: Failures before opening the circuit.
        recovery_timeout: Seconds before probing with a half-open call.
        half_open_max_calls: Successes needed to close the circuit.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("circuit_breaker.half_open", name=self.name)
        return self._state

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        async with self._lock:
            current = self.state

            if current == CircuitState.OPEN:
                logger.warning("circuit_breaker.rejected", name=self.name)
                raise CircuitBreakerError(f"Circuit '{self.name}' is open")

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            await self._on_failure()
            raise exc  # noqa: TRY201
        else:
            await self._on_success()
            return result

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("circuit_breaker.reopened", name=self.name)
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit_breaker.opened",
                    name=self.name,
                    failures=self._failure_count,
                )

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("circuit_breaker.closed", name=self.name)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def reset(self) -> None:
        """Force-reset the circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
