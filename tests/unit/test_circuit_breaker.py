"""Tests for threat_id.core.circuit_breaker — state transitions and behaviour."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from threat_id.core.circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState


class TestClosedState:
    """CLOSED state allows calls through."""

    async def test_closed_state_allows_successful_call(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=1.0)

        async def ok() -> str:
            return "success"

        result = await cb.call(ok)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    async def test_closed_state_propagates_exception(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=1.0)

        async def fail() -> None:
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            await cb.call(fail)

    async def test_single_failure_stays_closed(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=1.0)

        async def fail() -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await cb.call(fail)

        assert cb.state == CircuitState.CLOSED


class TestTransitionToOpen:
    """Circuit opens after failure_threshold consecutive failures."""

    async def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60.0)

        async def fail() -> None:
            raise RuntimeError("fail")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        assert cb.state == CircuitState.OPEN

    async def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60.0)

        async def fail() -> None:
            raise RuntimeError("fail")

        async def ok() -> str:
            return "ok"

        # 2 failures, then success, then 2 more failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        await cb.call(ok)

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        # Should still be closed (never reached 3 consecutive failures)
        assert cb.state == CircuitState.CLOSED


class TestOpenState:
    """OPEN state rejects calls immediately."""

    async def test_open_rejects_with_circuit_breaker_error(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=60.0)

        async def fail() -> None:
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        assert cb.state == CircuitState.OPEN

        async def ok() -> str:
            return "ok"

        with pytest.raises(CircuitBreakerError, match="open"):
            await cb.call(ok)


class TestTransitionToHalfOpen:
    """Circuit transitions OPEN -> HALF_OPEN after recovery_timeout."""

    async def test_becomes_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

        async def fail() -> None:
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        time.sleep(0.15)

        # Accessing state triggers the transition
        assert cb.state == CircuitState.HALF_OPEN


class TestHalfOpenSuccess:
    """Successful call in HALF_OPEN closes the circuit."""

    async def test_success_in_half_open_closes_circuit(self) -> None:
        cb = CircuitBreaker(
            "test", failure_threshold=2, recovery_timeout=0.1, half_open_max_calls=1
        )

        async def fail() -> None:
            raise RuntimeError("fail")

        async def ok() -> str:
            return "ok"

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        result = await cb.call(ok)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED


class TestHalfOpenFailure:
    """Failed call in HALF_OPEN reopens the circuit."""

    async def test_failure_in_half_open_reopens_circuit(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

        async def fail() -> None:
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        with pytest.raises(RuntimeError):
            await cb.call(fail)

        assert cb.state == CircuitState.OPEN


class TestReset:
    """reset() forces the circuit back to CLOSED."""

    async def test_reset_from_open(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=60.0)

        async def fail() -> None:
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        assert cb.state == CircuitState.OPEN

        cb.reset()

        assert cb.state == CircuitState.CLOSED

    async def test_reset_allows_calls_again(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=60.0)

        async def fail() -> None:
            raise RuntimeError("fail")

        async def ok() -> str:
            return "ok"

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(fail)

        cb.reset()

        result = await cb.call(ok)
        assert result == "ok"
