"""Tests for threat_id.alerting.channels.email — email alert channel."""

from __future__ import annotations

from email.message import EmailMessage
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.factories import make_alert_payload
from threat_id.alerting.channels.email import EmailAlertChannel
from threat_id.alerting.models import Severity
from threat_id.core.circuit_breaker import CircuitBreaker, CircuitBreakerError
from threat_id.core.config import EmailSettings
from threat_id.core.exceptions import EmailDeliveryError


def _configured_settings() -> EmailSettings:
    return EmailSettings(
        host="smtp.test.com",
        port=465,
        username="user@test.com",
        password="secret",
        sender="alerts@test.com",
        recipients=["admin@test.com", "ops@test.com"],
    )


def _unconfigured_settings() -> EmailSettings:
    return EmailSettings(
        host="localhost",
        port=1025,
        username="",
        password="",
        sender="",
        recipients=[],
    )


class TestEmailConstruction:
    """Email messages are built with correct subject and body."""

    def test_subject_contains_severity_and_label(self) -> None:
        channel = EmailAlertChannel(_configured_settings())
        payload = make_alert_payload(
            severity=Severity.CRITICAL,
            threat_label="gun",
            camera_id="cam-5",
        )
        msg = channel._build_message(payload)
        subject = msg["Subject"]
        assert "CRITICAL" in subject
        assert "gun" in subject
        assert "cam-5" in subject

    def test_body_contains_key_fields(self) -> None:
        channel = EmailAlertChannel(_configured_settings())
        payload = make_alert_payload(
            threat_label="knife",
            camera_id="cam-1",
            confidence=0.92,
        )
        msg = channel._build_message(payload)
        body = msg.get_content()
        assert "knife" in body
        assert "cam-1" in body
        assert "92" in body  # 92% confidence

    def test_recipients_set_correctly(self) -> None:
        settings = _configured_settings()
        channel = EmailAlertChannel(settings)
        payload = make_alert_payload()
        msg = channel._build_message(payload)
        assert "admin@test.com" in msg["To"]
        assert "ops@test.com" in msg["To"]

    def test_sender_set_correctly(self) -> None:
        settings = _configured_settings()
        channel = EmailAlertChannel(settings)
        payload = make_alert_payload()
        msg = channel._build_message(payload)
        assert msg["From"] == "alerts@test.com"


class TestEmailCircuitBreakerIntegration:
    """Circuit breaker guards the SMTP connection."""

    async def test_send_uses_circuit_breaker(self) -> None:
        cb = CircuitBreaker("test_email", failure_threshold=2, recovery_timeout=60.0)
        channel = EmailAlertChannel(_configured_settings(), circuit_breaker=cb)

        # Mock the executor-based send to fail
        with patch.object(channel, "_send_in_executor", side_effect=Exception("SMTP down")):
            with pytest.raises(EmailDeliveryError):
                await channel.send(make_alert_payload())

            with pytest.raises(EmailDeliveryError):
                await channel.send(make_alert_payload())

        # After 2 failures, circuit should be open
        assert cb.state.value == "open"

    async def test_circuit_breaker_rejects_when_open(self) -> None:
        cb = CircuitBreaker("test_email", failure_threshold=1, recovery_timeout=60.0)
        channel = EmailAlertChannel(_configured_settings(), circuit_breaker=cb)

        with patch.object(channel, "_send_in_executor", side_effect=Exception("fail")):
            with pytest.raises(EmailDeliveryError):
                await channel.send(make_alert_payload())

        # Now circuit is open; next call should also raise
        with pytest.raises(EmailDeliveryError):
            await channel.send(make_alert_payload())


class TestEmailIsAvailable:
    """is_available returns false when SMTP is not configured."""

    def test_unconfigured_returns_false(self) -> None:
        channel = EmailAlertChannel(_unconfigured_settings())
        assert channel.is_available is False

    def test_configured_returns_true(self) -> None:
        channel = EmailAlertChannel(_configured_settings())
        assert channel.is_available is True

    async def test_send_raises_when_not_configured(self) -> None:
        channel = EmailAlertChannel(_unconfigured_settings())
        with pytest.raises(EmailDeliveryError, match="not configured"):
            await channel.send(make_alert_payload())

    def test_name_is_email(self) -> None:
        channel = EmailAlertChannel(_configured_settings())
        assert channel.name == "email"
