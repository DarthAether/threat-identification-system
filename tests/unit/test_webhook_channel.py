"""Tests for threat_id.alerting.channels.webhook — webhook alert channel."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tests.factories import make_alert_payload
from threat_id.alerting.channels.webhook import WebhookAlertChannel
from threat_id.core.circuit_breaker import CircuitBreaker
from threat_id.core.config import WebhookSettings
from threat_id.core.exceptions import WebhookDeliveryError


def _configured_settings() -> WebhookSettings:
    return WebhookSettings(url="https://hooks.test.com/alert", timeout_seconds=5)


def _unconfigured_settings() -> WebhookSettings:
    return WebhookSettings(url="", timeout_seconds=5)


class TestWebhookPostPayload:
    """Webhook POSTs the correct JSON payload."""

    async def test_posts_json_payload(self) -> None:
        settings = _configured_settings()
        mock_response = httpx.Response(200, request=httpx.Request("POST", settings.url))

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        channel = WebhookAlertChannel(settings, http_client=mock_client)
        payload = make_alert_payload(threat_label="gun", camera_id="cam-2")
        await channel.send(payload)

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs.get("json") is not None or call_kwargs[1].get("json") is not None

    async def test_payload_contains_threat_label(self) -> None:
        settings = _configured_settings()
        mock_response = httpx.Response(200, request=httpx.Request("POST", settings.url))
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        channel = WebhookAlertChannel(settings, http_client=mock_client)
        payload = make_alert_payload(threat_label="rifle")
        await channel.send(payload)

        posted_json = mock_client.post.call_args[1]["json"]
        assert posted_json["threat_label"] == "rifle"

    async def test_not_configured_raises(self) -> None:
        channel = WebhookAlertChannel(_unconfigured_settings())
        with pytest.raises(WebhookDeliveryError, match="not configured"):
            await channel.send(make_alert_payload())

    def test_name_is_webhook(self) -> None:
        channel = WebhookAlertChannel(_configured_settings())
        assert channel.name == "webhook"

    def test_is_available_when_configured(self) -> None:
        assert WebhookAlertChannel(_configured_settings()).is_available is True

    def test_not_available_when_unconfigured(self) -> None:
        assert WebhookAlertChannel(_unconfigured_settings()).is_available is False


class TestWebhookCircuitBreaker:
    """Circuit breaker opens after repeated failures."""

    async def test_circuit_opens_on_failures(self) -> None:
        settings = _configured_settings()
        cb = CircuitBreaker("test_webhook", failure_threshold=2, recovery_timeout=60.0)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        channel = WebhookAlertChannel(settings, circuit_breaker=cb, http_client=mock_client)

        for _ in range(2):
            with pytest.raises(WebhookDeliveryError):
                await channel.send(make_alert_payload())

        assert cb.state.value == "open"

    async def test_non_2xx_raises_delivery_error(self) -> None:
        settings = _configured_settings()
        mock_response = httpx.Response(
            500,
            request=httpx.Request("POST", settings.url),
            text="Internal Server Error",
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        channel = WebhookAlertChannel(settings, http_client=mock_client)

        with pytest.raises(WebhookDeliveryError, match="500"):
            await channel.send(make_alert_payload())


class TestWebhookTimeout:
    """Timeout is respected during webhook delivery."""

    async def test_timeout_value_passed_to_client(self) -> None:
        settings = WebhookSettings(url="https://hooks.test.com/alert", timeout_seconds=3)
        mock_response = httpx.Response(200, request=httpx.Request("POST", settings.url))
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        channel = WebhookAlertChannel(settings, http_client=mock_client)
        await channel.send(make_alert_payload())

        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["timeout"] == 3

    async def test_timeout_error_raises_delivery_error(self) -> None:
        settings = _configured_settings()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.ReadTimeout("read timed out")
        )

        channel = WebhookAlertChannel(settings, http_client=mock_client)

        with pytest.raises(WebhookDeliveryError):
            await channel.send(make_alert_payload())
