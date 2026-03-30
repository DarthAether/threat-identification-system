"""Webhook alert channel — POSTs alert payloads to an external URL.

Uses ``httpx.AsyncClient`` for non-blocking HTTP and a circuit breaker
to back off when the remote endpoint is unreachable.
"""

from __future__ import annotations

import httpx
import structlog

from threat_id.alerting.models import AlertPayload
from threat_id.core.circuit_breaker import CircuitBreaker
from threat_id.core.config import WebhookSettings
from threat_id.core.exceptions import WebhookDeliveryError

logger = structlog.get_logger(__name__)


class WebhookAlertChannel:
    """Delivers alerts as JSON POST requests to a configured webhook URL."""

    def __init__(
        self,
        settings: WebhookSettings,
        circuit_breaker: CircuitBreaker | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._cb = circuit_breaker or CircuitBreaker(
            name="webhook_alert",
            failure_threshold=5,
            recovery_timeout=60.0,
        )
        self._client = http_client

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "webhook"

    @property
    def is_available(self) -> bool:
        return self._settings.is_configured

    # ── Public API ───────────────────────────────────────────────────

    async def send(self, payload: AlertPayload) -> None:
        """POST the alert payload as JSON to the webhook URL.

        Raises:
            WebhookDeliveryError: If the request fails or returns a
                non-2xx status code.
        """
        if not self.is_available:
            raise WebhookDeliveryError(
                "Webhook channel is not configured",
                detail="WEBHOOK_URL environment variable is empty",
            )

        try:
            await self._cb.call(self._post, payload)
        except WebhookDeliveryError:
            raise
        except Exception as exc:
            raise WebhookDeliveryError(
                f"Webhook delivery failed: {exc}",
                detail=str(exc),
            ) from exc

        logger.info(
            "webhook_alert.sent",
            camera_id=payload.camera_id,
            threat=payload.threat_label,
            url=self._settings.url,
        )

    # ── Internals ────────────────────────────────────────────────────

    async def _post(self, payload: AlertPayload) -> None:
        client = self._client or httpx.AsyncClient()
        own_client = self._client is None

        try:
            response = await client.post(
                self._settings.url,
                json=payload.model_dump(mode="json"),
                timeout=self._settings.timeout_seconds,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code >= 400:
                raise WebhookDeliveryError(
                    f"Webhook returned HTTP {response.status_code}",
                    detail=response.text[:500],
                )
        except httpx.HTTPError as exc:
            raise WebhookDeliveryError(
                f"HTTP error during webhook delivery: {exc}",
                detail=str(exc),
            ) from exc
        finally:
            if own_client:
                await client.aclose()
