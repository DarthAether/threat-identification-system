"""Email alert channel — delivers alerts via SMTP over SSL.

SMTP operations are blocking, so they run inside a thread-pool
executor to avoid stalling the async event loop.  A circuit breaker
guards the SMTP connection to prevent cascading failures when the
mail server is unreachable.
"""

from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage
from functools import partial

import structlog

from threat_id.alerting.models import AlertPayload
from threat_id.core.circuit_breaker import CircuitBreaker
from threat_id.core.config import EmailSettings
from threat_id.core.exceptions import EmailDeliveryError

logger = structlog.get_logger(__name__)


class EmailAlertChannel:
    """Sends alert emails via SMTP_SSL using the configured mail server."""

    def __init__(
        self,
        settings: EmailSettings,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._settings = settings
        self._cb = circuit_breaker or CircuitBreaker(
            name="email_alert",
            failure_threshold=3,
            recovery_timeout=120.0,
        )

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "email"

    @property
    def is_available(self) -> bool:
        return self._settings.is_configured

    # ── Public API ───────────────────────────────────────────────────

    async def send(self, payload: AlertPayload) -> None:
        """Build and send an alert email.

        Raises:
            EmailDeliveryError: If the email could not be delivered.
        """
        if not self.is_available:
            raise EmailDeliveryError(
                "Email channel is not configured",
                detail="SMTP credentials or recipients missing",
            )

        msg = self._build_message(payload)

        try:
            await self._cb.call(self._send_in_executor, msg)
        except EmailDeliveryError:
            raise
        except Exception as exc:
            raise EmailDeliveryError(
                f"Failed to send alert email: {exc}",
                detail=str(exc),
            ) from exc

        logger.info(
            "email_alert.sent",
            camera_id=payload.camera_id,
            threat=payload.threat_label,
            recipients=self._settings.recipients,
        )

    # ── Internals ────────────────────────────────────────────────────

    def _build_message(self, payload: AlertPayload) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = (
            f"[{payload.severity.value.upper()}] Threat detected: "
            f"{payload.threat_label} on camera {payload.camera_id}"
        )
        msg["From"] = self._settings.sender
        msg["To"] = ", ".join(self._settings.recipients)

        body = (
            f"Threat Identification System Alert\n"
            f"{'=' * 40}\n\n"
            f"Timestamp : {payload.timestamp.isoformat()}\n"
            f"Camera    : {payload.camera_id}\n"
            f"Threat    : {payload.threat_label}\n"
            f"Confidence: {payload.confidence:.1%}\n"
            f"Severity  : {payload.severity.value.upper()}\n"
            f"Bounding  : {payload.bbox}\n"
            f"Corr. ID  : {payload.correlation_id}\n"
        )
        msg.set_content(body)
        return msg

    async def _send_in_executor(self, msg: EmailMessage) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self._smtp_send, msg))

    def _smtp_send(self, msg: EmailMessage) -> None:
        """Blocking SMTP send — runs inside the thread pool."""
        try:
            with smtplib.SMTP_SSL(
                self._settings.host,
                self._settings.port,
            ) as server:
                server.login(
                    self._settings.username,
                    self._settings.password.get_secret_value(),
                )
                server.send_message(msg)
        except smtplib.SMTPException as exc:
            raise EmailDeliveryError(
                f"SMTP error: {exc}",
                detail=str(exc),
            ) from exc
