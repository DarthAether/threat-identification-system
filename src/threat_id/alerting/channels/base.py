"""Base protocol for alert delivery channels.

Every concrete channel implements this protocol so that the
AlertService can dispatch to heterogeneous backends uniformly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from threat_id.alerting.models import AlertPayload


@runtime_checkable
class AlertChannelProtocol(Protocol):
    """Structural sub-typing contract for alert channels."""

    async def send(self, payload: AlertPayload) -> None:
        """Deliver an alert through this channel.

        Implementations must raise the appropriate domain exception
        (EmailDeliveryError, WebhookDeliveryError, etc.) on failure.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable channel identifier (e.g. ``"email"``)."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether this channel is currently operational."""
        ...
