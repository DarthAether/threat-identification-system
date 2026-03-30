"""Central alerting service.

Subscribes to ``ThreatDetectedEvent`` on the event bus, enforces
per-label cooldowns, fans out to all enabled channels concurrently,
and publishes ``AlertFiredEvent`` for downstream consumers.
"""

from __future__ import annotations

import asyncio
import time
from typing import Sequence

import structlog

from threat_id.alerting.channels.base import AlertChannelProtocol
from threat_id.alerting.models import AlertPayload, Severity
from threat_id.core.config import AlertSettings
from threat_id.core.events import (
    AlertFiredEvent,
    EventBus,
    ThreatDetectedEvent,
)

logger = structlog.get_logger(__name__)


def _map_confidence_to_severity(confidence: float) -> Severity:
    """Deterministic severity mapping based on model confidence."""
    if confidence < 0.6:
        return Severity.WARNING
    if confidence < 0.8:
        return Severity.HIGH
    return Severity.CRITICAL


class AlertService:
    """Orchestrates alert dispatch across all enabled channels.

    Parameters
    ----------
    event_bus:
        Shared event bus instance for pub/sub.
    settings:
        Alert-related configuration (cooldowns, enabled channels).
    channels:
        Ordered list of channel implementations to try.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: AlertSettings,
        channels: Sequence[AlertChannelProtocol],
    ) -> None:
        self._bus = event_bus
        self._settings = settings
        self._channels = {ch.name: ch for ch in channels}
        self._cooldowns: dict[str, float] = {}

        self._bus.subscribe(ThreatDetectedEvent, self._on_threat_detected)
        logger.info(
            "alert_service.init",
            enabled=self._settings.enabled_channels,
            registered=list(self._channels.keys()),
        )

    # ── Event handler ────────────────────────────────────────────────

    async def _on_threat_detected(self, event: ThreatDetectedEvent) -> None:
        cooldown_key = f"{event.camera_id}:{event.label}"

        if self._is_in_cooldown(cooldown_key):
            logger.debug(
                "alert_service.cooldown_active",
                camera_id=event.camera_id,
                label=event.label,
            )
            return

        severity = _map_confidence_to_severity(event.confidence)
        payload = AlertPayload(
            timestamp=event.timestamp,
            camera_id=event.camera_id,
            threat_label=event.label,
            confidence=event.confidence,
            severity=severity,
            bbox=event.bbox,
            correlation_id=event.correlation_id or "",
        )

        self._cooldowns[cooldown_key] = time.monotonic()
        await self._dispatch(payload)

    # ── Dispatch ─────────────────────────────────────────────────────

    async def _dispatch(self, payload: AlertPayload) -> None:
        """Fan-out to every enabled and available channel concurrently."""
        tasks: list[asyncio.Task[None]] = []

        for channel_name in self._settings.enabled_channels:
            channel = self._channels.get(channel_name)
            if channel is None:
                logger.warning(
                    "alert_service.channel_not_registered",
                    channel=channel_name,
                )
                continue
            if not channel.is_available:
                logger.warning(
                    "alert_service.channel_unavailable",
                    channel=channel_name,
                )
                continue

            tasks.append(
                asyncio.create_task(
                    self._send_safe(channel, payload),
                    name=f"alert-{channel_name}",
                )
            )

        if not tasks:
            logger.warning("alert_service.no_channels_available")
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)

        succeeded: list[str] = []
        for task, result in zip(tasks, results, strict=True):
            channel_name = task.get_name().removeprefix("alert-")
            if isinstance(result, Exception):
                logger.error(
                    "alert_service.channel_failed",
                    channel=channel_name,
                    error=str(result),
                )
            else:
                succeeded.append(channel_name)

        for channel_name in succeeded:
            await self._bus.publish(
                AlertFiredEvent(
                    channel=channel_name,
                    threat_label=payload.threat_label,
                    success=True,
                    correlation_id=payload.correlation_id,
                )
            )

        logger.info(
            "alert_service.dispatched",
            threat=payload.threat_label,
            severity=payload.severity,
            channels_ok=succeeded,
            total=len(tasks),
        )

    @staticmethod
    async def _send_safe(
        channel: AlertChannelProtocol,
        payload: AlertPayload,
    ) -> None:
        """Wrapper that lets asyncio.gather collect exceptions."""
        await channel.send(payload)

    # ── Cooldown ─────────────────────────────────────────────────────

    def _is_in_cooldown(self, key: str) -> bool:
        last_fired = self._cooldowns.get(key)
        if last_fired is None:
            return False
        elapsed = time.monotonic() - last_fired
        return elapsed < self._settings.cooldown_seconds
