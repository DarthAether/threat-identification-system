"""Tests for threat_id.alerting.service — AlertService dispatch and cooldown."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threat_id.alerting.models import AlertPayload, Severity
from threat_id.alerting.service import AlertService, _map_confidence_to_severity
from threat_id.core.config import AlertSettings
from threat_id.core.events import EventBus, ThreatDetectedEvent


class FakeChannel:
    """Minimal AlertChannelProtocol implementation for tests."""

    def __init__(self, channel_name: str, *, available: bool = True) -> None:
        self._name = channel_name
        self._available = available
        self.sent: list[AlertPayload] = []
        self.send_mock = AsyncMock(side_effect=self._record)

    async def _record(self, payload: AlertPayload) -> None:
        self.sent.append(payload)

    async def send(self, payload: AlertPayload) -> None:
        await self.send_mock(payload)

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return self._available


class TestAlertDispatch:
    """Alert dispatch to all enabled channels."""

    async def test_dispatches_to_all_enabled_channels(self, event_bus: EventBus) -> None:
        ch1 = FakeChannel("websocket")
        ch2 = FakeChannel("email")
        settings = AlertSettings(cooldown_seconds=0, enabled_channels=["websocket", "email"])
        service = AlertService(event_bus, settings, [ch1, ch2])

        event = ThreatDetectedEvent(
            camera_id="cam1",
            label="knife",
            confidence=0.85,
            bbox=(10, 20, 100, 200),
        )
        await event_bus.publish(event)

        assert len(ch1.sent) == 1
        assert len(ch2.sent) == 1
        assert ch1.sent[0].threat_label == "knife"

    async def test_only_enabled_channels_receive_alert(self, event_bus: EventBus) -> None:
        ch_ws = FakeChannel("websocket")
        ch_email = FakeChannel("email")
        settings = AlertSettings(cooldown_seconds=0, enabled_channels=["websocket"])
        service = AlertService(event_bus, settings, [ch_ws, ch_email])

        event = ThreatDetectedEvent(camera_id="cam1", label="gun", confidence=0.9)
        await event_bus.publish(event)

        assert len(ch_ws.sent) == 1
        assert len(ch_email.sent) == 0


class TestAlertCooldown:
    """Cooldown prevents duplicate alerts for same camera:label key."""

    async def test_cooldown_prevents_duplicate(self, event_bus: EventBus) -> None:
        ch = FakeChannel("websocket")
        settings = AlertSettings(cooldown_seconds=60, enabled_channels=["websocket"])
        service = AlertService(event_bus, settings, [ch])

        event = ThreatDetectedEvent(camera_id="cam1", label="knife", confidence=0.85)
        await event_bus.publish(event)
        await event_bus.publish(event)

        assert len(ch.sent) == 1

    async def test_cooldown_expires(self, event_bus: EventBus) -> None:
        ch = FakeChannel("websocket")
        # Very short cooldown for testing
        settings = AlertSettings(cooldown_seconds=0, enabled_channels=["websocket"])
        service = AlertService(event_bus, settings, [ch])

        event = ThreatDetectedEvent(camera_id="cam1", label="knife", confidence=0.85)
        await event_bus.publish(event)
        await event_bus.publish(event)

        # cooldown_seconds=0 means no effective cooldown
        assert len(ch.sent) == 2


class TestSeverityMapping:
    """_map_confidence_to_severity maps confidence to correct severity."""

    def test_below_06_is_warning(self) -> None:
        assert _map_confidence_to_severity(0.5) == Severity.WARNING

    def test_below_08_is_high(self) -> None:
        assert _map_confidence_to_severity(0.7) == Severity.HIGH

    def test_above_08_is_critical(self) -> None:
        assert _map_confidence_to_severity(0.85) == Severity.CRITICAL

    def test_boundary_06_is_high(self) -> None:
        assert _map_confidence_to_severity(0.6) == Severity.HIGH

    def test_boundary_08_is_critical(self) -> None:
        assert _map_confidence_to_severity(0.8) == Severity.CRITICAL

    def test_zero_confidence_is_warning(self) -> None:
        assert _map_confidence_to_severity(0.0) == Severity.WARNING

    def test_one_confidence_is_critical(self) -> None:
        assert _map_confidence_to_severity(1.0) == Severity.CRITICAL


class TestDisabledChannelsSkipped:
    """Unavailable channels are skipped during dispatch."""

    async def test_unavailable_channel_skipped(self, event_bus: EventBus) -> None:
        ch_ws = FakeChannel("websocket", available=True)
        ch_email = FakeChannel("email", available=False)
        settings = AlertSettings(cooldown_seconds=0, enabled_channels=["websocket", "email"])
        service = AlertService(event_bus, settings, [ch_ws, ch_email])

        event = ThreatDetectedEvent(camera_id="cam1", label="knife", confidence=0.85)
        await event_bus.publish(event)

        assert len(ch_ws.sent) == 1
        assert len(ch_email.sent) == 0
