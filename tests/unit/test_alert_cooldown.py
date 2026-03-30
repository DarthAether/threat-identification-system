"""Focused tests on alert cooldown logic — per-label independence and expiry."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from threat_id.alerting.service import AlertService
from threat_id.core.config import AlertSettings
from threat_id.core.events import EventBus, ThreatDetectedEvent


class FakeCooldownChannel:
    """Minimal channel that tracks sent payloads."""

    def __init__(self, channel_name: str = "websocket") -> None:
        self._name = channel_name
        self.call_count = 0

    async def send(self, payload) -> None:
        self.call_count += 1

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return True


class TestPerLabelCooldownIndependence:
    """Cooldowns are tracked per camera_id:label key, so different labels are independent."""

    async def test_knife_cooldown_does_not_affect_gun(self, event_bus: EventBus) -> None:
        ch = FakeCooldownChannel()
        settings = AlertSettings(cooldown_seconds=60, enabled_channels=["websocket"])
        service = AlertService(event_bus, settings, [ch])

        knife_event = ThreatDetectedEvent(camera_id="cam1", label="knife", confidence=0.85)
        gun_event = ThreatDetectedEvent(camera_id="cam1", label="gun", confidence=0.9)

        await event_bus.publish(knife_event)
        assert ch.call_count == 1

        # Gun should still fire (different label)
        await event_bus.publish(gun_event)
        assert ch.call_count == 2

        # Knife again should be cooled down
        await event_bus.publish(knife_event)
        assert ch.call_count == 2

    async def test_same_label_different_camera_independent(self, event_bus: EventBus) -> None:
        ch = FakeCooldownChannel()
        settings = AlertSettings(cooldown_seconds=60, enabled_channels=["websocket"])
        service = AlertService(event_bus, settings, [ch])

        event_cam1 = ThreatDetectedEvent(camera_id="cam1", label="knife", confidence=0.85)
        event_cam2 = ThreatDetectedEvent(camera_id="cam2", label="knife", confidence=0.85)

        await event_bus.publish(event_cam1)
        assert ch.call_count == 1

        # Same label but different camera should fire
        await event_bus.publish(event_cam2)
        assert ch.call_count == 2


class TestCooldownResetAfterExpiry:
    """After cooldown_seconds elapse, the same label should fire again."""

    async def test_alert_fires_again_after_cooldown_expires(self, event_bus: EventBus) -> None:
        ch = FakeCooldownChannel()
        # Very short cooldown
        settings = AlertSettings(cooldown_seconds=0, enabled_channels=["websocket"])
        service = AlertService(event_bus, settings, [ch])

        event = ThreatDetectedEvent(camera_id="cam1", label="knife", confidence=0.85)

        await event_bus.publish(event)
        assert ch.call_count == 1

        # With cooldown_seconds=0, the check `elapsed < cooldown_seconds` is always False
        await event_bus.publish(event)
        assert ch.call_count == 2

    async def test_multiple_labels_all_reset(self, event_bus: EventBus) -> None:
        ch = FakeCooldownChannel()
        settings = AlertSettings(cooldown_seconds=0, enabled_channels=["websocket"])
        service = AlertService(event_bus, settings, [ch])

        for label in ["knife", "gun", "rifle"]:
            event = ThreatDetectedEvent(camera_id="cam1", label=label, confidence=0.85)
            await event_bus.publish(event)

        assert ch.call_count == 3

        # Fire them all again
        for label in ["knife", "gun", "rifle"]:
            event = ThreatDetectedEvent(camera_id="cam1", label=label, confidence=0.85)
            await event_bus.publish(event)

        assert ch.call_count == 6
