"""Tests for threat_id.core.events — EventBus pub/sub behaviour."""

from __future__ import annotations

import asyncio

import pytest

from threat_id.core.events import (
    BaseEvent,
    CameraStatusEvent,
    EventBus,
    FaceRecognizedEvent,
    ThreatDetectedEvent,
)


class TestEventBusPublishSubscribe:
    """Basic publish/subscribe on the EventBus."""

    async def test_subscriber_receives_published_event(self, event_bus: EventBus) -> None:
        received: list[ThreatDetectedEvent] = []

        async def handler(event: ThreatDetectedEvent) -> None:
            received.append(event)

        event_bus.subscribe(ThreatDetectedEvent, handler)
        evt = ThreatDetectedEvent(camera_id="cam1", label="knife", confidence=0.9)
        await event_bus.publish(evt)

        assert len(received) == 1
        assert received[0].label == "knife"
        assert received[0].camera_id == "cam1"

    async def test_no_subscribers_does_not_error(self, event_bus: EventBus) -> None:
        evt = ThreatDetectedEvent(camera_id="cam1", label="knife", confidence=0.9)
        # Should not raise
        await event_bus.publish(evt)

    async def test_subscriber_only_receives_matching_type(self, event_bus: EventBus) -> None:
        threat_events: list[ThreatDetectedEvent] = []
        camera_events: list[CameraStatusEvent] = []

        async def threat_handler(event: ThreatDetectedEvent) -> None:
            threat_events.append(event)

        async def camera_handler(event: CameraStatusEvent) -> None:
            camera_events.append(event)

        event_bus.subscribe(ThreatDetectedEvent, threat_handler)
        event_bus.subscribe(CameraStatusEvent, camera_handler)

        await event_bus.publish(ThreatDetectedEvent(camera_id="c1", label="gun", confidence=0.8))
        await event_bus.publish(CameraStatusEvent(camera_id="c2", status="connected"))

        assert len(threat_events) == 1
        assert len(camera_events) == 1


class TestMultipleSubscribers:
    """Multiple handlers for the same event type all get called."""

    async def test_all_subscribers_receive_event(self, event_bus: EventBus) -> None:
        results: list[str] = []

        async def handler_a(event: ThreatDetectedEvent) -> None:
            results.append("a")

        async def handler_b(event: ThreatDetectedEvent) -> None:
            results.append("b")

        async def handler_c(event: ThreatDetectedEvent) -> None:
            results.append("c")

        event_bus.subscribe(ThreatDetectedEvent, handler_a)
        event_bus.subscribe(ThreatDetectedEvent, handler_b)
        event_bus.subscribe(ThreatDetectedEvent, handler_c)

        await event_bus.publish(ThreatDetectedEvent(camera_id="c", label="x", confidence=0.5))

        assert sorted(results) == ["a", "b", "c"]


class TestHandlerErrorIsolation:
    """A failing handler must not prevent other handlers from running."""

    async def test_error_in_one_handler_does_not_crash_others(self, event_bus: EventBus) -> None:
        results: list[str] = []

        async def good_handler_1(event: ThreatDetectedEvent) -> None:
            results.append("good1")

        async def bad_handler(event: ThreatDetectedEvent) -> None:
            raise RuntimeError("handler failed")

        async def good_handler_2(event: ThreatDetectedEvent) -> None:
            results.append("good2")

        event_bus.subscribe(ThreatDetectedEvent, good_handler_1)
        event_bus.subscribe(ThreatDetectedEvent, bad_handler)
        event_bus.subscribe(ThreatDetectedEvent, good_handler_2)

        # Should not raise even though bad_handler fails
        await event_bus.publish(ThreatDetectedEvent(camera_id="c", label="x", confidence=0.5))

        assert "good1" in results
        assert "good2" in results


class TestUnsubscribe:
    """Unsubscribed handlers must no longer receive events."""

    async def test_unsubscribed_handler_not_called(self, event_bus: EventBus) -> None:
        calls: list[str] = []

        async def handler(event: ThreatDetectedEvent) -> None:
            calls.append("called")

        event_bus.subscribe(ThreatDetectedEvent, handler)
        await event_bus.publish(ThreatDetectedEvent(camera_id="c", label="x", confidence=0.5))
        assert len(calls) == 1

        event_bus.unsubscribe(ThreatDetectedEvent, handler)
        await event_bus.publish(ThreatDetectedEvent(camera_id="c", label="x", confidence=0.5))
        assert len(calls) == 1  # unchanged

    async def test_unsubscribe_nonexistent_handler_is_safe(self, event_bus: EventBus) -> None:
        async def handler(event: ThreatDetectedEvent) -> None:
            pass

        # Should not raise
        event_bus.unsubscribe(ThreatDetectedEvent, handler)
