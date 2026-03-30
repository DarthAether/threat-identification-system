"""Tests for threat_id.camera.manager — CameraManager lifecycle."""

from __future__ import annotations

import pytest

from tests.conftest import FakeCamera
from threat_id.camera.manager import CameraManager
from threat_id.core.events import CameraStatusEvent, EventBus


class TestAddRemoveListCameras:
    """Basic add/remove/list operations."""

    async def test_add_camera_registers_it(self, event_bus: EventBus) -> None:
        mgr = CameraManager(event_bus)
        cam = FakeCamera("cam-1")
        await mgr.add_camera(cam)

        assert "cam-1" in mgr.list_cameras()
        assert cam.is_opened

    async def test_remove_camera_deregisters_it(self, event_bus: EventBus) -> None:
        mgr = CameraManager(event_bus)
        cam = FakeCamera("cam-1")
        await mgr.add_camera(cam)
        await mgr.remove_camera("cam-1")

        assert "cam-1" not in mgr.list_cameras()
        assert not cam.is_opened

    async def test_list_cameras_returns_sorted(self, event_bus: EventBus) -> None:
        mgr = CameraManager(event_bus)
        await mgr.add_camera(FakeCamera("cam-c"))
        await mgr.add_camera(FakeCamera("cam-a"))
        await mgr.add_camera(FakeCamera("cam-b"))

        assert mgr.list_cameras() == ["cam-a", "cam-b", "cam-c"]

    async def test_get_camera_returns_camera(self, event_bus: EventBus) -> None:
        mgr = CameraManager(event_bus)
        cam = FakeCamera("cam-1")
        await mgr.add_camera(cam)
        assert mgr.get_camera("cam-1") is cam

    async def test_get_camera_returns_none_for_unknown(self, event_bus: EventBus) -> None:
        mgr = CameraManager(event_bus)
        assert mgr.get_camera("unknown") is None

    async def test_add_duplicate_is_noop(self, event_bus: EventBus) -> None:
        mgr = CameraManager(event_bus)
        cam = FakeCamera("cam-1")
        await mgr.add_camera(cam)
        await mgr.add_camera(cam)

        assert mgr.list_cameras() == ["cam-1"]

    async def test_remove_unknown_is_noop(self, event_bus: EventBus) -> None:
        mgr = CameraManager(event_bus)
        # Should not raise
        await mgr.remove_camera("nonexistent")


class TestStartAllStopAll:
    """Bulk lifecycle operations."""

    async def test_start_all_opens_all_cameras(self, event_bus: EventBus) -> None:
        mgr = CameraManager(event_bus)
        cams = [FakeCamera("cam-1"), FakeCamera("cam-2"), FakeCamera("cam-3")]

        await mgr.start_all(cams)

        assert len(mgr.list_cameras()) == 3
        for cam in cams:
            assert cam.is_opened

    async def test_stop_all_releases_all_cameras(self, event_bus: EventBus) -> None:
        mgr = CameraManager(event_bus)
        cams = [FakeCamera("cam-1"), FakeCamera("cam-2")]
        await mgr.start_all(cams)

        await mgr.stop_all()

        assert mgr.list_cameras() == []
        for cam in cams:
            assert not cam.is_opened

    async def test_start_all_handles_failing_camera(self, event_bus: EventBus) -> None:
        class FailingCamera(FakeCamera):
            async def open(self) -> None:
                raise RuntimeError("device busy")

        mgr = CameraManager(event_bus)
        good_cam = FakeCamera("good-cam")
        bad_cam = FailingCamera("bad-cam")

        await mgr.start_all([good_cam, bad_cam])

        # Good camera should be registered; bad camera should not
        assert "good-cam" in mgr.list_cameras()
        assert "bad-cam" not in mgr.list_cameras()


class TestCameraStatusEvent:
    """CameraStatusEvent is published on add/remove."""

    async def test_connected_event_on_add(self, event_bus: EventBus) -> None:
        received: list[CameraStatusEvent] = []

        async def handler(event: CameraStatusEvent) -> None:
            received.append(event)

        event_bus.subscribe(CameraStatusEvent, handler)
        mgr = CameraManager(event_bus)
        await mgr.add_camera(FakeCamera("cam-1"))

        assert len(received) == 1
        assert received[0].camera_id == "cam-1"
        assert received[0].status == "connected"

    async def test_disconnected_event_on_remove(self, event_bus: EventBus) -> None:
        received: list[CameraStatusEvent] = []

        async def handler(event: CameraStatusEvent) -> None:
            received.append(event)

        event_bus.subscribe(CameraStatusEvent, handler)
        mgr = CameraManager(event_bus)
        await mgr.add_camera(FakeCamera("cam-1"))
        received.clear()

        await mgr.remove_camera("cam-1")

        assert len(received) == 1
        assert received[0].status == "disconnected"

    async def test_error_event_on_failed_open(self, event_bus: EventBus) -> None:
        received: list[CameraStatusEvent] = []

        async def handler(event: CameraStatusEvent) -> None:
            received.append(event)

        event_bus.subscribe(CameraStatusEvent, handler)

        class FailingCamera(FakeCamera):
            async def open(self) -> None:
                raise RuntimeError("device error")

        mgr = CameraManager(event_bus)
        with pytest.raises(RuntimeError):
            await mgr.add_camera(FailingCamera("bad-cam"))

        assert len(received) == 1
        assert received[0].status == "error"
