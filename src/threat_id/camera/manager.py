"""Camera lifecycle manager.

Maintains a registry of active camera sources, coordinates startup /
shutdown, and publishes ``CameraStatusEvent`` whenever a camera's
connectivity state changes.
"""

from __future__ import annotations

import asyncio

import structlog

from threat_id.camera.protocol import CameraSource
from threat_id.core.events import CameraStatusEvent, EventBus

logger = structlog.get_logger(__name__)


class CameraManager:
    """Manages multiple camera sources with event-driven status tracking."""

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._cameras: dict[str, CameraSource] = {}

    # ── Registry operations ──────────────────────────────────────────

    async def add_camera(self, camera: CameraSource) -> None:
        """Register and open a camera source.

        If the camera is already registered the call is a no-op.
        """
        cam_id = camera.source_id
        if cam_id in self._cameras:
            logger.warning("camera_manager.already_registered", camera_id=cam_id)
            return

        try:
            await camera.open()
        except Exception:
            await self._publish_status(cam_id, "error")
            raise

        self._cameras[cam_id] = camera
        await self._publish_status(cam_id, "connected")
        logger.info("camera_manager.added", camera_id=cam_id)

    async def remove_camera(self, camera_id: str) -> None:
        """Release and deregister a camera source."""
        camera = self._cameras.pop(camera_id, None)
        if camera is None:
            logger.warning("camera_manager.not_found", camera_id=camera_id)
            return

        try:
            await camera.release()
        except Exception:
            logger.exception("camera_manager.release_error", camera_id=camera_id)

        await self._publish_status(camera_id, "disconnected")
        logger.info("camera_manager.removed", camera_id=camera_id)

    def get_camera(self, camera_id: str) -> CameraSource | None:
        """Return a camera source by ID, or ``None``."""
        return self._cameras.get(camera_id)

    def list_cameras(self) -> list[str]:
        """Return sorted IDs of all registered cameras."""
        return sorted(self._cameras.keys())

    # ── Bulk lifecycle ───────────────────────────────────────────────

    async def start_all(self, cameras: list[CameraSource]) -> None:
        """Register and open a batch of cameras concurrently.

        Cameras that fail to open are logged and skipped; remaining
        cameras continue unaffected.
        """
        results = await asyncio.gather(
            *(self.add_camera(cam) for cam in cameras),
            return_exceptions=True,
        )
        for cam, result in zip(cameras, results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "camera_manager.start_failed",
                    camera_id=cam.source_id,
                    error=str(result),
                )

    async def stop_all(self) -> None:
        """Release and deregister all cameras."""
        camera_ids = list(self._cameras.keys())
        for cam_id in camera_ids:
            await self.remove_camera(cam_id)

        logger.info("camera_manager.all_stopped", count=len(camera_ids))

    # ── Internals ────────────────────────────────────────────────────

    async def _publish_status(self, camera_id: str, status: str) -> None:
        await self._bus.publish(
            CameraStatusEvent(camera_id=camera_id, status=status)
        )
