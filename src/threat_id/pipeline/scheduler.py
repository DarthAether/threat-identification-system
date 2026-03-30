"""Pipeline scheduler — drives frame processing at a target FPS.

Each camera runs in its own ``asyncio.Task``.  A shared
``asyncio.Event`` enables cooperative, graceful shutdown.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import structlog

from threat_id.camera.protocol import CameraSource
from threat_id.pipeline.processor import FrameProcessor

logger = structlog.get_logger(__name__)

_CameraGetter = Callable[[str], CameraSource | None]
"""Callback that resolves a camera_id to its source (e.g. CameraManager.get_camera)."""

_DEFAULT_TARGET_FPS: int = 15


class PipelineScheduler:
    """Manages per-camera frame-processing loops as async background tasks."""

    def __init__(
        self,
        processor: FrameProcessor,
        camera_getter: _CameraGetter,
        *,
        target_fps: int = _DEFAULT_TARGET_FPS,
    ) -> None:
        self._processor = processor
        self._camera_getter = camera_getter
        self._target_fps = target_fps
        self._frame_interval: float = 1.0 / max(target_fps, 1)

        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}

    # ── Public API ───────────────────────────────────────────────────

    def start(self, camera_id: str) -> None:
        """Start the processing loop for *camera_id*.

        If the camera is already running this is a no-op.
        """
        if camera_id in self._tasks and not self._tasks[camera_id].done():
            logger.warning("scheduler.already_running", camera_id=camera_id)
            return

        stop_event = asyncio.Event()
        self._stop_events[camera_id] = stop_event

        task = asyncio.create_task(
            self._run_loop(camera_id, stop_event),
            name=f"pipeline-{camera_id}",
        )
        self._tasks[camera_id] = task
        logger.info(
            "scheduler.started",
            camera_id=camera_id,
            target_fps=self._target_fps,
        )

    async def stop(self, camera_id: str) -> None:
        """Signal the processing loop for *camera_id* to stop and await it."""
        stop_event = self._stop_events.pop(camera_id, None)
        if stop_event is not None:
            stop_event.set()

        task = self._tasks.pop(camera_id, None)
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                logger.warning("scheduler.force_cancelled", camera_id=camera_id)

        logger.info("scheduler.stopped", camera_id=camera_id)

    async def stop_all(self) -> None:
        """Stop all running camera loops."""
        camera_ids = list(self._tasks.keys())
        for cam_id in camera_ids:
            await self.stop(cam_id)
        logger.info("scheduler.all_stopped", count=len(camera_ids))

    # ── Internal loop ────────────────────────────────────────────────

    async def _run_loop(
        self,
        camera_id: str,
        stop_event: asyncio.Event,
    ) -> None:
        """Continuously read frames and pipe them through the processor."""
        logger.info("scheduler.loop_started", camera_id=camera_id)

        while not stop_event.is_set():
            t0 = time.monotonic()

            camera = self._camera_getter(camera_id)
            if camera is None or not camera.is_opened:
                logger.warning(
                    "scheduler.camera_unavailable",
                    camera_id=camera_id,
                )
                # Back off when the camera is missing / closed.
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                continue

            try:
                success, frame = await camera.read()
            except Exception:
                logger.exception("scheduler.read_error", camera_id=camera_id)
                await asyncio.sleep(0.1)
                continue

            if not success:
                await asyncio.sleep(0.01)
                continue

            try:
                await self._processor.process_frame(camera_id, frame)
            except Exception:
                logger.exception("scheduler.process_error", camera_id=camera_id)

            # Throttle to maintain target FPS
            elapsed = time.monotonic() - t0
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=sleep_time)
                except asyncio.TimeoutError:
                    pass

        logger.info("scheduler.loop_stopped", camera_id=camera_id)
