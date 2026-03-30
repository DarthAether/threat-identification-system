"""OpenCV-backed camera source for local USB / built-in cameras.

All ``cv2.VideoCapture`` calls are blocking; they are wrapped with
``run_in_executor`` to keep the async event loop responsive.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import cv2
import numpy as np
import structlog

from threat_id.core.config import CameraSettings
from threat_id.core.exceptions import CameraUnavailableError, FrameCaptureError

logger = structlog.get_logger(__name__)


class OpenCVCameraSource:
    """Reads frames from a local device via OpenCV's VideoCapture."""

    def __init__(
        self,
        source: int | str,
        camera_id: str,
        settings: CameraSettings | None = None,
    ) -> None:
        self._source = source
        self._camera_id = camera_id
        self._settings = settings or CameraSettings()
        self._cap: cv2.VideoCapture | None = None

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def source_id(self) -> str:
        return self._camera_id

    # ── Lifecycle ────────────────────────────────────────────────────

    async def open(self) -> None:
        """Open the video capture device and apply settings.

        Raises:
            CameraUnavailableError: If the device cannot be opened.
        """
        loop = asyncio.get_running_loop()
        cap = await loop.run_in_executor(None, partial(cv2.VideoCapture, self._source))

        if not cap.isOpened():
            raise CameraUnavailableError(
                f"Cannot open camera source: {self._source}",
                detail=f"camera_id={self._camera_id}",
            )

        self._apply_settings(cap)
        self._cap = cap

        logger.info(
            "opencv_source.opened",
            camera_id=self._camera_id,
            source=self._source,
            width=self._settings.frame_width,
            height=self._settings.frame_height,
            fps=self._settings.fps,
        )

    async def read(self) -> tuple[bool, np.ndarray]:
        """Capture the next frame from the device.

        Raises:
            FrameCaptureError: If the underlying read call throws.
        """
        if not self.is_opened:
            raise FrameCaptureError(
                "Camera is not opened",
                detail=f"camera_id={self._camera_id}",
            )

        loop = asyncio.get_running_loop()
        try:
            ret, frame = await loop.run_in_executor(None, self._cap.read)  # type: ignore[union-attr]
        except Exception as exc:
            raise FrameCaptureError(
                f"Frame capture failed: {exc}",
                detail=f"camera_id={self._camera_id}",
            ) from exc

        if not ret:
            return False, np.empty(0, dtype=np.uint8)

        return True, frame

    async def release(self) -> None:
        """Release the video capture device."""
        if self._cap is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._cap.release)
            self._cap = None
            logger.info("opencv_source.released", camera_id=self._camera_id)

    # ── Internals ────────────────────────────────────────────────────

    def _apply_settings(self, cap: Any) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self._settings.fps)
