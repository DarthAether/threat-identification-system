"""RTSP camera source with automatic reconnection.

Uses OpenCV's VideoCapture under the hood but adds exponential-backoff
reconnection logic suitable for network-attached IP cameras that may
drop connectivity intermittently.
"""

from __future__ import annotations

import asyncio
import math
from functools import partial
from typing import Any

import cv2
import numpy as np
import structlog

from threat_id.core.config import CameraSettings
from threat_id.core.exceptions import CameraUnavailableError, FrameCaptureError

logger = structlog.get_logger(__name__)

_DEFAULT_BASE_DELAY: float = 1.0
_DEFAULT_MAX_DELAY: float = 60.0
_DEFAULT_MAX_RETRIES: int = 10


class RTSPCameraSource:
    """Reads frames from an RTSP stream with automatic reconnection."""

    def __init__(
        self,
        rtsp_url: str,
        camera_id: str,
        settings: CameraSettings | None = None,
        *,
        base_delay: float = _DEFAULT_BASE_DELAY,
        max_delay: float = _DEFAULT_MAX_DELAY,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._url = rtsp_url
        self._camera_id = camera_id
        self._settings = settings or CameraSettings()

        self._base_delay = base_delay
        self._max_delay = max_delay
        self._max_retries = max_retries

        self._cap: cv2.VideoCapture | None = None
        self._consecutive_failures: int = 0

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def source_id(self) -> str:
        return self._camera_id

    # ── Lifecycle ────────────────────────────────────────────────────

    async def open(self) -> None:
        """Open the RTSP stream, retrying with exponential backoff.

        Raises:
            CameraUnavailableError: If the stream cannot be opened
                after exhausting all retries.
        """
        for attempt in range(1, self._max_retries + 1):
            cap = await self._try_open()
            if cap is not None and cap.isOpened():
                self._apply_settings(cap)
                self._cap = cap
                self._consecutive_failures = 0
                logger.info(
                    "rtsp_source.connected",
                    camera_id=self._camera_id,
                    url=self._safe_url,
                    attempt=attempt,
                )
                return

            delay = self._backoff_delay(attempt)
            logger.warning(
                "rtsp_source.connect_retry",
                camera_id=self._camera_id,
                attempt=attempt,
                delay_s=round(delay, 2),
            )
            await asyncio.sleep(delay)

        raise CameraUnavailableError(
            f"Cannot open RTSP stream after {self._max_retries} attempts: {self._safe_url}",
            detail=f"camera_id={self._camera_id}",
        )

    async def read(self) -> tuple[bool, np.ndarray]:
        """Capture the next frame, reconnecting on transient failures.

        Raises:
            FrameCaptureError: If the read fails and reconnection is
                not possible within the retry budget.
        """
        if not self.is_opened:
            # Attempt transparent reconnect
            try:
                await self.open()
            except CameraUnavailableError as exc:
                raise FrameCaptureError(
                    f"RTSP stream unavailable: {exc}",
                    detail=f"camera_id={self._camera_id}",
                ) from exc

        loop = asyncio.get_running_loop()
        try:
            ret, frame = await loop.run_in_executor(None, self._cap.read)  # type: ignore[union-attr]
        except Exception as exc:
            self._consecutive_failures += 1
            raise FrameCaptureError(
                f"RTSP frame capture failed: {exc}",
                detail=f"camera_id={self._camera_id}",
            ) from exc

        if not ret:
            self._consecutive_failures += 1
            logger.warning(
                "rtsp_source.frame_dropped",
                camera_id=self._camera_id,
                consecutive_failures=self._consecutive_failures,
            )

            if self._consecutive_failures >= self._max_retries:
                await self._reconnect()

            return False, np.empty(0, dtype=np.uint8)

        self._consecutive_failures = 0
        return True, frame

    async def release(self) -> None:
        """Release the RTSP stream."""
        if self._cap is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._cap.release)
            self._cap = None
            logger.info("rtsp_source.released", camera_id=self._camera_id)

    # ── Internals ────────────────────────────────────────────────────

    async def _try_open(self) -> cv2.VideoCapture | None:
        loop = asyncio.get_running_loop()
        try:
            cap = await loop.run_in_executor(
                None,
                partial(cv2.VideoCapture, self._url, cv2.CAP_FFMPEG),
            )
            return cap
        except Exception:
            logger.exception(
                "rtsp_source.open_error",
                camera_id=self._camera_id,
            )
            return None

    async def _reconnect(self) -> None:
        logger.info("rtsp_source.reconnecting", camera_id=self._camera_id)
        await self.release()
        try:
            await self.open()
        except CameraUnavailableError:
            logger.error(
                "rtsp_source.reconnect_failed",
                camera_id=self._camera_id,
            )

    def _backoff_delay(self, attempt: int) -> float:
        delay = self._base_delay * math.pow(2, attempt - 1)
        return min(delay, self._max_delay)

    def _apply_settings(self, cap: Any) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self._settings.fps)

    @property
    def _safe_url(self) -> str:
        """Strip credentials from the URL for safe logging."""
        if "@" in self._url:
            scheme_end = self._url.find("://")
            at_pos = self._url.find("@")
            if scheme_end != -1 and at_pos != -1:
                return self._url[: scheme_end + 3] + "***@" + self._url[at_pos + 1 :]
        return self._url
