"""Protocol definition for camera frame sources.

Any backend (USB webcam via OpenCV, RTSP stream, recorded video file)
implements this protocol so the pipeline can consume frames uniformly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CameraSource(Protocol):
    """Structural sub-typing contract for camera frame sources."""

    async def open(self) -> None:
        """Open the underlying device or stream.

        Raises:
            CameraUnavailableError: If the source cannot be opened.
        """
        ...

    async def read(self) -> tuple[bool, np.ndarray]:
        """Capture the next frame.

        Returns:
            A ``(success, frame)`` tuple.  When ``success`` is False the
            frame array may be empty.

        Raises:
            FrameCaptureError: If reading fails unexpectedly.
        """
        ...

    async def release(self) -> None:
        """Release the underlying device or stream."""
        ...

    @property
    def is_opened(self) -> bool:
        """Whether the source is currently open and producing frames."""
        ...

    @property
    def source_id(self) -> str:
        """Unique, human-readable identifier for this source."""
        ...
