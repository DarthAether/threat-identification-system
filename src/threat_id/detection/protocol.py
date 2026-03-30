"""Detection backend protocol and shared data structures.

Any object that satisfies :class:`DetectionBackend` can be used by
:class:`~threat_id.detection.service.DetectionService` without importing
a concrete implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class RawDetection:
    """A single bounding-box detection returned by a backend.

    Coordinates are in **pixel** space of the input frame.
    """

    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@runtime_checkable
class DetectionBackend(Protocol):
    """Structural interface every detection backend must satisfy.

    The lifecycle is explicit:

    1. ``load_model()`` — allocate GPU/CPU resources.
    2. ``detect()`` — run inference (may be called many times).
    3. ``unload_model()`` — release resources.
    """

    def load_model(self) -> None:
        """Load the model weights into memory."""
        ...

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        """Run inference on a single BGR frame.

        Parameters
        ----------
        frame:
            An ``(H, W, 3)`` uint8 NumPy array in BGR colour order.

        Returns
        -------
        list[RawDetection]
        """
        ...

    def unload_model(self) -> None:
        """Free model resources (GPU memory, file handles, etc.)."""
        ...
