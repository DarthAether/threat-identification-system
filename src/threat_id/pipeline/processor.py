"""Single-frame processing pipeline.

Orchestrates the detect -> recognise -> publish flow for one video
frame.  The processor is stateless; all mutable context lives in the
services it delegates to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import structlog

from threat_id.core.events import (
    EventBus,
    FaceRecognizedEvent,
    ThreatDetectedEvent,
)

logger = structlog.get_logger(__name__)


# ── Lightweight protocols for upstream services ──────────────────────


class DetectionService(Protocol):
    """Minimal contract for the object-detection backend."""

    async def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return a list of detections.

        Each dict contains at least:
        ``label`` (str), ``confidence`` (float), ``bbox`` (tuple[int,int,int,int]).
        """
        ...


class RecognitionService(Protocol):
    """Minimal contract for the face-recognition backend."""

    async def recognize(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return a list of face matches.

        Each dict contains at least:
        ``identity`` (str), ``confidence`` (float).
        """
        ...


# ── Result container ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Immutable summary of a single frame's processing run."""

    detections: list[dict[str, Any]] = field(default_factory=list)
    face_matches: list[dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ── Processor ────────────────────────────────────────────────────────


class FrameProcessor:
    """Runs detection and recognition on a single frame and emits events."""

    def __init__(
        self,
        detection_service: DetectionService,
        recognition_service: RecognitionService,
        event_bus: EventBus,
    ) -> None:
        self._detector = detection_service
        self._recogniser = recognition_service
        self._bus = event_bus

    async def process_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
    ) -> ProcessingResult:
        """Run the full detect + recognise pipeline for *frame*.

        Returns a ``ProcessingResult`` summarising everything that was
        found, regardless of whether downstream events succeed.
        """
        t0 = time.perf_counter()

        detections = await self._detector.detect(frame)
        face_matches = await self._recogniser.recognize(frame)

        # Publish threat events
        for det in detections:
            bbox_raw = det.get("bbox", (0, 0, 0, 0))
            bbox: tuple[int, int, int, int] = (
                int(bbox_raw[0]),
                int(bbox_raw[1]),
                int(bbox_raw[2]),
                int(bbox_raw[3]),
            )
            await self._bus.publish(
                ThreatDetectedEvent(
                    camera_id=camera_id,
                    label=det["label"],
                    confidence=det["confidence"],
                    bbox=bbox,
                )
            )

        # Publish face-recognition events
        for match in face_matches:
            await self._bus.publish(
                FaceRecognizedEvent(
                    camera_id=camera_id,
                    identity=match["identity"],
                    confidence=match["confidence"],
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1_000

        logger.debug(
            "frame_processor.done",
            camera_id=camera_id,
            detections=len(detections),
            faces=len(face_matches),
            time_ms=round(elapsed_ms, 2),
        )

        return ProcessingResult(
            detections=detections,
            face_matches=face_matches,
            processing_time_ms=elapsed_ms,
        )
