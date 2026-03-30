"""Tests for threat_id.pipeline.processor — FrameProcessor orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.factories import make_frame
from threat_id.core.events import (
    EventBus,
    FaceRecognizedEvent,
    ThreatDetectedEvent,
)
from threat_id.pipeline.processor import FrameProcessor, ProcessingResult


class FakeDetectorForPipeline:
    """Fake detection service returning dict-based results for the processor."""

    def __init__(self, detections: list[dict[str, Any]] | None = None) -> None:
        self._detections = detections or []

    async def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return list(self._detections)


class FakeRecognizerForPipeline:
    """Fake recognition service returning dict-based results for the processor."""

    def __init__(self, matches: list[dict[str, Any]] | None = None) -> None:
        self._matches = matches or []

    async def recognize(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return list(self._matches)


class TestProcessFrameOrchestration:
    """process_frame orchestrates detection + recognition correctly."""

    async def test_returns_processing_result(self, event_bus: EventBus) -> None:
        detector = FakeDetectorForPipeline()
        recognizer = FakeRecognizerForPipeline()
        processor = FrameProcessor(detector, recognizer, event_bus)

        result = await processor.process_frame("cam1", make_frame())

        assert isinstance(result, ProcessingResult)
        assert result.detections == []
        assert result.face_matches == []

    async def test_detections_and_faces_returned(self, event_bus: EventBus) -> None:
        detector = FakeDetectorForPipeline(
            detections=[
                {"label": "knife", "confidence": 0.9, "bbox": (10, 20, 100, 200)},
            ]
        )
        recognizer = FakeRecognizerForPipeline(
            matches=[
                {"identity": "alice", "confidence": 0.95},
            ]
        )
        processor = FrameProcessor(detector, recognizer, event_bus)

        result = await processor.process_frame("cam1", make_frame())

        assert len(result.detections) == 1
        assert len(result.face_matches) == 1
        assert result.detections[0]["label"] == "knife"
        assert result.face_matches[0]["identity"] == "alice"


class TestThreatDetectedEventPublished:
    """ThreatDetectedEvent is published when threats are found."""

    async def test_publishes_threat_event(self, event_bus: EventBus) -> None:
        received: list[ThreatDetectedEvent] = []

        async def handler(event: ThreatDetectedEvent) -> None:
            received.append(event)

        event_bus.subscribe(ThreatDetectedEvent, handler)

        detector = FakeDetectorForPipeline(
            detections=[
                {"label": "gun", "confidence": 0.88, "bbox": (10, 20, 100, 200)},
            ]
        )
        recognizer = FakeRecognizerForPipeline()
        processor = FrameProcessor(detector, recognizer, event_bus)

        await processor.process_frame("cam-3", make_frame())

        assert len(received) == 1
        assert received[0].label == "gun"
        assert received[0].camera_id == "cam-3"
        assert received[0].confidence == 0.88

    async def test_multiple_detections_publish_multiple_events(self, event_bus: EventBus) -> None:
        received: list[ThreatDetectedEvent] = []

        async def handler(event: ThreatDetectedEvent) -> None:
            received.append(event)

        event_bus.subscribe(ThreatDetectedEvent, handler)

        detector = FakeDetectorForPipeline(
            detections=[
                {"label": "knife", "confidence": 0.9, "bbox": (0, 0, 50, 50)},
                {"label": "gun", "confidence": 0.8, "bbox": (60, 60, 120, 120)},
            ]
        )
        processor = FrameProcessor(detector, FakeRecognizerForPipeline(), event_bus)

        await processor.process_frame("cam1", make_frame())

        assert len(received) == 2
        labels = {e.label for e in received}
        assert labels == {"knife", "gun"}


class TestFaceRecognizedEventPublished:
    """FaceRecognizedEvent is published when faces are matched."""

    async def test_publishes_face_event(self, event_bus: EventBus) -> None:
        received: list[FaceRecognizedEvent] = []

        async def handler(event: FaceRecognizedEvent) -> None:
            received.append(event)

        event_bus.subscribe(FaceRecognizedEvent, handler)

        recognizer = FakeRecognizerForPipeline(
            matches=[{"identity": "bob", "confidence": 0.92}]
        )
        processor = FrameProcessor(FakeDetectorForPipeline(), recognizer, event_bus)

        await processor.process_frame("cam-5", make_frame())

        assert len(received) == 1
        assert received[0].identity == "bob"
        assert received[0].camera_id == "cam-5"


class TestProcessingTimeRecorded:
    """processing_time_ms is recorded in the result."""

    async def test_processing_time_is_positive(self, event_bus: EventBus) -> None:
        processor = FrameProcessor(
            FakeDetectorForPipeline(),
            FakeRecognizerForPipeline(),
            event_bus,
        )
        result = await processor.process_frame("cam1", make_frame())
        assert result.processing_time_ms >= 0.0

    async def test_processing_time_is_float(self, event_bus: EventBus) -> None:
        processor = FrameProcessor(
            FakeDetectorForPipeline(),
            FakeRecognizerForPipeline(),
            event_bus,
        )
        result = await processor.process_frame("cam1", make_frame())
        assert isinstance(result.processing_time_ms, float)
