"""Tests for threat_id.detection.service — DetectionService business logic."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import FakeDetectionBackend
from tests.factories import make_frame
from threat_id.core.config import DetectionSettings
from threat_id.detection.models import ThreatLevel
from threat_id.detection.protocol import RawDetection
from threat_id.detection.service import DetectionService, _map_threat_level


class TestDetectFiltering:
    """detect() filters results based on confidence threshold."""

    def test_filters_below_threshold(self) -> None:
        backend = FakeDetectionBackend(
            detections=[
                RawDetection(label="knife", confidence=0.3, x1=0, y1=0, x2=50, y2=50),
                RawDetection(label="gun", confidence=0.8, x1=10, y1=10, x2=60, y2=60),
            ]
        )
        settings = DetectionSettings(confidence_threshold=0.5)
        service = DetectionService(settings, backend)

        results = service.detect(make_frame())
        assert len(results) == 1
        assert results[0].label == "gun"

    def test_override_min_confidence(self) -> None:
        backend = FakeDetectionBackend(
            detections=[
                RawDetection(label="knife", confidence=0.3, x1=0, y1=0, x2=50, y2=50),
                RawDetection(label="gun", confidence=0.8, x1=10, y1=10, x2=60, y2=60),
            ]
        )
        settings = DetectionSettings(confidence_threshold=0.5)
        service = DetectionService(settings, backend)

        results = service.detect(make_frame(), min_confidence=0.2)
        assert len(results) == 2

    def test_all_below_threshold_returns_empty(self) -> None:
        backend = FakeDetectionBackend(
            detections=[
                RawDetection(label="knife", confidence=0.1, x1=0, y1=0, x2=50, y2=50),
            ]
        )
        settings = DetectionSettings(confidence_threshold=0.5)
        service = DetectionService(settings, backend)

        results = service.detect(make_frame())
        assert results == []


class TestDetectThreatCategories:
    """detect() identifies threats based on threat_categories config."""

    def test_knife_is_threat(self) -> None:
        backend = FakeDetectionBackend(
            detections=[
                RawDetection(label="knife", confidence=0.9, x1=0, y1=0, x2=50, y2=50),
            ]
        )
        settings = DetectionSettings(threat_categories=["knife", "gun"])
        service = DetectionService(settings, backend)

        results = service.detect(make_frame())
        assert results[0].is_threat is True

    def test_person_is_not_threat(self) -> None:
        backend = FakeDetectionBackend(
            detections=[
                RawDetection(label="person", confidence=0.95, x1=0, y1=0, x2=50, y2=50),
            ]
        )
        settings = DetectionSettings(threat_categories=["knife", "gun"])
        service = DetectionService(settings, backend)

        results = service.detect(make_frame())
        assert results[0].is_threat is False
        assert results[0].threat_level == ThreatLevel.NONE

    def test_case_insensitive_category_matching(self) -> None:
        backend = FakeDetectionBackend(
            detections=[
                RawDetection(label="Knife", confidence=0.85, x1=0, y1=0, x2=50, y2=50),
            ]
        )
        settings = DetectionSettings(threat_categories=["knife"])
        service = DetectionService(settings, backend)

        results = service.detect(make_frame())
        assert results[0].is_threat is True


class TestDetectThreatLevelMapping:
    """detect() maps to correct ThreatLevel based on confidence bands."""

    def test_critical_at_high_confidence(self) -> None:
        assert _map_threat_level(0.95, is_threat=True) == ThreatLevel.CRITICAL

    def test_high_at_medium_confidence(self) -> None:
        assert _map_threat_level(0.80, is_threat=True) == ThreatLevel.HIGH

    def test_medium_at_lower_confidence(self) -> None:
        assert _map_threat_level(0.65, is_threat=True) == ThreatLevel.MEDIUM

    def test_low_at_lowest_confidence(self) -> None:
        assert _map_threat_level(0.3, is_threat=True) == ThreatLevel.LOW

    def test_none_when_not_threat(self) -> None:
        assert _map_threat_level(0.99, is_threat=False) == ThreatLevel.NONE


class TestDetectWithNoDetections:
    """detect() with no detections returns empty list."""

    def test_empty_detections(self) -> None:
        backend = FakeDetectionBackend(detections=[])
        settings = DetectionSettings(confidence_threshold=0.5)
        service = DetectionService(settings, backend)

        results = service.detect(make_frame())
        assert results == []


class TestDetectWithFakeBackend:
    """detect() works end-to-end with the FakeDetectionBackend fixture."""

    def test_default_fake_backend_returns_results(
        self, fake_yolo_backend: FakeDetectionBackend
    ) -> None:
        settings = DetectionSettings(confidence_threshold=0.5)
        service = DetectionService(settings, fake_yolo_backend)

        results = service.detect(make_frame())
        assert len(results) == 2

        labels = {r.label for r in results}
        assert "knife" in labels
        assert "person" in labels

    def test_default_fake_backend_knife_is_threat(
        self, fake_yolo_backend: FakeDetectionBackend
    ) -> None:
        settings = DetectionSettings(
            confidence_threshold=0.5,
            threat_categories=["knife", "gun"],
        )
        service = DetectionService(settings, fake_yolo_backend)
        results = service.detect(make_frame())

        knife_result = next(r for r in results if r.label == "knife")
        assert knife_result.is_threat is True
        assert knife_result.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)
