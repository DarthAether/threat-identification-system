"""High-level detection service.

Orchestrates a :class:`DetectionBackend`, applies business rules
(confidence filtering, threat-category mapping), and optionally
records Prometheus metrics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import structlog

from threat_id.core.config import DetectionSettings
from threat_id.detection.models import (
    BoundingBox,
    DetectionResult,
    ThreatLevel,
)
from threat_id.detection.protocol import DetectionBackend

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


# ── Optional Prometheus metrics reference ──────────────────────────────────


class MetricsReference(Protocol):
    """Structural interface for a metrics container.

    Allows the service to record counters/histograms without importing
    ``prometheus_client`` at module level.
    """

    detection_count: Any  # Counter
    detection_latency: Any  # Histogram


# ── Threat-level mapping ───────────────────────────────────────────────────

# Labels are mapped based on confidence bands.  A label that appears in the
# configured ``threat_categories`` is always at least LOW.
_CONFIDENCE_BANDS: list[tuple[float, ThreatLevel]] = [
    (0.9, ThreatLevel.CRITICAL),
    (0.75, ThreatLevel.HIGH),
    (0.6, ThreatLevel.MEDIUM),
    (0.0, ThreatLevel.LOW),
]


def _map_threat_level(confidence: float, is_threat: bool) -> ThreatLevel:
    """Determine the threat level from a boolean flag and confidence score."""
    if not is_threat:
        return ThreatLevel.NONE

    for threshold, level in _CONFIDENCE_BANDS:
        if confidence >= threshold:
            return level
    return ThreatLevel.LOW


# ── Service ─────────────────────────────────────────────────────────────────


class DetectionService:
    """Stateless service coordinating detection and post-processing.

    Parameters
    ----------
    settings:
        Detection configuration (thresholds, categories, etc.)
    backend:
        Any object satisfying :class:`DetectionBackend`.
    metrics:
        Optional Prometheus metrics container.  When provided the service
        increments ``detection_count`` and observes ``detection_latency``.
    """

    def __init__(
        self,
        settings: DetectionSettings,
        backend: DetectionBackend,
        *,
        metrics: MetricsReference | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._metrics = metrics

        # Pre-compute a set for O(1) threat category lookups.
        self._threat_categories: set[str] = set(
            cat.lower() for cat in settings.threat_categories
        )
        self._confidence_threshold: float = settings.confidence_threshold

    # ── Public API ──────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        *,
        min_confidence: float | None = None,
    ) -> list[DetectionResult]:
        """Run detection, apply business rules, and return results.

        Parameters
        ----------
        frame:
            ``(H, W, 3)`` uint8 BGR image.
        min_confidence:
            Override the default confidence threshold for this call.

        Returns
        -------
        list[DetectionResult]
            Filtered and enriched detections.
        """
        threshold = min_confidence if min_confidence is not None else self._confidence_threshold

        start = time.perf_counter()
        raw_detections = self._backend.detect(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logger.debug(
            "detection.inference_complete",
            raw_count=len(raw_detections),
            elapsed_ms=round(elapsed_ms, 2),
        )

        results: list[DetectionResult] = []
        for det in raw_detections:
            if det.confidence < threshold:
                continue

            is_threat = det.label.lower() in self._threat_categories
            threat_level = _map_threat_level(det.confidence, is_threat)

            results.append(
                DetectionResult(
                    label=det.label,
                    confidence=round(det.confidence, 4),
                    bbox=BoundingBox(x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2),
                    is_threat=is_threat,
                    threat_level=threat_level,
                )
            )

        self._record_metrics(results, elapsed_ms)
        return results

    # ── Metrics ─────────────────────────────────────────────────────────

    def _record_metrics(
        self,
        results: list[DetectionResult],
        elapsed_ms: float,
    ) -> None:
        """Push counters/histograms if a metrics reference is configured."""
        if self._metrics is None:
            return

        try:
            for result in results:
                self._metrics.detection_count.labels(
                    label=result.label,
                    is_threat=str(result.is_threat).lower(),
                ).inc()
            self._metrics.detection_latency.observe(elapsed_ms / 1000.0)
        except Exception:
            # Metrics must never break the hot path.
            logger.warning("detection.metrics_error", exc_info=True)
