"""Prometheus metrics for the threat-identification system.

Exposes counters, histograms, and gauges covering the full request
lifecycle: camera ingestion, model inference, threat detection,
alerting, and HTTP serving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    make_asgi_app,
)

if TYPE_CHECKING:
    from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)

# ── Metric Definitions ────────────────────────────────────────────────────────

# Latency buckets tuned for real-time video processing (ms-scale) and
# model inference (potentially seconds).
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_HTTP_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

detection_latency_seconds = Histogram(
    "detection_latency_seconds",
    "End-to-end detection pipeline latency in seconds.",
    labelnames=("backend",),
    buckets=_LATENCY_BUCKETS,
)

frames_processed_total = Counter(
    "frames_processed_total",
    "Total video frames processed per camera.",
    labelnames=("camera_id",),
)

threats_detected_total = Counter(
    "threats_detected_total",
    "Total threats detected, by label and camera.",
    labelnames=("label", "camera_id"),
)

alerts_fired_total = Counter(
    "alerts_fired_total",
    "Total alerts dispatched, by channel and success.",
    labelnames=("channel", "success"),
)

face_matches_total = Counter(
    "face_matches_total",
    "Total successful face recognition matches per camera.",
    labelnames=("camera_id",),
)

camera_fps = Gauge(
    "camera_fps",
    "Current frames-per-second being captured per camera.",
    labelnames=("camera_id",),
)

model_inference_seconds = Histogram(
    "model_inference_seconds",
    "Model inference latency in seconds, by model type.",
    labelnames=("model_type",),
    buckets=_LATENCY_BUCKETS,
)

active_cameras = Gauge(
    "active_cameras",
    "Number of cameras currently streaming.",
)

active_websocket_connections = Gauge(
    "active_websocket_connections",
    "Number of active WebSocket connections.",
)

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests received.",
    labelnames=("method", "path", "status"),
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    labelnames=("method", "path"),
    buckets=_HTTP_BUCKETS,
)


# ── Collector Class ───────────────────────────────────────────────────────────


class MetricsCollector:
    """Convenience facade over the raw Prometheus metrics.

    All methods are cheap (no I/O) and safe to call from hot paths.
    """

    # -- Detection pipeline --------------------------------------------------

    @staticmethod
    def observe_detection_latency(backend: str, seconds: float) -> None:
        """Record one detection pipeline invocation latency."""
        detection_latency_seconds.labels(backend=backend).observe(seconds)

    @staticmethod
    def inc_frames_processed(camera_id: str, count: int = 1) -> None:
        """Increment the frame counter for *camera_id*."""
        frames_processed_total.labels(camera_id=camera_id).inc(count)

    @staticmethod
    def inc_threats_detected(label: str, camera_id: str, count: int = 1) -> None:
        """Record one or more threat detections."""
        threats_detected_total.labels(label=label, camera_id=camera_id).inc(count)

    # -- Alerting ------------------------------------------------------------

    @staticmethod
    def inc_alerts_fired(channel: str, *, success: bool) -> None:
        """Record an alert dispatch attempt."""
        alerts_fired_total.labels(channel=channel, success=str(success).lower()).inc()

    # -- Face recognition ----------------------------------------------------

    @staticmethod
    def inc_face_matches(camera_id: str, count: int = 1) -> None:
        """Record successful face matches."""
        face_matches_total.labels(camera_id=camera_id).inc(count)

    # -- Camera gauges -------------------------------------------------------

    @staticmethod
    def set_camera_fps(camera_id: str, fps: float) -> None:
        """Update the current FPS reading for *camera_id*."""
        camera_fps.labels(camera_id=camera_id).set(fps)

    @staticmethod
    def set_active_cameras(count: int) -> None:
        """Set the total number of active cameras."""
        active_cameras.set(count)

    # -- Model inference -----------------------------------------------------

    @staticmethod
    def observe_model_inference(model_type: str, seconds: float) -> None:
        """Record one model inference duration."""
        model_inference_seconds.labels(model_type=model_type).observe(seconds)

    # -- WebSocket connections -----------------------------------------------

    @staticmethod
    def inc_websocket_connections() -> None:
        """Track a new WebSocket connection."""
        active_websocket_connections.inc()

    @staticmethod
    def dec_websocket_connections() -> None:
        """Track a closed WebSocket connection."""
        active_websocket_connections.dec()

    # -- HTTP ----------------------------------------------------------------

    @staticmethod
    def observe_http_request(
        method: str,
        path: str,
        status: int,
        duration: float,
    ) -> None:
        """Record one HTTP request with status and duration."""
        status_str = str(status)
        http_requests_total.labels(method=method, path=path, status=status_str).inc()
        http_request_duration_seconds.labels(method=method, path=path).observe(duration)

    # -- ASGI app for /metrics endpoint --------------------------------------

    @staticmethod
    def create_metrics_app(registry: CollectorRegistry = REGISTRY) -> ASGIApp:
        """Return a Starlette-compatible ASGI app that serves ``/metrics``.

        Mount this on your FastAPI / Starlette application::

            app.mount("/metrics", MetricsCollector.create_metrics_app())
        """
        return make_asgi_app(registry=registry)
