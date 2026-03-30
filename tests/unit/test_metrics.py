"""Tests for threat_id.observability.metrics — MetricsCollector facade."""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY, CollectorRegistry

from threat_id.observability.metrics import (
    MetricsCollector,
    alerts_fired_total,
    detection_latency_seconds,
    frames_processed_total,
    threats_detected_total,
)


class TestRecordDetection:
    """MetricsCollector.inc_threats_detected increments the counter."""

    def test_threats_detected_counter_increments(self) -> None:
        before = threats_detected_total.labels(label="knife", camera_id="cam-1")._value.get()
        MetricsCollector.inc_threats_detected("knife", "cam-1")
        after = threats_detected_total.labels(label="knife", camera_id="cam-1")._value.get()
        assert after == before + 1

    def test_frames_processed_counter_increments(self) -> None:
        before = frames_processed_total.labels(camera_id="cam-test")._value.get()
        MetricsCollector.inc_frames_processed("cam-test", 5)
        after = frames_processed_total.labels(camera_id="cam-test")._value.get()
        assert after == before + 5

    def test_alerts_fired_counter_increments(self) -> None:
        before = alerts_fired_total.labels(channel="email", success="true")._value.get()
        MetricsCollector.inc_alerts_fired("email", success=True)
        after = alerts_fired_total.labels(channel="email", success="true")._value.get()
        assert after == before + 1


class TestRecordLatency:
    """MetricsCollector.observe_detection_latency records histogram samples."""

    def test_observe_detection_latency(self) -> None:
        bucket_before = detection_latency_seconds.labels(backend="yolo")._sum.get()
        MetricsCollector.observe_detection_latency("yolo", 0.05)
        bucket_after = detection_latency_seconds.labels(backend="yolo")._sum.get()
        assert bucket_after >= bucket_before + 0.05

    def test_observe_model_inference(self) -> None:
        from threat_id.observability.metrics import model_inference_seconds

        bucket_before = model_inference_seconds.labels(model_type="facenet")._sum.get()
        MetricsCollector.observe_model_inference("facenet", 0.1)
        bucket_after = model_inference_seconds.labels(model_type="facenet")._sum.get()
        assert bucket_after >= bucket_before + 0.1


class TestGaugeMetrics:
    """Gauge metrics can be set and read."""

    def test_set_camera_fps(self) -> None:
        from threat_id.observability.metrics import camera_fps

        MetricsCollector.set_camera_fps("cam-gauge-test", 29.5)
        value = camera_fps.labels(camera_id="cam-gauge-test")._value.get()
        assert value == 29.5

    def test_set_active_cameras(self) -> None:
        from threat_id.observability.metrics import active_cameras

        MetricsCollector.set_active_cameras(3)
        assert active_cameras._value.get() == 3

    def test_websocket_connections_inc_dec(self) -> None:
        from threat_id.observability.metrics import active_websocket_connections

        before = active_websocket_connections._value.get()
        MetricsCollector.inc_websocket_connections()
        assert active_websocket_connections._value.get() == before + 1
        MetricsCollector.dec_websocket_connections()
        assert active_websocket_connections._value.get() == before


class TestHttpMetrics:
    """HTTP request metrics are recorded properly."""

    def test_observe_http_request(self) -> None:
        from threat_id.observability.metrics import http_requests_total

        before = http_requests_total.labels(
            method="GET", path="/health", status="200"
        )._value.get()
        MetricsCollector.observe_http_request("GET", "/health", 200, 0.01)
        after = http_requests_total.labels(
            method="GET", path="/health", status="200"
        )._value.get()
        assert after == before + 1
