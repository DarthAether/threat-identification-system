"""Locust load test for the Threat Identification System API.

Run with:
    locust -f tests/load/locustfile.py --host http://localhost:8000

Environment variables:
    LOCUST_USERS        Target number of concurrent users (default: 10)
    LOCUST_SPAWN_RATE   Users spawned per second (default: 2)
    AUTH_TOKEN           Bearer token for authenticated endpoints
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
from locust import HttpUser, between, task


def _make_test_jpeg() -> bytes:
    """Generate a small synthetic JPEG image for upload tests."""
    try:
        import cv2

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Draw a small rectangle to make it non-trivial
        frame[10:50, 10:50] = [128, 64, 32]
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()
    except ImportError:
        # Minimal valid JPEG if cv2 is unavailable
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05"
            b"\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\xa0\x00\x00\x00"
            b"\xff\xd9"
        )


# Pre-generate the test image once
_TEST_IMAGE_BYTES = _make_test_jpeg()


class ThreatIdUser(HttpUser):
    """Simulates a user interacting with the Threat Identification System API."""

    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        """Set up authentication headers if a token is available."""
        token = os.environ.get("AUTH_TOKEN", "")
        if token:
            self.client.headers["Authorization"] = f"Bearer {token}"

    @task(5)
    def health_check(self) -> None:
        """Hit the liveness endpoint (highest weight)."""
        self.client.get("/health", name="/health")

    @task(2)
    def ready_check(self) -> None:
        """Hit the readiness endpoint."""
        self.client.get("/ready", name="/ready")

    @task(1)
    def detect_image(self) -> None:
        """Upload a small test image to the detection endpoint."""
        self.client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(_TEST_IMAGE_BYTES), "image/jpeg")},
            name="/api/v1/detect",
        )

    @task(1)
    def list_alerts(self) -> None:
        """Fetch the recent alerts list."""
        self.client.get(
            "/api/v1/alerts",
            params={"limit": 20},
            name="/api/v1/alerts",
        )

    @task(1)
    def list_cameras(self) -> None:
        """Fetch the camera list."""
        self.client.get("/api/v1/cameras", name="/api/v1/cameras")
