"""Integration tests for detection endpoint."""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from threat_id.core.security import Role, TokenPayload


class TestDetectEndpoint:
    """POST /api/v1/detect — image upload detection."""

    def test_detect_with_image_returns_detection_response(self, test_client) -> None:
        # Create a minimal valid JPEG-like image using numpy + cv2
        import cv2

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)
        image_bytes = buf.tobytes()

        response = test_client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "frame_width" in data
        assert "frame_height" in data
        assert "inference_ms" in data
        assert isinstance(data["detections"], list)

    def test_detect_without_file_returns_422(self, test_client) -> None:
        response = test_client.post("/api/v1/detect")
        assert response.status_code == 422

    def test_detect_without_auth_returns_401(self, test_client) -> None:
        from threat_id.api.dependencies import get_current_user

        original = test_client.app.dependency_overrides.get(get_current_user)
        test_client.app.dependency_overrides.pop(get_current_user, None)

        response = test_client.post("/api/v1/detect")
        assert response.status_code in (401, 403)

        if original is not None:
            test_client.app.dependency_overrides[get_current_user] = original

    def test_detect_response_includes_threat_info(self, test_client) -> None:
        import cv2

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)

        response = test_client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
        )

        data = response.json()
        for det in data["detections"]:
            assert "label" in det
            assert "confidence" in det
            assert "is_threat" in det
            assert "threat_level" in det
            assert "bbox" in det
