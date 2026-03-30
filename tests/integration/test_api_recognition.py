"""Integration tests for recognition endpoints."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest


class TestFacesListEndpoint:
    """GET /api/v1/faces — list known face identities."""

    def test_faces_returns_list(self, test_client) -> None:
        response = test_client.get("/api/v1/faces")
        assert response.status_code == 200
        data = response.json()
        assert "faces" in data
        assert "total" in data
        assert isinstance(data["faces"], list)


class TestRecognizeEndpoint:
    """POST /api/v1/recognize — image-based face recognition."""

    def test_recognize_with_image(self, test_client) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)

        response = test_client.post(
            "/api/v1/recognize",
            files={"file": ("face.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
        assert "total_faces_detected" in data
        assert "inference_ms" in data

    def test_recognize_without_file_returns_422(self, test_client) -> None:
        response = test_client.post("/api/v1/recognize")
        assert response.status_code == 422
