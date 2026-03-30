"""Integration tests for camera CRUD endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestCameraListEndpoint:
    """GET /api/v1/cameras — list registered cameras."""

    def test_list_cameras_returns_empty_initially(self, test_client) -> None:
        response = test_client.get("/api/v1/cameras")
        assert response.status_code == 200
        data = response.json()
        assert "cameras" in data
        assert "total" in data
        assert data["total"] == 0

    def test_list_cameras_with_registered_cameras(self, test_client) -> None:
        # Configure mock to return cameras
        mock_cam = MagicMock()
        mock_cam.is_opened = True
        test_client.app.state.camera_manager.list_cameras.return_value = ["cam-1", "cam-2"]
        test_client.app.state.camera_manager.get_camera.return_value = mock_cam

        response = test_client.get("/api/v1/cameras")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

        # Reset
        test_client.app.state.camera_manager.list_cameras.return_value = []


class TestCameraDeleteEndpoint:
    """DELETE /api/v1/cameras/{id} — remove a camera."""

    def test_delete_nonexistent_camera_returns_error(self, test_client) -> None:
        test_client.app.state.camera_manager.get_camera.return_value = None
        response = test_client.delete("/api/v1/cameras/nonexistent")
        # Should return 503 (CameraUnavailableError)
        assert response.status_code in (404, 503)

    def test_delete_existing_camera(self, test_client) -> None:
        mock_cam = MagicMock()
        mock_cam.is_opened = True
        test_client.app.state.camera_manager.get_camera.return_value = mock_cam
        test_client.app.state.camera_manager.remove_camera = AsyncMock()

        response = test_client.delete("/api/v1/cameras/cam-1")
        assert response.status_code == 200
        data = response.json()
        assert "removed" in data.get("message", "").lower() or "message" in data

        # Reset
        test_client.app.state.camera_manager.get_camera.return_value = None
