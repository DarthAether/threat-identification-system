"""Shared pytest fixtures for the threat-identification-system test suite."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
from sqlalchemy import event as sa_event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from threat_id.alerting.models import AlertPayload
from threat_id.core.config import (
    AlertSettings,
    ApiSettings,
    CameraSettings,
    DatabaseSettings,
    DetectionSettings,
    EmailSettings,
    JwtSettings,
    ObservabilitySettings,
    RecognitionSettings,
    RedisSettings,
    Settings,
    WebhookSettings,
)
from threat_id.core.events import EventBus
from threat_id.db.models import Base
from threat_id.detection.protocol import DetectionBackend, RawDetection
from threat_id.recognition.protocol import RawFaceMatch, RecognitionBackend

from tests.factories import make_frame


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    """Return a Settings instance with test-safe values."""
    return Settings(
        api=ApiSettings(host="127.0.0.1", port=9999, debug=True, log_level="warning"),
        jwt=JwtSettings(secret_key="test-secret-key-do-not-use-in-prod"),
        database=DatabaseSettings(
            host="localhost",
            port=5432,
            name="test_db",
            user="test_user",
            password="test_pass",
        ),
        redis=RedisSettings(url="redis://localhost:6379/15"),
        detection=DetectionSettings(
            backend="yolo",
            model_path="fake.pt",
            confidence_threshold=0.5,
            device="cpu",
            threat_categories=["knife", "gun", "rifle", "pistol"],
        ),
        recognition=RecognitionSettings(
            model_name="Facenet",
            similarity_threshold=0.6,
            faces_dir=tmp_path / "faces",
        ),
        camera=CameraSettings(default_source="0", frame_width=640, frame_height=480, fps=10),
        alert=AlertSettings(cooldown_seconds=30, enabled_channels=["websocket"]),
        email=EmailSettings(
            host="localhost",
            port=1025,
            username="",
            password="",
            sender="",
            recipients=[],
        ),
        webhook=WebhookSettings(url="", timeout_seconds=5),
        observability=ObservabilitySettings(
            prometheus_enabled=False,
            audit_log_enabled=False,
        ),
    )


# ---------------------------------------------------------------------------
# Fake Detection Backend
# ---------------------------------------------------------------------------


class FakeDetectionBackend:
    """In-memory detection backend returning canned results."""

    def __init__(self, detections: list[RawDetection] | None = None) -> None:
        self._detections = detections or [
            RawDetection(label="knife", confidence=0.92, x1=10, y1=20, x2=100, y2=200),
            RawDetection(label="person", confidence=0.88, x1=50, y1=50, x2=200, y2=400),
        ]
        self._loaded = False

    def load_model(self) -> None:
        self._loaded = True

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        return list(self._detections)

    def unload_model(self) -> None:
        self._loaded = False


@pytest.fixture()
def fake_yolo_backend() -> FakeDetectionBackend:
    """Return a FakeDetectionBackend with canned results."""
    return FakeDetectionBackend()


# ---------------------------------------------------------------------------
# Fake Recognition Backend
# ---------------------------------------------------------------------------


class FakeRecognitionBackend:
    """In-memory recognition backend returning canned results."""

    def __init__(self, matches: list[RawFaceMatch] | None = None) -> None:
        self._matches = matches or [
            RawFaceMatch(
                identity="john_doe",
                confidence=0.95,
                x1=30,
                y1=40,
                x2=130,
                y2=180,
            ),
        ]

    def compute_embedding(self, face_image: np.ndarray) -> np.ndarray:
        return np.random.default_rng(0).standard_normal(128).astype(np.float32)

    def find_matches(
        self,
        frame: np.ndarray,
        known_embeddings: dict[str, np.ndarray],
    ) -> list[RawFaceMatch]:
        return list(self._matches)


@pytest.fixture()
def fake_deepface_backend() -> FakeRecognitionBackend:
    """Return a FakeRecognitionBackend with canned results."""
    return FakeRecognitionBackend()


# ---------------------------------------------------------------------------
# Fake Camera
# ---------------------------------------------------------------------------


class FakeCamera:
    """Minimal fake satisfying the CameraSource protocol."""

    def __init__(self, source_id: str = "fake-cam-0") -> None:
        self._source_id = source_id
        self._opened = False
        self._frame = make_frame()

    async def open(self) -> None:
        self._opened = True

    async def read(self) -> tuple[bool, np.ndarray]:
        return True, self._frame.copy()

    async def release(self) -> None:
        self._opened = False

    @property
    def is_opened(self) -> bool:
        return self._opened

    @property
    def source_id(self) -> str:
        return self._source_id


@pytest.fixture()
def fake_camera() -> FakeCamera:
    """Return a FakeCamera yielding pre-built frames."""
    return FakeCamera()


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------


@pytest.fixture()
def event_bus() -> EventBus:
    """Return a fresh EventBus."""
    return EventBus()


# ---------------------------------------------------------------------------
# Async Database Session (SQLite)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create an async SQLite engine, make all tables, yield a session, then tear down."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # SQLite needs PRAGMA foreign_keys for each connection
    @sa_event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# ---------------------------------------------------------------------------
# FastAPI Test Client
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_client(test_settings: Settings, fake_yolo_backend: FakeDetectionBackend, fake_deepface_backend: FakeRecognitionBackend, db_session: AsyncSession):
    """Create a FastAPI TestClient with dependency overrides for all fakes."""
    from unittest.mock import AsyncMock, MagicMock

    from fastapi.testclient import TestClient

    from threat_id.api.app import create_app
    from threat_id.api.dependencies import (
        get_current_user,
        get_db_session,
        get_detection_service,
        get_recognition_service,
        get_settings,
    )
    from threat_id.core.security import Role, TokenPayload
    from threat_id.detection.service import DetectionService

    # Build the app without lifespan (we override deps instead)
    app = create_app(settings=test_settings)

    # Create real detection service with fake backend
    detection_service = DetectionService(test_settings.detection, fake_yolo_backend)

    # Fake recognition service
    fake_recognition_service = MagicMock()
    fake_recognition_service.recognize.return_value = []

    # Override dependencies
    async def _override_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def _override_settings() -> Settings:
        return test_settings

    def _override_detection_service() -> DetectionService:
        return detection_service

    def _override_recognition_service():
        return fake_recognition_service

    def _override_current_user() -> TokenPayload:
        from datetime import datetime, timedelta, timezone

        return TokenPayload(
            sub="testuser",
            role=Role.ADMIN,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            token_type="access",
        )

    app.dependency_overrides[get_db_session] = _override_session
    app.dependency_overrides[get_settings] = _override_settings
    app.dependency_overrides[get_detection_service] = _override_detection_service
    app.dependency_overrides[get_recognition_service] = _override_recognition_service
    app.dependency_overrides[get_current_user] = _override_current_user

    # Also set app.state attributes the routers may reference
    app.state.settings = test_settings
    app.state.detection_service = detection_service
    app.state.recognition_service = fake_recognition_service

    # Camera manager and event bus on state
    camera_manager_mock = MagicMock()
    camera_manager_mock.list_cameras.return_value = []
    camera_manager_mock.get_camera.return_value = None
    app.state.camera_manager = camera_manager_mock
    app.state.event_bus = EventBus()
    app.state.ws_manager = MagicMock()
    app.state.health_checker = None
    app.state.metrics_collector = None
    app.state.pipeline_scheduler = MagicMock()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
