"""Test data factories for consistent, overridable test objects."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from faker import Faker

from threat_id.alerting.models import AlertPayload, Severity
from threat_id.db.models import User
from threat_id.detection.models import BoundingBox, DetectionResult, ThreatLevel

fake = Faker()


def make_detection_result(**overrides: Any) -> DetectionResult:
    """Build a DetectionResult with sensible defaults; override any field."""
    defaults: dict[str, Any] = {
        "label": "knife",
        "confidence": 0.85,
        "bbox": BoundingBox(x1=10, y1=20, x2=100, y2=200),
        "is_threat": True,
        "threat_level": ThreatLevel.HIGH,
    }
    defaults.update(overrides)
    return DetectionResult(**defaults)


def make_alert_payload(**overrides: Any) -> AlertPayload:
    """Build an AlertPayload with sensible defaults; override any field."""
    defaults: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc),
        "camera_id": "cam-01",
        "threat_label": "knife",
        "confidence": 0.85,
        "severity": Severity.HIGH,
        "bbox": (10, 20, 100, 200),
        "correlation_id": fake.uuid4()[:32],
    }
    defaults.update(overrides)
    return AlertPayload(**defaults)


def make_user(**overrides: Any) -> User:
    """Build a User ORM object with sensible defaults; override any field."""
    from threat_id.core.security import hash_password

    defaults: dict[str, Any] = {
        "username": fake.user_name(),
        "hashed_password": hash_password("testpass123"),
        "role": "viewer",
        "is_active": True,
    }
    defaults.update(overrides)
    return User(**defaults)


def make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Return a random uint8 BGR frame of the requested size."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
