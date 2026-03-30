"""Pydantic models for detection API requests, responses, and domain objects."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# ── Enums ───────────────────────────────────────────────────────────────────


class ThreatLevel(StrEnum):
    """Ordered severity tiers — higher ordinal means higher risk."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Domain Objects ──────────────────────────────────────────────────────────


class BoundingBox(BaseModel):
    """Pixel-space axis-aligned bounding box. All coordinates must be >= 0."""

    x1: Annotated[int, Field(ge=0)]
    y1: Annotated[int, Field(ge=0)]
    x2: Annotated[int, Field(ge=0)]
    y2: Annotated[int, Field(ge=0)]

    @field_validator("x2")
    @classmethod
    def _x2_gte_x1(cls, v: int, info: object) -> int:
        # info.data may not yet contain x1 during partial construction
        data = getattr(info, "data", {})
        x1 = data.get("x1")
        if x1 is not None and v < x1:
            msg = f"x2 ({v}) must be >= x1 ({x1})"
            raise ValueError(msg)
        return v

    @field_validator("y2")
    @classmethod
    def _y2_gte_y1(cls, v: int, info: object) -> int:
        data = getattr(info, "data", {})
        y1 = data.get("y1")
        if y1 is not None and v < y1:
            msg = f"y2 ({v}) must be >= y1 ({y1})"
            raise ValueError(msg)
        return v


class DetectionResult(BaseModel):
    """Single detection after post-processing (confidence filter + threat mapping)."""

    label: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    bbox: BoundingBox
    is_threat: bool
    threat_level: ThreatLevel


# ── API Models ──────────────────────────────────────────────────────────────


class DetectionRequest(BaseModel):
    """Payload for the ``POST /detect`` endpoint.

    The image is expected as a base64-encoded JPEG/PNG.
    """

    image_b64: str = Field(..., description="Base64-encoded image bytes (JPEG or PNG).")
    camera_id: str = Field(default="default", description="Originating camera identifier.")
    min_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override per-request confidence threshold.",
    )


class DetectionResponse(BaseModel):
    """Response returned by the detection endpoint."""

    camera_id: str
    detections: list[DetectionResult]
    frame_width: int
    frame_height: int
    inference_ms: float = Field(description="Wall-clock inference time in milliseconds.")
