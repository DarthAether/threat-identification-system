"""Pydantic models for the face recognition domain and API layer."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from threat_id.detection.models import BoundingBox


# ── Domain Models ───────────────────────────────────────────────────────────


class Identity(BaseModel):
    """Resolved identity of a recognised person."""

    id: str = Field(description="Unique identifier (database PK or slug).")
    name: str = Field(description="Human-readable display name.")


class FaceMatch(BaseModel):
    """Single recognition result — a face matched to a known identity."""

    identity: Identity
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    bbox: BoundingBox


# ── API Models ──────────────────────────────────────────────────────────────


class RecognitionResponse(BaseModel):
    """Response payload for the ``POST /recognize`` endpoint."""

    camera_id: str
    matches: list[FaceMatch]
    total_faces_detected: int = Field(
        description="Number of faces detected in the frame (matched + unmatched)."
    )
    inference_ms: float = Field(description="Wall-clock inference time in milliseconds.")
