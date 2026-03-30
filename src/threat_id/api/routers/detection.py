"""Object detection endpoint.

- ``POST /v1/detect`` — upload an image and receive detection results.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Annotated

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, Depends, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from threat_id.api.dependencies import (
    get_current_user,
    get_db_session,
    get_detection_service,
    require_role,
)
from threat_id.core.security import Role, TokenPayload
from threat_id.detection.models import BoundingBox, DetectionResult, ThreatLevel
from threat_id.detection.service import DetectionService
from threat_id.db.models import DetectionRecord
from threat_id.db.repositories import DetectionRepository

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/detect", tags=["detection"])


# ── Response schemas ────────────────────────────────────────────────────────


class DetectionItemResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    is_threat: bool
    threat_level: ThreatLevel


class DetectionResponse(BaseModel):
    detections: list[DetectionItemResponse]
    frame_width: int
    frame_height: int
    inference_ms: float


# ── Endpoint ────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=DetectionResponse,
    summary="Run object detection on an uploaded image",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def detect(
    file: UploadFile,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    detection_service: Annotated[DetectionService, Depends(get_detection_service)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DetectionResponse:
    """Decode an uploaded image, run the detection pipeline, and persist results."""
    contents = await file.read()
    frame = _decode_image(contents)
    h, w = frame.shape[:2]

    t0 = time.perf_counter()
    results: list[DetectionResult] = detection_service.detect(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Persist detection records
    repo = DetectionRepository(session)
    records = [
        DetectionRecord(
            camera_id="upload",
            label=r.label,
            confidence=r.confidence,
            is_threat=r.is_threat,
            bbox={
                "x1": r.bbox.x1,
                "y1": r.bbox.y1,
                "x2": r.bbox.x2,
                "y2": r.bbox.y2,
            },
        )
        for r in results
    ]
    if records:
        await repo.create_batch(records)

    logger.info(
        "detection.api_complete",
        user=current_user.sub,
        detections=len(results),
        inference_ms=round(elapsed_ms, 2),
    )

    return DetectionResponse(
        detections=[
            DetectionItemResponse(
                label=r.label,
                confidence=r.confidence,
                bbox=r.bbox,
                is_threat=r.is_threat,
                threat_level=r.threat_level,
            )
            for r in results
        ],
        frame_width=w,
        frame_height=h,
        inference_ms=round(elapsed_ms, 2),
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG) into a BGR NumPy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        from threat_id.core.exceptions import DetectionError  # noqa: PLC0415

        raise DetectionError("Unable to decode uploaded image")
    return frame
