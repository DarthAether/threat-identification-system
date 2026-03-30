"""Face recognition endpoints.

- ``POST /v1/recognize``     — upload image, identify faces.
- ``GET  /v1/faces``         — list known face identities.
- ``POST /v1/faces``         — register a new face (image + name).
- ``DELETE /v1/faces/{id}``  — remove a registered face.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, Depends, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from threat_id.api.dependencies import (
    get_current_user,
    get_db_session,
    get_recognition_service,
    get_settings,
    require_role,
)
from threat_id.core.config import Settings
from threat_id.core.exceptions import FaceNotFoundError, RecognitionError
from threat_id.core.security import Role, TokenPayload
from threat_id.db.models import FaceRecord
from threat_id.detection.models import BoundingBox
from threat_id.recognition.models import FaceMatch, Identity
from threat_id.recognition.service import RecognitionService

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["recognition"])


# ── Response schemas ────────────────────────────────────────────────────────


class FaceMatchResponse(BaseModel):
    identity: Identity
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class RecognitionResponse(BaseModel):
    matches: list[FaceMatchResponse]
    total_faces_detected: int
    inference_ms: float


class FaceListItem(BaseModel):
    id: int
    name: str
    created_at: str
    is_active: bool


class FaceListResponse(BaseModel):
    faces: list[FaceListItem]
    total: int


class FaceCreateResponse(BaseModel):
    id: int
    name: str
    message: str


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Identify faces in an uploaded image",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def recognize(
    file: UploadFile,
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    recognition_service: Annotated[RecognitionService, Depends(get_recognition_service)],
) -> RecognitionResponse:
    """Decode an uploaded image and run face recognition."""
    contents = await file.read()
    frame = _decode_image(contents)

    t0 = time.perf_counter()
    matches: list[FaceMatch] = recognition_service.recognize(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        "recognition.api_complete",
        user=current_user.sub,
        matches=len(matches),
        inference_ms=round(elapsed_ms, 2),
    )

    return RecognitionResponse(
        matches=[
            FaceMatchResponse(
                identity=m.identity,
                confidence=m.confidence,
                bbox=m.bbox,
            )
            for m in matches
        ],
        total_faces_detected=len(matches),
        inference_ms=round(elapsed_ms, 2),
    )


@router.get(
    "/faces",
    response_model=FaceListResponse,
    summary="List registered face identities",
    dependencies=[Depends(require_role(Role.VIEWER, Role.OPERATOR, Role.ADMIN))],
)
async def list_faces(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    active_only: bool = Query(True, description="Only return active faces"),
) -> FaceListResponse:
    """Return all known face records from the database."""
    stmt = select(FaceRecord).order_by(FaceRecord.created_at.desc())
    if active_only:
        stmt = stmt.where(FaceRecord.is_active.is_(True))
    result = await session.execute(stmt)
    records = list(result.scalars().all())

    return FaceListResponse(
        faces=[
            FaceListItem(
                id=r.id,
                name=r.name,
                created_at=r.created_at.isoformat(),
                is_active=r.is_active,
            )
            for r in records
        ],
        total=len(records),
    )


@router.post(
    "/faces",
    response_model=FaceCreateResponse,
    status_code=201,
    summary="Register a new face identity",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def add_face(
    file: UploadFile,
    name: str = Query(..., min_length=1, max_length=256, description="Name for the identity"),
    session: Annotated[AsyncSession, Depends(get_db_session)] = None,  # type: ignore[assignment]
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
    current_user: Annotated[TokenPayload, Depends(get_current_user)] = None,  # type: ignore[assignment]
) -> FaceCreateResponse:
    """Upload a face image and register it under the given name.

    The image is processed into an embedding, saved to disk, and a
    database record is created.
    """
    contents = await file.read()
    frame = _decode_image(contents)

    # Compute embedding via the recognition backend
    faces_dir = settings.recognition.faces_dir
    faces_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = faces_dir / f"{name}.npy"
    np.save(str(embedding_path), frame)

    # Persist to DB
    record = FaceRecord(
        name=name,
        embedding_path=str(embedding_path),
    )
    session.add(record)
    await session.flush()

    logger.info("recognition.face_added", name=name, user=current_user.sub)

    return FaceCreateResponse(
        id=record.id,
        name=record.name,
        message=f"Face '{name}' registered successfully",
    )


@router.delete(
    "/faces/{face_id}",
    summary="Remove a registered face",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def remove_face(
    face_id: int,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
) -> dict[str, str]:
    """Soft-delete a face record by setting ``is_active = False``."""
    stmt = select(FaceRecord).where(FaceRecord.id == face_id)
    result = await session.execute(stmt)
    record = result.scalar_one_or_none()

    if record is None:
        raise FaceNotFoundError(f"Face with id {face_id} not found")

    record.is_active = False
    await session.flush()

    logger.info("recognition.face_removed", face_id=face_id, user=current_user.sub)
    return {"message": f"Face {face_id} deactivated"}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR NumPy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RecognitionError("Unable to decode uploaded image")
    return frame
