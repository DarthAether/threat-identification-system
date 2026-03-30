"""Camera management endpoints.

- ``GET    /v1/cameras``            — list registered cameras.
- ``POST   /v1/cameras``            — add a new camera source.
- ``DELETE /v1/cameras/{id}``       — remove a camera.
- ``POST   /v1/cameras/{id}/start`` — start processing pipeline.
- ``POST   /v1/cameras/{id}/stop``  — stop processing pipeline.
- ``GET    /v1/cameras/{id}/status`` — camera health status.
"""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from threat_id.api.dependencies import (
    get_camera_manager,
    get_current_user,
    require_role,
)
from threat_id.camera.manager import CameraManager
from threat_id.core.exceptions import CameraError, CameraUnavailableError
from threat_id.core.security import Role, TokenPayload

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/cameras", tags=["cameras"])


# ── Request / Response schemas ──────────────────────────────────────────────


class CameraCreateRequest(BaseModel):
    device_id: str = Field(
        ...,
        description="Device index (e.g. '0') or RTSP URL",
        min_length=1,
    )
    name: str = Field(
        default="",
        max_length=256,
        description="Human-readable camera name",
    )


class CameraInfo(BaseModel):
    id: str
    is_opened: bool


class CameraListResponse(BaseModel):
    cameras: list[CameraInfo]
    total: int


class CameraStatusResponse(BaseModel):
    id: str
    is_opened: bool
    is_processing: bool


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=CameraListResponse,
    summary="List all registered cameras",
    dependencies=[Depends(require_role(Role.VIEWER, Role.OPERATOR, Role.ADMIN))],
)
async def list_cameras(
    camera_manager: Annotated[CameraManager, Depends(get_camera_manager)],
) -> CameraListResponse:
    """Return metadata for every registered camera."""
    camera_ids = camera_manager.list_cameras()
    items: list[CameraInfo] = []
    for cid in camera_ids:
        cam = camera_manager.get_camera(cid)
        items.append(
            CameraInfo(
                id=cid,
                is_opened=cam.is_opened if cam else False,
            )
        )
    return CameraListResponse(cameras=items, total=len(items))


@router.post(
    "",
    status_code=201,
    summary="Register a new camera",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def add_camera(
    body: CameraCreateRequest,
    camera_manager: Annotated[CameraManager, Depends(get_camera_manager)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    request: Request,
) -> dict[str, str]:
    """Add a new camera source (USB device or RTSP stream)."""
    device = body.device_id

    # Determine source type
    if device.startswith("rtsp://") or device.startswith("rtsps://"):
        from threat_id.camera.rtsp_source import RtspSource  # noqa: PLC0415

        source = RtspSource(url=device, source_id=body.name or device)
    else:
        from threat_id.camera.opencv_source import OpenCVSource  # noqa: PLC0415

        source = OpenCVSource(device_index=device, source_id=body.name or device)

    await camera_manager.add_camera(source)

    logger.info("camera.added", device=device, user=current_user.sub)
    return {"message": f"Camera '{source.source_id}' added", "id": source.source_id}


@router.delete(
    "/{camera_id}",
    summary="Remove a camera",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def remove_camera(
    camera_id: str,
    camera_manager: Annotated[CameraManager, Depends(get_camera_manager)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
) -> dict[str, str]:
    """Deregister and release a camera source."""
    cam = camera_manager.get_camera(camera_id)
    if cam is None:
        raise CameraUnavailableError(f"Camera '{camera_id}' not found")

    await camera_manager.remove_camera(camera_id)

    logger.info("camera.removed", camera_id=camera_id, user=current_user.sub)
    return {"message": f"Camera '{camera_id}' removed"}


@router.post(
    "/{camera_id}/start",
    summary="Start processing pipeline for a camera",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def start_camera(
    camera_id: str,
    camera_manager: Annotated[CameraManager, Depends(get_camera_manager)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    request: Request,
) -> dict[str, str]:
    """Start the frame-processing pipeline for the specified camera."""
    cam = camera_manager.get_camera(camera_id)
    if cam is None:
        raise CameraUnavailableError(f"Camera '{camera_id}' not found")

    scheduler = request.app.state.pipeline_scheduler
    scheduler.start(camera_id)

    logger.info("camera.pipeline_started", camera_id=camera_id, user=current_user.sub)
    return {"message": f"Processing started for camera '{camera_id}'"}


@router.post(
    "/{camera_id}/stop",
    summary="Stop processing pipeline for a camera",
    dependencies=[Depends(require_role(Role.OPERATOR, Role.ADMIN))],
)
async def stop_camera(
    camera_id: str,
    camera_manager: Annotated[CameraManager, Depends(get_camera_manager)],
    current_user: Annotated[TokenPayload, Depends(get_current_user)],
    request: Request,
) -> dict[str, str]:
    """Stop the frame-processing pipeline for the specified camera."""
    cam = camera_manager.get_camera(camera_id)
    if cam is None:
        raise CameraUnavailableError(f"Camera '{camera_id}' not found")

    scheduler = request.app.state.pipeline_scheduler
    await scheduler.stop(camera_id)

    logger.info("camera.pipeline_stopped", camera_id=camera_id, user=current_user.sub)
    return {"message": f"Processing stopped for camera '{camera_id}'"}


@router.get(
    "/{camera_id}/status",
    response_model=CameraStatusResponse,
    summary="Get camera health status",
    dependencies=[Depends(require_role(Role.VIEWER, Role.OPERATOR, Role.ADMIN))],
)
async def camera_status(
    camera_id: str,
    camera_manager: Annotated[CameraManager, Depends(get_camera_manager)],
    request: Request,
) -> CameraStatusResponse:
    """Return the current connectivity and processing status of a camera."""
    cam = camera_manager.get_camera(camera_id)
    if cam is None:
        raise CameraUnavailableError(f"Camera '{camera_id}' not found")

    scheduler = request.app.state.pipeline_scheduler
    is_processing = (
        camera_id in scheduler._tasks
        and not scheduler._tasks[camera_id].done()
    )

    return CameraStatusResponse(
        id=camera_id,
        is_opened=cam.is_opened,
        is_processing=is_processing,
    )
