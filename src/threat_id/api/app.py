"""FastAPI application factory.

``create_app`` builds a fully-configured :class:`FastAPI` instance with:

- Lifespan context manager (startup / shutdown orchestration)
- CORS middleware
- Correlation ID and request logging middleware
- Global exception handling for the ThreatIdError hierarchy
- All API routers mounted under ``/api/v1``
- Prometheus metrics endpoint at ``/metrics``
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from threat_id.api.middleware import (
    CorrelationIdMiddleware,
    RequestLoggingMiddleware,
    register_exception_handlers,
)
from threat_id.api.websocket_manager import ConnectionManager
from threat_id.core.config import Settings, get_settings as _default_get_settings
from threat_id.core.events import EventBus
from threat_id.core.logging import setup_logging
from threat_id.db.engine import close_db, init_db

logger = structlog.get_logger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown orchestration for all subsystems."""
    settings: Settings = app.state.settings

    # 1. Logging
    setup_logging(
        log_level=settings.api.log_level,
        json_output=not settings.api.debug,
    )
    logger.info("app.startup_begin")

    # 2. Database
    engine, session_factory = init_db(settings.database)
    app.state.db_engine = engine
    app.state.session_factory = session_factory
    logger.info("app.db_initialised")

    # 3. Event bus
    event_bus = EventBus()
    app.state.event_bus = event_bus

    # 4. Detection backend + service
    detection_service = _build_detection_service(settings)
    app.state.detection_service = detection_service
    logger.info("app.detection_service_ready", backend=settings.detection.backend)

    # 5. Recognition service
    recognition_service = _build_recognition_service(settings)
    app.state.recognition_service = recognition_service
    logger.info("app.recognition_service_ready")

    # 6. Alert service
    from threat_id.alerting.service import AlertService  # noqa: PLC0415
    from threat_id.alerting.channels.websocket import WebSocketAlertChannel  # noqa: PLC0415

    ws_alert_channel = WebSocketAlertChannel()
    channels = [ws_alert_channel]

    # Optionally add configured channels
    if "email" in settings.alert.enabled_channels and settings.email.is_configured:
        from threat_id.alerting.channels.email import EmailAlertChannel  # noqa: PLC0415

        channels.append(EmailAlertChannel(settings.email))

    if "webhook" in settings.alert.enabled_channels and settings.webhook.is_configured:
        from threat_id.alerting.channels.webhook import WebhookAlertChannel  # noqa: PLC0415

        channels.append(WebhookAlertChannel(settings.webhook))

    alert_service = AlertService(event_bus, settings.alert, channels)
    app.state.alert_service = alert_service

    # 7. Camera manager
    from threat_id.camera.manager import CameraManager  # noqa: PLC0415

    camera_manager = CameraManager(event_bus)
    app.state.camera_manager = camera_manager

    # 8. WebSocket connection manager
    ws_manager = ConnectionManager()
    app.state.ws_manager = ws_manager

    # 9. Metrics collector (lazy import — may not be installed)
    metrics_collector = _build_metrics_collector(settings)
    app.state.metrics_collector = metrics_collector

    # 10. Health checker
    health_checker = _build_health_checker(settings, engine)
    app.state.health_checker = health_checker

    # 11. Pipeline scheduler
    from threat_id.pipeline.processor import FrameProcessor  # noqa: PLC0415
    from threat_id.pipeline.scheduler import PipelineScheduler  # noqa: PLC0415

    frame_processor = FrameProcessor(
        detection_service=detection_service,
        recognition_service=recognition_service,
        event_bus=event_bus,
    )
    pipeline_scheduler = PipelineScheduler(
        processor=frame_processor,
        camera_getter=camera_manager.get_camera,
        target_fps=settings.camera.fps,
    )
    app.state.pipeline_scheduler = pipeline_scheduler

    logger.info("app.startup_complete")

    # ── Run ──
    yield

    # ── Shutdown ──
    logger.info("app.shutdown_begin")

    await pipeline_scheduler.stop_all()
    await camera_manager.stop_all()
    await close_db()

    logger.info("app.shutdown_complete")


# ── Builder helpers ─────────────────────────────────────────────────────────


def _build_detection_service(settings: Settings) -> object:
    """Instantiate the correct detection backend and wrap it in a service."""
    from threat_id.detection.service import DetectionService  # noqa: PLC0415

    if settings.detection.backend == "onnx":
        from threat_id.detection.onnx_backend import OnnxBackend  # noqa: PLC0415

        backend = OnnxBackend(settings.detection)
    else:
        from threat_id.detection.yolo_backend import YoloBackend  # noqa: PLC0415

        backend = YoloBackend(settings.detection)

    backend.load_model()
    return DetectionService(settings.detection, backend)


def _build_recognition_service(settings: Settings) -> object:
    """Instantiate the recognition backend, face store, and service."""
    from threat_id.recognition.deepface_backend import DeepFaceBackend  # noqa: PLC0415
    from threat_id.recognition.face_store import FaceStore  # noqa: PLC0415
    from threat_id.recognition.service import RecognitionService  # noqa: PLC0415

    backend = DeepFaceBackend(settings.recognition)
    face_store = FaceStore(settings.recognition.faces_dir)
    face_store.load()

    return RecognitionService(settings.recognition, backend, face_store)


def _build_metrics_collector(settings: Settings) -> object | None:
    """Build a Prometheus metrics collector if enabled."""
    if not settings.observability.prometheus_enabled:
        return None
    try:
        from threat_id.observability.metrics import MetricsCollector  # noqa: PLC0415

        return MetricsCollector()
    except ImportError:
        logger.warning("app.prometheus_not_available")
        return None


def _build_health_checker(settings: Settings, engine: object) -> object | None:
    """Build a health checker if the module is available."""
    try:
        from threat_id.observability.health import HealthChecker  # noqa: PLC0415

        return HealthChecker(engine=engine, settings=settings)
    except (ImportError, TypeError):
        logger.warning("app.health_checker_not_available")
        return None


# ── Prometheus metrics endpoint ─────────────────────────────────────────────


def _make_metrics_endpoint() -> object | None:
    """Return an ASGI app serving Prometheus metrics, or ``None``."""
    try:
        from prometheus_client import (  # noqa: PLC0415
            CONTENT_TYPE_LATEST,
            generate_latest,
        )
        from starlette.requests import Request  # noqa: PLC0415
        from starlette.responses import Response  # noqa: PLC0415
        from starlette.routing import Route  # noqa: PLC0415

        async def metrics_view(request: Request) -> Response:
            body = generate_latest()
            return Response(content=body, media_type=CONTENT_TYPE_LATEST)

        return Route("/metrics", endpoint=metrics_view)
    except ImportError:
        return None


# ── Factory ─────────────────────────────────────────────────────────────────


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a fully-configured :class:`FastAPI` application.

    Parameters
    ----------
    settings:
        Optional pre-built settings. When ``None`` the default
        :func:`get_settings` factory is used.

    Returns
    -------
    FastAPI
    """
    if settings is None:
        settings = _default_get_settings()

    app = FastAPI(
        title="Threat Identification System",
        version="1.0.0",
        description="Enterprise security surveillance API",
        docs_url="/docs" if settings.api.debug else None,
        redoc_url="/redoc" if settings.api.debug else None,
        lifespan=_lifespan,
    )

    # Store settings before lifespan runs
    app.state.settings = settings

    # ── Middleware (order matters: outermost first) ──────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # ── Exception handlers ──────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ─────────────────────────────────────────────────────
    from threat_id.api.routers.health import router as health_router  # noqa: PLC0415
    from threat_id.api.routers.auth import router as auth_router  # noqa: PLC0415
    from threat_id.api.routers.detection import router as detection_router  # noqa: PLC0415
    from threat_id.api.routers.recognition import router as recognition_router  # noqa: PLC0415
    from threat_id.api.routers.alerts import router as alerts_router  # noqa: PLC0415
    from threat_id.api.routers.camera import router as camera_router  # noqa: PLC0415
    from threat_id.api.routers.analytics import router as analytics_router  # noqa: PLC0415
    from threat_id.api.routers.admin import router as admin_router  # noqa: PLC0415

    # Health probes at root (no prefix)
    app.include_router(health_router)

    # All API routes under /api/v1
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(detection_router, prefix="/api/v1")
    app.include_router(recognition_router, prefix="/api/v1")
    app.include_router(alerts_router, prefix="/api/v1")
    app.include_router(camera_router, prefix="/api/v1")
    app.include_router(analytics_router, prefix="/api/v1")
    app.include_router(admin_router, prefix="/api/v1")

    # ── Prometheus metrics mount ────────────────────────────────────
    metrics_route = _make_metrics_endpoint()
    if metrics_route is not None:
        app.routes.append(metrics_route)

    return app
