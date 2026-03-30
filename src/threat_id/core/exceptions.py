"""Exception hierarchy with machine-readable error codes.

Every domain exception carries a unique `code` for API error responses
and observability. HTTP status mapping lives in the API middleware.
"""

from __future__ import annotations


class ThreatIdError(Exception):
    """Base exception for the entire application."""

    code: str = "THREAT_ID_ERROR"
    status_code: int = 500

    def __init__(self, message: str = "", *, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)


# ── Configuration ────────────────────────────────────────────────────────────

class ConfigurationError(ThreatIdError):
    code = "CONFIG_ERROR"
    status_code = 500


# ── Camera ───────────────────────────────────────────────────────────────────

class CameraError(ThreatIdError):
    code = "CAMERA_ERROR"
    status_code = 503


class CameraUnavailableError(CameraError):
    code = "CAMERA_UNAVAILABLE"


class FrameCaptureError(CameraError):
    code = "FRAME_CAPTURE_FAILED"


# ── Detection ────────────────────────────────────────────────────────────────

class DetectionError(ThreatIdError):
    code = "DETECTION_ERROR"
    status_code = 500


class ModelLoadError(DetectionError):
    code = "MODEL_LOAD_FAILED"


class InferenceError(DetectionError):
    code = "INFERENCE_FAILED"


# ── Recognition ──────────────────────────────────────────────────────────────

class RecognitionError(ThreatIdError):
    code = "RECOGNITION_ERROR"
    status_code = 500


class FaceStoreError(RecognitionError):
    code = "FACE_STORE_ERROR"


class FaceNotFoundError(RecognitionError):
    code = "FACE_NOT_FOUND"
    status_code = 404


# ── Alerting ─────────────────────────────────────────────────────────────────

class AlertError(ThreatIdError):
    code = "ALERT_ERROR"
    status_code = 502


class EmailDeliveryError(AlertError):
    code = "EMAIL_DELIVERY_FAILED"


class WebhookDeliveryError(AlertError):
    code = "WEBHOOK_DELIVERY_FAILED"


class SoundPlaybackError(AlertError):
    code = "SOUND_PLAYBACK_FAILED"


# ── Auth ─────────────────────────────────────────────────────────────────────

class AuthError(ThreatIdError):
    code = "AUTH_ERROR"
    status_code = 401


class InvalidCredentialsError(AuthError):
    code = "INVALID_CREDENTIALS"


class InsufficientPermissionsError(AuthError):
    code = "INSUFFICIENT_PERMISSIONS"
    status_code = 403


class TokenExpiredError(AuthError):
    code = "TOKEN_EXPIRED"
