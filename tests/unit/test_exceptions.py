"""Tests for threat_id.core.exceptions — hierarchy, codes, and status mappings."""

from __future__ import annotations

import pytest

from threat_id.core.exceptions import (
    AlertError,
    AuthError,
    CameraError,
    CameraUnavailableError,
    ConfigurationError,
    DetectionError,
    EmailDeliveryError,
    FaceNotFoundError,
    FaceStoreError,
    FrameCaptureError,
    InferenceError,
    InsufficientPermissionsError,
    InvalidCredentialsError,
    ModelLoadError,
    RecognitionError,
    SoundPlaybackError,
    ThreatIdError,
    TokenExpiredError,
    WebhookDeliveryError,
)

ALL_EXCEPTION_CLASSES = [
    ThreatIdError,
    ConfigurationError,
    CameraError,
    CameraUnavailableError,
    FrameCaptureError,
    DetectionError,
    ModelLoadError,
    InferenceError,
    RecognitionError,
    FaceStoreError,
    FaceNotFoundError,
    AlertError,
    EmailDeliveryError,
    WebhookDeliveryError,
    SoundPlaybackError,
    AuthError,
    InvalidCredentialsError,
    InsufficientPermissionsError,
    TokenExpiredError,
]


class TestExceptionHierarchy:
    """Every domain exception must subclass ThreatIdError."""

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTION_CLASSES)
    def test_all_subclass_threat_id_error(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, ThreatIdError)

    def test_camera_errors_subclass_camera_error(self) -> None:
        assert issubclass(CameraUnavailableError, CameraError)
        assert issubclass(FrameCaptureError, CameraError)

    def test_detection_errors_subclass_detection_error(self) -> None:
        assert issubclass(ModelLoadError, DetectionError)
        assert issubclass(InferenceError, DetectionError)

    def test_recognition_errors_subclass_recognition_error(self) -> None:
        assert issubclass(FaceStoreError, RecognitionError)
        assert issubclass(FaceNotFoundError, RecognitionError)

    def test_alert_errors_subclass_alert_error(self) -> None:
        assert issubclass(EmailDeliveryError, AlertError)
        assert issubclass(WebhookDeliveryError, AlertError)
        assert issubclass(SoundPlaybackError, AlertError)

    def test_auth_errors_subclass_auth_error(self) -> None:
        assert issubclass(InvalidCredentialsError, AuthError)
        assert issubclass(InsufficientPermissionsError, AuthError)
        assert issubclass(TokenExpiredError, AuthError)


class TestErrorCodes:
    """Each exception must have a unique error code string."""

    def test_all_codes_are_unique(self) -> None:
        codes = [cls.code for cls in ALL_EXCEPTION_CLASSES]
        assert len(codes) == len(set(codes)), f"Duplicate codes found: {codes}"

    def test_base_error_code(self) -> None:
        assert ThreatIdError.code == "THREAT_ID_ERROR"

    def test_invalid_credentials_code(self) -> None:
        assert InvalidCredentialsError.code == "INVALID_CREDENTIALS"

    def test_token_expired_code(self) -> None:
        assert TokenExpiredError.code == "TOKEN_EXPIRED"

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTION_CLASSES)
    def test_code_is_non_empty_string(self, exc_cls: type) -> None:
        assert isinstance(exc_cls.code, str)
        assert len(exc_cls.code) > 0


class TestStatusCodeMappings:
    """Verify HTTP status codes for key exception families."""

    def test_base_error_is_500(self) -> None:
        assert ThreatIdError.status_code == 500

    def test_configuration_error_is_500(self) -> None:
        assert ConfigurationError.status_code == 500

    def test_camera_error_is_503(self) -> None:
        assert CameraError.status_code == 503

    def test_camera_unavailable_inherits_503(self) -> None:
        # Subclass inherits parent status_code
        assert CameraUnavailableError.status_code == 503

    def test_detection_error_is_500(self) -> None:
        assert DetectionError.status_code == 500

    def test_alert_error_is_502(self) -> None:
        assert AlertError.status_code == 502

    def test_auth_error_is_401(self) -> None:
        assert AuthError.status_code == 401

    def test_insufficient_permissions_is_403(self) -> None:
        assert InsufficientPermissionsError.status_code == 403

    def test_face_not_found_is_404(self) -> None:
        assert FaceNotFoundError.status_code == 404


class TestExceptionInstantiation:
    """Exceptions carry message and detail fields."""

    def test_message_is_stored(self) -> None:
        exc = ThreatIdError("something went wrong")
        assert exc.message == "something went wrong"
        assert str(exc) == "something went wrong"

    def test_detail_is_stored(self) -> None:
        exc = ThreatIdError("msg", detail="extra info")
        assert exc.detail == "extra info"

    def test_detail_defaults_to_none(self) -> None:
        exc = ThreatIdError("msg")
        assert exc.detail is None
