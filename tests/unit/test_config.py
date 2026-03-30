"""Tests for threat_id.core.config — settings validation and construction."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from threat_id.core.config import (
    DatabaseSettings,
    DetectionSettings,
    EmailSettings,
    JwtSettings,
    Settings,
)


class TestSettingsConstruction:
    """Default Settings construction produces valid sub-settings."""

    def test_default_settings_has_all_sub_settings(self) -> None:
        settings = Settings()
        assert settings.api is not None
        assert settings.jwt is not None
        assert settings.database is not None
        assert settings.detection is not None
        assert settings.recognition is not None
        assert settings.camera is not None
        assert settings.alert is not None
        assert settings.email is not None
        assert settings.webhook is not None
        assert settings.observability is not None

    def test_default_detection_confidence_threshold(self) -> None:
        settings = Settings()
        assert settings.detection.confidence_threshold == 0.5

    def test_default_api_port(self) -> None:
        settings = Settings()
        assert settings.api.port == 8000

    def test_default_jwt_algorithm(self) -> None:
        settings = Settings()
        assert settings.jwt.algorithm == "HS256"


class TestConfidenceThresholdValidation:
    """confidence_threshold must be between 0 and 1 inclusive."""

    def test_valid_threshold_zero(self) -> None:
        ds = DetectionSettings(confidence_threshold=0.0)
        assert ds.confidence_threshold == 0.0

    def test_valid_threshold_one(self) -> None:
        ds = DetectionSettings(confidence_threshold=1.0)
        assert ds.confidence_threshold == 1.0

    def test_valid_threshold_midrange(self) -> None:
        ds = DetectionSettings(confidence_threshold=0.75)
        assert ds.confidence_threshold == 0.75

    def test_invalid_threshold_negative(self) -> None:
        with pytest.raises(ValidationError, match="confidence_threshold"):
            DetectionSettings(confidence_threshold=-0.1)

    def test_invalid_threshold_above_one(self) -> None:
        with pytest.raises(ValidationError, match="confidence_threshold"):
            DetectionSettings(confidence_threshold=1.5)


class TestSecretStrFields:
    """SecretStr fields must not expose values in repr or str."""

    def test_jwt_secret_key_hidden_in_repr(self) -> None:
        jwt = JwtSettings(secret_key="super-secret")
        repr_str = repr(jwt)
        assert "super-secret" not in repr_str

    def test_jwt_secret_key_hidden_in_str(self) -> None:
        jwt = JwtSettings(secret_key="super-secret")
        str_val = str(jwt)
        assert "super-secret" not in str_val

    def test_db_password_hidden_in_repr(self) -> None:
        db = DatabaseSettings(password="db-secret-pass")
        repr_str = repr(db)
        assert "db-secret-pass" not in repr_str

    def test_email_password_hidden_in_repr(self) -> None:
        email = EmailSettings(password="email-pass")
        repr_str = repr(email)
        assert "email-pass" not in repr_str

    def test_jwt_secret_value_accessible_via_get(self) -> None:
        jwt = JwtSettings(secret_key="super-secret")
        assert jwt.secret_key.get_secret_value() == "super-secret"


class TestDatabaseSettingsAsyncUrl:
    """DatabaseSettings.async_url property builds correct connection string."""

    def test_async_url_format(self) -> None:
        db = DatabaseSettings(
            host="db-host",
            port=5433,
            name="mydb",
            user="admin",
            password="s3cret",
        )
        url = db.async_url
        assert url == "postgresql+asyncpg://admin:s3cret@db-host:5433/mydb"

    def test_sync_url_format(self) -> None:
        db = DatabaseSettings(
            host="db-host",
            port=5433,
            name="mydb",
            user="admin",
            password="s3cret",
        )
        url = db.sync_url
        assert url == "postgresql+psycopg2://admin:s3cret@db-host:5433/mydb"

    def test_async_url_uses_default_values(self) -> None:
        db = DatabaseSettings()
        url = db.async_url
        assert "localhost" in url
        assert "5432" in url
        assert "threat_id" in url
