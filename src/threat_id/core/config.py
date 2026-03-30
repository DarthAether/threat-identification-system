"""Centralised configuration via pydantic-settings.

Every setting comes from environment variables or a .env file.
No hardcoded values exist anywhere else in the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]
    debug: bool = False
    log_level: str = "info"


class JwtSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JWT_")

    secret_key: SecretStr = SecretStr("CHANGE-ME-IN-PRODUCTION")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "threat_id"
    user: str = "postgres"
    password: SecretStr = SecretStr("changeme")

    @property
    def async_url(self) -> str:
        pw = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pw}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        pw = self.password.get_secret_value()
        return f"postgresql+psycopg2://{self.user}:{pw}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = "redis://localhost:6379/0"


class DetectionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DETECTION_")

    backend: Literal["yolo", "onnx"] = "yolo"
    model_path: str = "yolov5s.pt"
    confidence_threshold: float = 0.5
    device: str = "cpu"
    threat_categories: list[str] = ["knife", "gun", "rifle", "pistol"]

    @field_validator("confidence_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            msg = "confidence_threshold must be between 0 and 1"
            raise ValueError(msg)
        return v


class RecognitionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECOGNITION_")

    model_name: str = "Facenet"
    distance_metric: str = "cosine"
    similarity_threshold: float = 0.6
    faces_dir: Path = Path("./data/faces")


class CameraSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAMERA_")

    default_source: str = "0"
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30


class AlertSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALERT_")

    cooldown_seconds: int = 30
    enabled_channels: list[str] = ["websocket"]


class EmailSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMTP_")

    host: str = "smtp.gmail.com"
    port: int = 465
    username: str = ""
    password: SecretStr = SecretStr("")
    sender: str = ""
    recipients: list[str] = []

    @property
    def is_configured(self) -> bool:
        return bool(self.username and self.sender and self.recipients)


class WebhookSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WEBHOOK_")

    url: str = ""
    timeout_seconds: int = 10

    @property
    def is_configured(self) -> bool:
        return bool(self.url)


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    prometheus_enabled: bool = True
    audit_log_enabled: bool = True
    audit_retention_days: int = 90


class Settings(BaseSettings):
    """Root settings — aggregates all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api: ApiSettings = ApiSettings()
    jwt: JwtSettings = JwtSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    detection: DetectionSettings = DetectionSettings()
    recognition: RecognitionSettings = RecognitionSettings()
    camera: CameraSettings = CameraSettings()
    alert: AlertSettings = AlertSettings()
    email: EmailSettings = EmailSettings()
    webhook: WebhookSettings = WebhookSettings()
    observability: ObservabilitySettings = ObservabilitySettings()


def get_settings() -> Settings:
    """Factory function — allows dependency injection override in tests."""
    return Settings()
