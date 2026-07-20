"""
Configuration management for DeepShield.

Follows 12-Factor App principles:
- Config from environment variables
- Separate config per environment
- No secrets in code
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ModelConfig(BaseSettings):
    """ML model configuration."""

    # Model paths
    weights_dir: Path = Field(
        default=Path("models/weights"),
        description="Directory containing model weights",
    )
    checkpoints_dir: Path = Field(
        default=Path("models/checkpoints"),
        description="Directory for training checkpoints",
    )

    # Model selection
    face_detector: str = Field(
        default="retinaface",
        description="Face detection model: retinaface, mtcnn, mediapipe",
    )
    forgery_detector: str = Field(
        default="ensemble",
        description="Forgery detection backbone: ensemble, efficientnet, xception, vit",
    )
    audio_model: str = Field(
        default="whisper-base",
        description="Audio analysis model: whisper-base, whisper-small",
    )

    # Inference
    device: str = Field(
        default="auto",
        description="Compute device: auto, cpu, cuda, cuda:0",
    )
    batch_size: int = Field(default=16, ge=1, le=256)
    num_workers: int = Field(default=4, ge=0)
    use_onnx: bool = Field(default=False, description="Use ONNX runtime for inference")
    fp16: bool = Field(default=True, description="Use mixed precision inference")

    # Thresholds
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence for fake classification",
    )
    ensemble_weights: dict[str, float] = Field(
        default={
            "frequency": 0.15,
            "temporal": 0.20,
            "biological": 0.15,
            "attention": 0.25,
            "audio_sync": 0.10,
            "forgery": 0.15,
        }
    )


class PreprocessingConfig(BaseSettings):
    """Data preprocessing configuration."""

    # Face extraction
    target_face_size: tuple[int, int] = (256, 256)
    face_padding: float = 0.2
    min_face_confidence: float = 0.9

    # Frame sampling
    target_fps: int = 10
    max_frames: int = 32
    min_frames: int = 8

    # Audio
    audio_sample_rate: int = 16000
    audio_duration: float = 5.0
    n_mels: int = 80

    # Augmentation (for training only)
    augment_training: bool = True
    jpeg_quality_range: tuple[int, int] = (50, 100)
    noise_std_range: tuple[float, float] = (0.0, 0.02)


class APIConfig(BaseSettings):
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = ["http://localhost:3000"]
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    rate_limit: str = "100/minute"


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: str = "sqlite:///data/deepshield.db"
    pool_size: int = 5
    max_overflow: int = 10


class RedisConfig(BaseSettings):
    """Redis configuration for caching."""

    url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 3600
    max_connections: int = 10


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file: Optional[str] = "logs/deepshield.log"
    rotation: str = "100 MB"
    retention: str = "30 days"


class Settings(BaseSettings):
    """Root settings — aggregates all configuration modules."""

    # Environment
    env: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = False
    project_name: str = "DeepShield"
    version: str = "1.0.0"

    # Sub-configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        env_prefix = "DEEPSHIELD_"
        env_nested_delimiter = "__"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings (cached singleton)."""
    return Settings()
