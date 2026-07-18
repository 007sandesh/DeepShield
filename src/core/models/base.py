"""
Base model interface and registry.

All detection models inherit from BaseModel and register themselves
for automatic discovery and ensemble construction.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class ModelType(str, Enum):
    """Model category in the detection pipeline."""

    FACE_DETECTION = "face_detection"
    FORGERY_DETECTION = "forgery_detection"
    FREQUENCY = "frequency"
    TEMPORAL = "temporal"
    BIOLOGICAL = "biological"
    AUDIO_SYNC = "audio_sync"
    ATTENTION = "attention"
    ENSEMBLE = "ensemble"


@dataclass
class ModelOutput:
    """Standardized output from any detection model."""

    prediction: str  # "real" or "fake"
    confidence: float  # 0.0 - 1.0
    probabilities: dict[str, float] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    explainability: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "explainability": self.explainability,
            "metadata": self.metadata,
            "inference_time_ms": self.inference_time_ms,
        }


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all DeepShield detection models.

    Every model must:
    1. Accept standardized input (numpy array or tensor)
    2. Return ModelOutput with prediction + confidence
    3. Support explainability (heatmap/attention)
    4. Be device-aware (CPU/CUDA)
    """

    model_type: ModelType
    model_name: str

    def __init__(self, device: str = "auto", **kwargs) -> None:
        super().__init__()
        self.device = self._resolve_device(device)
        self._is_loaded = False
        self._load_time_ms: float = 0.0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @abstractmethod
    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load model weights from file or use pre-trained defaults."""
        ...

    @abstractmethod
    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Run inference on input data.

        Args:
            input_data: Input image/video frames (H, W, C) or (N, H, W, C)
            **kwargs: Model-specific arguments

        Returns:
            ModelOutput with prediction and metadata
        """
        ...

    @abstractmethod
    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """
        Generate explainability artifacts (heatmaps, attention maps, etc.).

        Args:
            input_data: Same input used for prediction

        Returns:
            Dict containing visualization data
        """
        ...

    def _measure_inference(self, func, *args, **kwargs):
        """Measure inference time of a function."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to model's device."""
        return tensor.to(self.device)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def load_time_ms(self) -> float:
        return self._load_time_ms

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.model_type.value}, "
            f"device={self.device}, "
            f"loaded={self._is_loaded})"
        )


class ModelRegistry:
    """
    Central registry for all detection models.

    Models register themselves on import, enabling:
    - Automatic discovery
    - Configuration-driven instantiation
    - Dynamic ensemble construction
    """

    _models: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, model_class: type[BaseModel]) -> type[BaseModel]:
        """Register a model class. Use as decorator."""
        name = model_class.model_name
        if name in cls._models:
            logger.warning(f"Overwriting model registration: {name}")
        cls._models[name] = model_class
        logger.debug(f"Registered model: {name} ({model_class.model_type.value})")
        return model_class

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """Get a registered model class by name."""
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(
                f"Model '{name}' not found. Available: {available}"
            )
        return cls._models[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Instantiate a registered model."""
        model_class = cls.get(name)
        return model_class(**kwargs)

    @classmethod
    def list_models(cls) -> dict[str, str]:
        """List all registered models with their types."""
        return {
            name: model.model_type.value
            for name, model in cls._models.items()
        }

    @classmethod
    def get_by_type(cls, model_type: ModelType) -> dict[str, type[BaseModel]]:
        """Get all registered models of a specific type."""
        return {
            name: model
            for name, model in cls._models.items()
            if model.model_type == model_type
        }
