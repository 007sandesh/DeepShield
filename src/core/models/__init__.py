"""
Core detection models for DeepShield.

Each model represents a detection layer in the ensemble.
"""

from src.core.models.base import BaseModel, ModelRegistry
from src.core.models.face_detector import FaceDetector
from src.core.models.forgery_detector import ForgeryDetector
from src.core.models.ai_image_detector import AIImageDetector
from src.core.models.frequency_analyzer import FrequencyAnalyzer
from src.core.models.temporal_analyzer import TemporalAnalyzer
from src.core.models.biological_signals import BiologicalSignalAnalyzer
from src.core.models.audio_sync import AudioSyncAnalyzer
from src.core.models.attention_network import AttentionNetwork
from src.core.models.ensemble import EnsembleClassifier

__all__ = [
    "BaseModel",
    "ModelRegistry",
    "FaceDetector",
    "ForgeryDetector",
    "AIImageDetector",
    "FrequencyAnalyzer",
    "TemporalAnalyzer",
    "BiologicalSignalAnalyzer",
    "AudioSyncAnalyzer",
    "AttentionNetwork",
    "EnsembleClassifier",
]
