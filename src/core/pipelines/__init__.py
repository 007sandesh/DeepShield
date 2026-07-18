"""
Detection Pipelines for DeepShield.
"""

from src.core.pipelines.image_pipeline import ImageDetectionPipeline
from src.core.pipelines.video_pipeline import VideoDetectionPipeline

__all__ = ["ImageDetectionPipeline", "VideoDetectionPipeline"]
