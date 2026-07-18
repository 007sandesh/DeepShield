"""
Frame Sampling strategies for video analysis.

Efficiently samples frames from video to maximize detection
accuracy while minimizing computation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import cv2
import numpy as np
from loguru import logger


class SamplingStrategy(str, Enum):
    """Frame sampling strategies."""

    UNIFORM = "uniform"  # Evenly spaced frames
    ADAPTIVE = "adaptive"  # Based on motion/content
    KEYFRAME = "keyframe"  # Scene change detection
    DENSE = "dense"  # Every frame (short videos)


class FrameSampler:
    """
    Intelligent frame sampler for video analysis.

    Supports multiple strategies:
    - Uniform: Simple even spacing
    - Adaptive: Sample more during motion
    - Keyframe: Sample at scene changes
    - Dense: All frames (for short clips)
    """

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        target_fps: float = 10.0,
        max_frames: int = 32,
        min_frames: int = 8,
    ) -> None:
        self.strategy = strategy
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.min_frames = min_frames

    def sample(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> tuple[list[np.ndarray], float]:
        """
        Sample frames from video.

        Args:
            video_path: Path to video file
            max_frames: Override max frames

        Returns:
            Tuple of (frames, fps)
        """
        max_f = max_frames or self.max_frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video: {total_frames} frames @ {video_fps:.1f} FPS "
            f"({total_frames / video_fps:.1f}s)"
        )

        if self.strategy == SamplingStrategy.UNIFORM:
            frames = self._uniform_sample(cap, video_fps, total_frames, max_f)
        elif self.strategy == SamplingStrategy.ADAPTIVE:
            frames = self._adaptive_sample(cap, video_fps, total_frames, max_f)
        elif self.strategy == SamplingStrategy.KEYFRAME:
            frames = self._keyframe_sample(cap, video_fps, total_frames, max_f)
        elif self.strategy == SamplingStrategy.DENSE:
            frames = self._dense_sample(cap, video_fps, total_frames, max_f)
        else:
            frames = self._uniform_sample(cap, video_fps, total_frames, max_f)

        cap.release()

        logger.info(f"Sampled {len(frames)} frames using {self.strategy.value} strategy")

        return frames, video_fps

    def _uniform_sample(
        self, cap: cv2.VideoCapture, fps: float, total: int, max_frames: int
    ) -> list[np.ndarray]:
        """Evenly spaced frame sampling."""
        interval = max(1, int(fps / self.target_fps))
        frame_indices = list(range(0, total, interval))[:max_frames]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        return frames

    def _adaptive_sample(
        self, cap: cv2.VideoCapture, fps: float, total: int, max_frames: int
    ) -> list[np.ndarray]:
        """
        Adaptive sampling based on motion magnitude.

        Samples more frames during high-motion segments
        where deepfake artifacts are more likely to appear.
        """
        # First pass: compute motion scores
        motion_scores = []
        prev_gray = None
        frame_idx = 0
        sample_interval = max(1, total // (max_frames * 3))

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame_idx >= total:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Compute motion magnitude
                diff = cv2.absdiff(prev_gray, gray)
                motion = np.mean(diff)
                motion_scores.append((frame_idx, motion))
            else:
                motion_scores.append((frame_idx, 0.0))

            prev_gray = gray
            frame_idx += sample_interval

        if not motion_scores:
            return self._uniform_sample(cap, fps, total, max_frames)

        # Sort by motion (highest first) and select top frames
        motion_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, _ in motion_scores[:max_frames]])

        # Read selected frames
        frames = []
        for idx in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        return frames

    def _keyframe_sample(
        self, cap: cv2.VideoCapture, fps: float, total: int, max_frames: int
    ) -> list[np.ndarray]:
        """
        Sample at scene changes / keyframes.

        Detects sudden changes in content which often
        correlate with deepfake boundaries.
        """
        frames = []
        prev_hist = None
        frame_idx = 0
        interval = max(1, total // (max_frames * 5))

        while len(frames) < max_frames and frame_idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Compute color histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                # Compare histograms
                diff = cv2.compareHist(
                    prev_hist.astype(np.float32),
                    hist.astype(np.float32),
                    cv2.HISTCMP_CHISQR,
                )

                # Large change = keyframe
                if diff > 0.5 or len(frames) == 0:
                    frames.append(frame)

            prev_hist = hist
            frame_idx += interval

        return frames

    def _dense_sample(
        self, cap: cv2.VideoCapture, fps: float, total: int, max_frames: int
    ) -> list[np.ndarray]:
        """Dense sampling — every frame up to max."""
        interval = max(1, total // max_frames)
        frames = []

        for idx in range(0, total, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= max_frames:
                break

        return frames
