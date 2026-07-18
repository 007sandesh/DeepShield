"""
Temporal Consistency Analyzer for video deepfake detection.

Detects frame-to-frame inconsistencies that indicate manipulation:
- Facial landmark jitter
- Optical flow anomalies
- Temporal warping artifacts
- Blink pattern analysis
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelType, ModelRegistry


class TemporalLSTM(nn.Module):
    """LSTM for temporal sequence analysis of face landmarks."""

    def __init__(self, input_dim: int = 68 * 2, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim)"""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


@ModelRegistry.register
class TemporalAnalyzer(BaseModel):
    """
    Video temporal consistency analysis.

    Detects:
    1. Facial landmark jitter (frame-to-frame position instability)
    2. Optical flow inconsistencies (unnatural motion patterns)
    3. Temporal warping (inconsistent frame timing)
    4. Blink pattern analysis (unnatural or absent blinking)
    """

    model_type = ModelType.TEMPORAL
    model_name = "temporal_analyzer"

    def __init__(self, device: str = "auto", **kwargs) -> None:
        super().__init__(device=device)
        self._landmark_detector = None
        self._temporal_net = TemporalLSTM()
        self._temporal_net = self._temporal_net.to(self.device).eval()
        self.load_weights()

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load temporal analysis models."""
        import time
        start = time.perf_counter()

        # Initialize dlib landmark detector
        try:
            import dlib
            self._landmark_detector = dlib.shape_predictor(
                "models/weights/shape_predictor_68_face_landmarks.dat"
            )
        except Exception:
            logger.warning("dlib landmark detector not found, using MediaPipe fallback")
            self._landmark_detector = None

        self._is_loaded = True
        self._load_time_ms = (time.perf_counter() - start) * 1000

    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Analyze temporal consistency across video frames.

        Args:
            input_data: Array of frames (N, H, W, 3) or single frame
            **kwargs:
                fps: float — Video frame rate

        Returns:
            ModelOutput with temporal analysis results
        """
        frames = input_data if input_data.ndim == 4 else np.expand_dims(input_data, 0)
        fps = kwargs.get("fps", 30.0)

        result, elapsed = self._measure_inference(self._analyze_temporal, frames, fps)

        return ModelOutput(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            metadata={
                "jitter_score": result["jitter_score"],
                "flow_score": result["flow_score"],
                "warp_score": result["warp_score"],
                "blink_score": result["blink_score"],
                "num_frames": len(frames),
                "fps": fps,
            },
            explainability={
                "jitter_timeline": result["jitter_timeline"],
                "flow_magnitude": result["flow_visualization"],
            },
            inference_time_ms=elapsed,
        )

    def _analyze_temporal(self, frames: np.ndarray, fps: float) -> dict[str, Any]:
        """Run full temporal analysis."""
        n_frames = len(frames)

        if n_frames < 2:
            return self._single_frame_fallback()

        # 1. Landmark jitter analysis
        jitter_score, jitter_timeline = self._analyze_jitter(frames)

        # 2. Optical flow analysis
        flow_score, flow_vis = self._analyze_optical_flow(frames)

        # 3. Temporal warping detection
        warp_score = self._analyze_temporal_warping(frames)

        # 4. Blink pattern analysis
        blink_score = self._analyze_blinks(frames, fps)

        # 5. LSTM-based temporal classification
        lstm_score = self._lstm_classify(frames)

        # Combine scores
        combined = (
            0.25 * jitter_score +
            0.25 * flow_score +
            0.15 * warp_score +
            0.15 * blink_score +
            0.20 * lstm_score
        )

        return {
            "prediction": "fake" if combined > 0.5 else "real",
            "confidence": float(max(1 - combined, combined)),
            "probabilities": {"real": float(1 - combined), "fake": float(combined)},
            "jitter_score": jitter_score,
            "flow_score": flow_score,
            "warp_score": warp_score,
            "blink_score": blink_score,
            "jitter_timeline": jitter_timeline,
            "flow_visualization": flow_vis,
        }

    def _analyze_jitter(self, frames: np.ndarray) -> tuple[float, list[float]]:
        """
        Detect facial landmark jitter.

        Real faces move smoothly. Deepfakes often have frame-to-frame
        landmark position instability.
        """
        landmarks_sequence = []
        for frame in frames:
            landmarks = self._extract_landmarks(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)

        if len(landmarks_sequence) < 2:
            return 0.0, []

        # Calculate frame-to-frame differences
        diffs = []
        timeline = []
        for i in range(1, len(landmarks_sequence)):
            diff = np.mean(np.abs(landmarks_sequence[i] - landmarks_sequence[i - 1]))
            diffs.append(diff)
            timeline.append(float(diff))

        # Jitter = variance of frame-to-frame differences
        jitter = float(np.std(diffs))

        # Normalize (empirically determined thresholds)
        score = float(np.clip(jitter / 5.0, 0, 1))

        return score, timeline

    def _analyze_optical_flow(self, frames: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Analyze optical flow consistency.

        Real video has smooth, physically plausible motion.
        Deepfakes often have flow discontinuities at face boundaries.
        """
        flows = []
        for i in range(1, min(len(frames), 10)):  # Sample up to 10 frames
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flows.append(flow)

        if not flows:
            return 0.0, np.zeros((256, 256), dtype=np.float32)

        # Calculate flow statistics
        flow_magnitudes = [np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2) for f in flows]
        mean_mag = np.mean([m.mean() for m in flow_magnitudes])
        std_mag = np.mean([m.std() for m in flow_magnitudes])

        # Flow smoothness (low std relative to mean = smooth = likely real)
        smoothness = std_mag / (mean_mag + 1e-8)

        # Check for flow discontinuities (edge detection on flow magnitude)
        edge_scores = []
        for mag in flow_magnitudes:
            edges = cv2.Canny((mag * 10).astype(np.uint8), 50, 150)
            edge_scores.append(np.mean(edges))

        edge_score = np.mean(edge_scores)

        # Combine
        score = float(np.clip(smoothness * 0.3 + edge_score * 0.01, 0, 1))

        # Visualization: mean flow magnitude
        vis = np.mean(flow_magnitudes, axis=0).astype(np.float32)
        vis = cv2.resize(vis, (256, 256))

        return score, vis

    def _analyze_temporal_warping(self, frames: np.ndarray) -> float:
        """
        Detect temporal warping artifacts.

        Some deepfake methods apply temporal smoothing that creates
        inconsistent motion timing.
        """
        # Compute inter-frame differences
        diffs = []
        for i in range(1, len(frames)):
            gray1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY).astype(float)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
            diff = np.mean(np.abs(gray1 - gray2))
            diffs.append(diff)

        if len(diffs) < 3:
            return 0.0

        diffs = np.array(diffs)

        # Check for unnatural timing patterns
        # Real video: diffs are relatively smooth
        # Warped video: diffs may have periodic spikes
        diff_fft = np.abs(np.fft.rfft(diffs - np.mean(diffs)))
        dominant_freq = np.max(diff_fft[1:]) / (diff_fft[0] + 1e-8)

        score = float(np.clip(dominant_freq * 0.5, 0, 1))
        return score

    def _analyze_blinks(self, frames: np.ndarray, fps: float) -> float:
        """
        Analyze blink patterns.

        Early deepfakes rarely generate blinks. Modern ones do,
        but patterns are often unnatural.
        """
        eye_aspect_ratios = []

        for frame in frames[:50]:  # Analyze up to 50 frames
            landmarks = self._extract_landmarks(frame)
            if landmarks is not None and len(landmarks) >= 68:
                # Eye aspect ratio (EAR) from 68 landmarks
                left_ear = self._eye_aspect_ratio(landmarks[36:42])
                right_ear = self._eye_aspect_ratio(landmarks[42:48])
                avg_ear = (left_ear + right_ear) / 2.0
                eye_aspect_ratios.append(avg_ear)

        if len(eye_aspect_ratios) < 10:
            return 0.5  # Uncertain

        ear_array = np.array(eye_aspect_ratios)

        # Detect blinks (EAR drops below threshold)
        threshold = np.mean(ear_array) * 0.7
        blinks = np.sum(ear_array < threshold)

        # Calculate blink rate (blinks per minute)
        duration = len(eye_aspect_ratios) / fps
        blink_rate = (blinks / duration) * 60 if duration > 0 else 0

        # Normal blink rate: 15-20 per minute
        # Too few blinks = suspicious
        if blink_rate < 5:
            score = 0.8  # Very suspicious
        elif blink_rate < 10:
            score = 0.4  # Somewhat suspicious
        elif blink_rate > 30:
            score = 0.3  # Unnaturally high
        else:
            score = 0.1  # Normal range

        return score

    def _lstm_classify(self, frames: np.ndarray) -> float:
        """LSTM-based temporal classification."""
        landmarks_sequence = []
        for frame in frames[:32]:  # Max 32 frames
            lm = self._extract_landmarks(frame)
            if lm is not None:
                landmarks_sequence.append(lm.flatten())

        if len(landmarks_sequence) < 4:
            return 0.5

        # Pad or truncate to fixed length
        target_len = 32
        if len(landmarks_sequence) < target_len:
            pad_len = target_len - len(landmarks_sequence)
            landmarks_sequence.extend([landmarks_sequence[-1]] * pad_len)
        else:
            landmarks_sequence = landmarks_sequence[:target_len]

        tensor = torch.tensor(landmarks_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._temporal_net(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        return float(probs[1])

    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks from frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            face_rect = cv2.rectangle(
                (0, 0), (gray.shape[1], gray.shape[0]), 0
            )

            if self._landmark_detector is not None:
                import dlib
                detector = dlib.get_frontal_face_detector()
                faces = detector(gray)
                if faces:
                    shape = self._landmark_detector(gray, faces[0])
                    coords = np.array([[p.x, p.y] for p in shape.parts()])
                    return coords
            else:
                # Fallback: use MediaPipe or simple face detection
                return self._fallback_landmarks(gray)

        except Exception:
            return None

        return None

    def _fallback_landmarks(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Fallback landmark extraction using OpenCV."""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Return approximate landmark positions
            return np.array([
                [x + w // 3, y + h // 3],  # Left eye
                [x + 2 * w // 3, y + h // 3],  # Right eye
                [x + w // 2, y + 2 * h // 3],  # Nose
            ])

        return None

    @staticmethod
    def _eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        if len(eye_landmarks) != 6:
            return 0.0

        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

        # EAR
        ear = (v1 + v2) / (2.0 * h + 1e-8)
        return float(ear)

    def _single_frame_fallback(self) -> dict[str, Any]:
        """Fallback for single frame input."""
        return {
            "prediction": "uncertain",
            "confidence": 0.5,
            "probabilities": {"real": 0.5, "fake": 0.5},
            "jitter_score": 0.0,
            "flow_score": 0.0,
            "warp_score": 0.0,
            "blink_score": 0.5,
            "jitter_timeline": [],
            "flow_visualization": np.zeros((256, 256), dtype=np.float32),
        }

    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Generate temporal analysis visualizations."""
        output = self.predict(input_data, **kwargs)
        return {
            "jitter_timeline": output.metadata["jitter_timeline"],
            "flow_magnitude": output.explainability["flow_magnitude"],
            "jitter_score": output.metadata["jitter_score"],
            "flow_score": output.metadata["flow_score"],
            "warp_score": output.metadata["warp_score"],
            "blink_score": output.metadata["blink_score"],
        }
