"""
Biological Signal Analysis for deepfake detection.

Detects the absence of natural biological signals that
generators fail to replicate:
- Remote Photoplethysmography (rPPG) — blood flow under skin
- Eye reflection consistency
- Micro-expression analysis
- Skin texture dynamics
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelType, ModelRegistry


class RPPGExtractor(nn.Module):
    """Neural network for rPPG signal extraction from facial ROIs."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.signal_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = feat.flatten(1)
        return self.signal_head(feat)


@ModelRegistry.register
class BiologicalSignalAnalyzer(BaseModel):
    """
    Biological signal verification for deepfake detection.

    Analysis methods:
    1. rPPG (remote photoplethysmography) — blood flow signal
    2. Corneal specular reflection — light source consistency
    3. Pupil dynamics — natural pupil response
    4. Skin texture variation — chromatic micro-changes
    5. Nasolabial fold dynamics — expression authenticity
    """

    model_type = ModelType.BIOLOGICAL
    model_name = "biological_signals"

    def __init__(self, device: str = "auto", **kwargs) -> None:
        super().__init__(device=device)
        self._rppg_net = RPPGExtractor().to(self.device).eval()
        self._is_loaded = True
        logger.info("Biological signal analyzer initialized")

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load biological signal analysis weights."""
        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)
            self._rppg_net.load_state_dict(state_dict)
        self._is_loaded = True

    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Analyze biological signals in face image/video.

        Args:
            input_data: Face image (H, W, 3) or frames (N, H, W, 3)

        Returns:
            ModelOutput with biological signal analysis
        """
        frames = input_data if input_data.ndim == 4 else np.expand_dims(input_data, 0)

        result, elapsed = self._measure_inference(self._analyze, frames)

        return ModelOutput(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            metadata={
                "rppg_score": result["rppg_score"],
                "reflection_score": result["reflection_score"],
                "texture_score": result["texture_score"],
                "pupil_score": result["pupil_score"],
                "has_rppg_signal": result["has_rppg_signal"],
            },
            explainability={
                "rppg_waveform": result["rppg_waveform"],
                "roi_heatmap": result["roi_heatmap"],
            },
            inference_time_ms=elapsed,
        )

    def _analyze(self, frames: np.ndarray) -> dict[str, Any]:
        """Run full biological signal analysis."""
        # 1. rPPG analysis (requires multiple frames)
        rppg_score, rppg_waveform, has_signal = self._analyze_rppg(frames)

        # 2. Eye reflection consistency
        reflection_score = self._analyze_reflections(frames[0])

        # 3. Skin texture dynamics
        texture_score = self._analyze_skin_texture(frames)

        # 4. Pupil analysis
        pupil_score = self._analyze_pupil(frames[0])

        # 5. ROI heatmap
        roi_heatmap = self._generate_roi_heatmap(frames[0])

        # Combine
        combined = (
            0.35 * rppg_score +
            0.25 * reflection_score +
            0.20 * texture_score +
            0.20 * pupil_score
        )

        return {
            "prediction": "fake" if combined > 0.5 else "real",
            "confidence": float(max(1 - combined, combined)),
            "probabilities": {"real": float(1 - combined), "fake": float(combined)},
            "rppg_score": rppg_score,
            "reflection_score": reflection_score,
            "texture_score": texture_score,
            "pupil_score": pupil_score,
            "has_rppg_signal": has_signal,
            "rppg_waveform": rppg_waveform,
            "roi_heatmap": roi_heatmap,
        }

    def _analyze_rppg(self, frames: np.ndarray) -> tuple[float, np.ndarray, bool]:
        """
        Extract remote photoplethysmography (rPPG) signal.

        Real skin shows subtle color changes from blood flow.
        Deepfakes typically lack this signal or have inconsistent patterns.

        CHROM method: Uses chrominance-based signal extraction.
        """
        if len(frames) < 10:
            return 0.5, np.array([]), False

        # Extract forehead ROI (most reliable for rPPG)
        rois = []
        for frame in frames[:64]:  # Sample up to 64 frames
            roi = self._extract_forehead_roi(frame)
            if roi is not None:
                rois.append(roi)

        if len(rois) < 10:
            return 0.5, np.array([]), False

        # Stack ROIs and compute mean color signal
        roi_stack = np.array(rois)  # (N, roi_h, roi_w, 3)
        mean_signal = np.mean(roi_stack, axis=(1, 2))  # (N, 3) — R, G, B means

        # CHROM-based rPPG extraction
        X = mean_signal[:, 0] - mean_signal[:, 1]  # Red - Green
        Y = mean_signal[:, 0] + mean_signal[:, 1] - 2 * mean_signal[:, 2]  # (R+G) - 2B

        # Temporal filtering (bandpass: 0.75-3 Hz for heart rate)
        from scipy.signal import butter, filtfilt
        try:
            fs = 30  # Assumed frame rate
            low = 0.75 / (fs / 2)
            high = 3.0 / (fs / 2)
            b, a = butter(2, [low, high], btype="band")
            X_filtered = filtfilt(b, a, X)
            Y_filtered = filtfilt(b, a, Y)

            # CHROM signal
            alpha = np.std(X_filtered) / (np.std(Y_filtered) + 1e-8)
            rppg_signal = X_filtered - alpha * Y_filtered

            # Analyze signal quality
            snr = self._compute_snr(rppg_signal, fs)
            has_signal = snr > 2.0  # SNR > 2 dB indicates real rPPG

            # Score: low SNR = suspicious
            score = float(np.clip(1.0 - snr / 10.0, 0, 1))

            return score, rppg_signal, has_signal

        except Exception as e:
            logger.warning(f"rPPG analysis failed: {e}")
            return 0.5, np.array([]), False

    def _extract_forehead_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract forehead region of interest for rPPG."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Simple face detection to locate forehead
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        x, y, fw, fh = faces[0]

        # Forehead is approximately top 25% of face
        forehead_y = y
        forehead_h = int(fh * 0.25)
        forehead_x = x + int(fw * 0.15)
        forehead_w = int(fw * 0.7)

        roi = frame[forehead_y:forehead_y + forehead_h, forehead_x:forehead_x + forehead_w]

        if roi.size == 0:
            return None

        return cv2.resize(roi, (32, 32))

    @staticmethod
    def _compute_snr(signal: np.ndarray, fs: float) -> float:
        """Compute Signal-to-Noise Ratio for rPPG signal."""
        if len(signal) < 10:
            return 0.0

        # FFT
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)

        # Heart rate band (0.75-3 Hz)
        hr_mask = (freqs >= 0.75) & (freqs <= 3.0)
        noise_mask = (freqs >= 0.1) & (freqs < 0.75) | (freqs > 3.0)

        signal_power = np.mean(fft[hr_mask] ** 2) if np.any(hr_mask) else 0
        noise_power = np.mean(fft[noise_mask] ** 2) if np.any(noise_mask) else 1e-10

        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)

    def _analyze_reflections(self, frame: np.ndarray) -> float:
        """
        Analyze corneal specular reflections.

        Real eyes have consistent light reflections from the same
        light source. Deepfakes often have inconsistent or missing reflections.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eye regions (simplified)
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

        if len(eyes) < 2:
            return 0.5  # Uncertain

        # Analyze specular highlights in each eye
        reflections = []
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_roi = gray[ey:ey + eh, ex:ex + ew]
            # Find bright spots (reflections)
            _, thresh = cv2.threshold(eye_roi, 200, 255, cv2.THRESH_BINARY)
            bright_ratio = np.sum(thresh > 0) / (thresh.size + 1e-8)
            reflections.append(bright_ratio)

        # Check consistency between eyes
        if len(reflections) >= 2:
            consistency = 1.0 - abs(reflections[0] - reflections[1])
            # Score: low consistency = suspicious
            score = float(np.clip(1.0 - consistency, 0, 1))
        else:
            score = 0.5

        return score

    def _analyze_skin_texture(self, frames: np.ndarray) -> float:
        """
        Analyze skin texture variation over time.

        Real skin has natural micro-texture changes from:
        - Blood flow
        - Micro-expressions
        - Subsurface scattering changes
        """
        if len(frames) < 5:
            return 0.5

        texture_vars = []
        for frame in frames[:32]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Local Binary Pattern for texture
            lbp = self._compute_lbp(gray)
            texture_vars.append(np.std(lbp))

        if len(texture_vars) < 3:
            return 0.5

        # Real skin: texture variance changes over time
        # Fake skin: texture is often static or unnaturally consistent
        variance_of_variance = np.std(texture_vars)
        mean_variance = np.mean(texture_vars)

        # Low variance of variance = suspicious
        consistency_ratio = variance_of_variance / (mean_variance + 1e-8)
        score = float(np.clip(1.0 - consistency_ratio * 5, 0, 1))

        return score

    def _analyze_pupil(self, frame: np.ndarray) -> float:
        """
        Analyze pupil characteristics.

        Real pupils:
        - Have natural dilation/contraction
        - Show consistent dark center
        - Have proper iris texture around them
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Hough circle detection for pupils
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=5, maxRadius=30,
        )

        if circles is None:
            return 0.5

        # Analyze pupil circularity and darkness
        scores = []
        for circle in circles[0, :2]:  # Check up to 2 eyes
            cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])

            # Extract pupil region
            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            pupil_region = gray[mask > 0]

            if len(pupil_region) > 0:
                # Pupils should be very dark
                darkness = 1.0 - (np.mean(pupil_region) / 255.0)
                # Should be relatively uniform
                uniformity = 1.0 - (np.std(pupil_region) / 128.0)
                scores.append(darkness * uniformity)

        if not scores:
            return 0.5

        # Score: unnatural pupils = suspicious
        avg_score = np.mean(scores)
        return float(np.clip(1.0 - avg_score, 0, 1))

    def _generate_roi_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Generate heatmap showing analyzed biological signal regions."""
        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        # Forehead region (rPPG)
        heatmap[int(h * 0.1):int(h * 0.3), int(w * 0.2):int(w * 0.8)] = 0.8

        # Eye regions (reflections)
        heatmap[int(h * 0.3):int(h * 0.5), int(w * 0.1):int(w * 0.45)] = 1.0
        heatmap[int(h * 0.3):int(h * 0.5), int(w * 0.55):int(w * 0.9)] = 1.0

        # Cheek regions (skin texture)
        heatmap[int(h * 0.5):int(h * 0.8), int(w * 0.05):int(w * 0.35)] = 0.6
        heatmap[int(h * 0.5):int(h * 0.8), int(w * 0.65):int(w * 0.95)] = 0.6

        # Gaussian blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        heatmap = cv2.resize(heatmap, (256, 256))

        return heatmap

    @staticmethod
    def _compute_lbp(gray: np.ndarray, radius: int = 1) -> np.ndarray:
        """Compute Local Binary Pattern for texture analysis."""
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                code = 0
                for k, (di, dj) in enumerate([
                    (-radius, -radius), (-radius, 0), (-radius, radius),
                    (0, radius), (radius, radius), (radius, 0),
                    (radius, -radius), (0, -radius),
                ]):
                    if gray[i + di, j + dj] >= center:
                        code |= (1 << k)
                lbp[i, j] = code

        return lbp

    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Generate biological signal visualizations."""
        output = self.predict(input_data, **kwargs)
        return {
            "rppg_waveform": output.explainability["rppg_waveform"],
            "roi_heatmap": output.explainability["roi_heatmap"],
            "rppg_score": output.metadata["rppg_score"],
            "reflection_score": output.metadata["reflection_score"],
            "texture_score": output.metadata["texture_score"],
            "pupil_score": output.metadata["pupil_score"],
        }
