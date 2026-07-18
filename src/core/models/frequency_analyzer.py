"""
Frequency Domain Analysis for GAN artifact detection.

Detects spectral fingerprints left by GAN generators through
DCT, FFT, and wavelet analysis.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelType, ModelRegistry


class FrequencyFeatureExtractor(nn.Module):
    """CNN that learns to classify frequency domain patterns."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


@ModelRegistry.register
class FrequencyAnalyzer(BaseModel):
    """
    Frequency domain deepfake detection.

    Analysis methods:
    1. FFT magnitude spectrum — periodic GAN artifacts
    2. DCT coefficient analysis — block-level anomalies
    3. Band-pass filtering — spectral energy distribution
    4. High-frequency noise pattern — upsampling artifacts
    """

    model_type = ModelType.FREQUENCY
    model_name = "frequency_analyzer"

    def __init__(self, device: str = "auto", **kwargs) -> None:
        super().__init__(device=device)
        self._net = FrequencyFeatureExtractor()
        self._net = self._net.to(self.device).eval()
        self._is_loaded = True
        logger.info("Frequency analyzer initialized")

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load frequency analysis model weights."""
        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)
            self._net.load_state_dict(state_dict)
        self._is_loaded = True

    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Analyze frequency domain patterns.

        Args:
            input_data: BGR face image (H, W, 3)

        Returns:
            ModelOutput with frequency analysis results
        """
        result, elapsed = self._measure_inference(self._analyze, input_data)

        return ModelOutput(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            metadata={
                "fft_score": result["fft_score"],
                "dct_score": result["dct_score"],
                "band_energy": result["band_energy"],
                "noise_pattern": result["noise_pattern"],
                "smoothness_score": result.get("smoothness_score", 0.0),
            },
            explainability={
                "fft_magnitude": result["fft_visualization"],
                "dct_coefficients": result["dct_visualization"],
            },
            inference_time_ms=elapsed,
        )

    def _analyze(self, image: np.ndarray) -> dict[str, Any]:
        """Run full frequency analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (256, 256))

        # 1. FFT Analysis
        fft_score, fft_vis = self._analyze_fft(resized)

        # 2. DCT Analysis
        dct_score, dct_vis = self._analyze_dct(resized)

        # 3. Band energy distribution
        band_energy = self._analyze_bands(resized)

        # 4. High-frequency noise pattern
        noise_pattern = self._analyze_noise_pattern(resized)

        # 5. CNN-based frequency classification
        tensor = self._to_tensor(image)
        with torch.no_grad():
            logits = self._net(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Combine scores — prefer classical spectral cues over untrained CNN head
        smoothness_score = self._analyze_texture_smoothness(resized)
        combined_score = (
            0.25 * fft_score
            + 0.20 * dct_score
            + 0.20 * noise_pattern
            + 0.25 * smoothness_score
            + 0.10 * float(probs[1])
        )

        return {
            "prediction": "fake" if combined_score > 0.5 else "real",
            "confidence": float(max(1 - combined_score, combined_score)),
            "probabilities": {"real": float(1 - combined_score), "fake": float(combined_score)},
            "fft_score": fft_score,
            "dct_score": dct_score,
            "band_energy": band_energy,
            "noise_pattern": noise_pattern,
            "smoothness_score": smoothness_score,
            "fft_visualization": fft_vis,
            "dct_visualization": dct_vis,
        }

    def _analyze_fft(self, gray: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Analyze FFT magnitude spectrum for GAN artifacts.

        GANs leave periodic patterns in frequency domain due to
        upsampling operations (checkerboard artifacts).
        """
        # Compute 2D FFT
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

        # Look for periodic spikes (GAN fingerprint)
        # Analyze radial profile
        center = np.array(magnitude.shape) // 2
        radial_profile = self._radial_profile(magnitude, center)

        # Detect periodicity using autocorrelation
        if len(radial_profile) > 10:
            autocorr = np.correlate(radial_profile, radial_profile, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            # Look for peaks in autocorrelation (periodicity indicator)
            peaks = self._find_peaks(autocorr[1:])  # Skip zero-lag
            periodicity_score = len(peaks) / max(len(autocorr) // 4, 1)
            score = float(np.clip(periodicity_score, 0, 1))
        else:
            score = 0.0

        return score, magnitude

    def _analyze_dct(self, gray: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Analyze DCT coefficients for block-level anomalies.

        GAN-generated images often have unnatural DCT coefficient
        distributions at block boundaries (8x8 or 16x16).
        """
        # Block-wise DCT
        block_size = 8
        h, w = gray.shape
        dct_blocks = []

        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray[i:i + block_size, j:j + block_size].astype(np.float32)
                dct_block = cv2.dct(block)
                dct_blocks.append(dct_block)

        if not dct_blocks:
            return 0.0, np.zeros_like(gray, dtype=np.float32)

        dct_array = np.array(dct_blocks)

        # Analyze high-frequency coefficient distribution
        # Real images: coefficients decay smoothly
        # Fake images: coefficients may have unnatural patterns
        mean_coeffs = np.mean(dct_array, axis=0)

        # Check ratio of high-freq to low-freq energy
        low_freq = np.sum(np.abs(mean_coeffs[:4, :4]))
        high_freq = np.sum(np.abs(mean_coeffs[4:, 4:]))
        ratio = high_freq / (low_freq + 1e-8)

        # GAN artifacts often have elevated high-freq energy
        score = float(np.clip(ratio * 2, 0, 1))

        return score, mean_coeffs

    def _analyze_bands(self, gray: np.ndarray) -> dict[str, float]:
        """Analyze energy distribution across frequency bands."""
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Define concentric bands
        bands = {}
        for radius, label in [(32, "low"), (64, "mid_low"), (128, "mid_high")]:
            mask = np.zeros_like(magnitude)
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask[dist <= radius] = 1.0
            band_energy = np.sum(magnitude * mask)
            total_energy = np.sum(magnitude) + 1e-8
            bands[label] = float(band_energy / total_energy)

        return bands

    def _analyze_noise_pattern(self, gray: np.ndarray) -> float:
        """
        Analyze high-frequency noise patterns.

        GAN upsampling creates specific noise signatures
        that differ from natural camera noise.
        """
        # High-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(gray.astype(np.float32), -1, kernel)

        # Analyze noise statistics
        noise_std = np.std(high_freq)
        noise_mean = np.abs(np.mean(high_freq))

        # Natural images: noise is centered around 0, moderate variance
        # GAN images: noise patterns are more structured
        regularity = noise_std / (noise_mean + 1e-8)

        # Check for grid patterns (upsampling artifacts)
        f_noise = np.fft.fft2(high_freq)
        f_mag = np.abs(np.fft.fftshift(f_noise))
        center = np.array(f_mag.shape) // 2
        peaks = self._find_peaks(self._radial_profile(f_mag, center))

        grid_score = len(peaks) / 20.0  # Normalize
        score = float(np.clip(grid_score * 0.5 + regularity * 0.01, 0, 1))

        return score

    def _analyze_texture_smoothness(self, gray: np.ndarray) -> float:
        """
        Score how unnaturally smooth an image is.

        Diffusion / ChatGPT-style portraits often lack natural facial
        micro-texture (pores, fine noise), producing very low Laplacian variance.
        """
        lap_var = float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())
        # Empirically: natural cropped faces often >> 100; heavy AI smoothing << 40
        if lap_var < 25:
            score = 0.95
        elif lap_var < 50:
            score = 0.80
        elif lap_var < 90:
            score = 0.55
        elif lap_var < 150:
            score = 0.30
        else:
            score = 0.10
        return float(score)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to model input tensor."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        resized = cv2.resize(rgb, (256, 256))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    @staticmethod
    def _radial_profile(data: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        """Compute radial average of 2D array."""
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / (nr + 1e-8)
        return radialprofile

    @staticmethod
    def _find_peaks(data: np.ndarray, min_prominence: float = 0.1) -> list[int]:
        """Simple peak detection."""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                if data[i] > min_prominence * np.max(data + 1e-8):
                    peaks.append(i)
        return peaks

    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Generate frequency domain visualizations."""
        output = self.predict(input_data, **kwargs)
        return {
            "fft_magnitude": output.explainability["fft_magnitude"],
            "dct_coefficients": output.explainability["dct_coefficients"],
            "fft_score": output.metadata["fft_score"],
            "dct_score": output.metadata["dct_score"],
            "band_energy": output.metadata["band_energy"],
            "noise_pattern": output.metadata["noise_pattern"],
        }
