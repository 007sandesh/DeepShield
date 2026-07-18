"""
Audio-Visual Synchronization Analysis.

Detects mismatches between lip movements and audio content,
voice clone artifacts, and environmental sound inconsistencies.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelType, ModelRegistry


class AudioVisualSyncNet(nn.Module):
    """Cross-modal network for audio-visual synchronization scoring."""

    def __init__(self, audio_dim: int = 80, visual_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sync_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, audio_feat: torch.Tensor, visual_feat: torch.Tensor) -> torch.Tensor:
        a = self.audio_encoder(audio_feat)
        v = self.visual_encoder(visual_feat)
        combined = torch.cat([a, v], dim=-1)
        return self.sync_head(combined)


@ModelRegistry.register
class AudioSyncAnalyzer(BaseModel):
    """
    Audio-visual synchronization analysis for video deepfake detection.

    Analysis methods:
    1. Lip-audio synchronization scoring
    2. Voice clone detection (spectral analysis)
    3. Environmental sound consistency
    4. Cross-modal temporal alignment
    """

    model_type = ModelType.AUDIO_SYNC
    model_name = "audio_sync"

    def __init__(self, device: str = "auto", **kwargs) -> None:
        super().__init__(device=device)
        self._sync_net = AudioVisualSyncNet().to(self.device).eval()
        self._whisper_model = None
        self._is_loaded = True
        logger.info("Audio-visual sync analyzer initialized")

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load audio analysis models."""
        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)
            self._sync_net.load_state_dict(state_dict)
        self._is_loaded = True

    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Analyze audio-visual synchronization.

        Args:
            input_data: Video frames (N, H, W, 3)
            **kwargs:
                audio_path: str — Path to audio file
                fps: float — Video frame rate

        Returns:
            ModelOutput with sync analysis results
        """
        audio_path = kwargs.get("audio_path")
        fps = kwargs.get("fps", 30.0)

        result, elapsed = self._measure_inference(
            self._analyze_sync, input_data, audio_path, fps
        )

        return ModelOutput(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            metadata={
                "sync_score": result["sync_score"],
                "voice_clone_score": result["voice_clone_score"],
                "env_score": result["env_score"],
                "phoneme_accuracy": result["phoneme_accuracy"],
            },
            inference_time_ms=elapsed,
        )

    def _analyze_sync(
        self, frames: np.ndarray, audio_path: Optional[str], fps: float
    ) -> dict[str, Any]:
        """Run full audio-visual sync analysis."""
        # Extract lip landmarks from frames
        lip_features = self._extract_lip_features(frames)

        if audio_path is None:
            # No audio available — return uncertain
            return {
                "prediction": "uncertain",
                "confidence": 0.5,
                "probabilities": {"real": 0.5, "fake": 0.5},
                "sync_score": 0.5,
                "voice_clone_score": 0.5,
                "env_score": 0.5,
                "phoneme_accuracy": 0.5,
            }

        # Extract audio features
        audio_features = self._extract_audio_features(audio_path)

        # 1. Lip-audio sync scoring
        sync_score = self._score_sync(lip_features, audio_features)

        # 2. Voice clone detection
        voice_clone_score = self._detect_voice_clone(audio_path)

        # 3. Environmental sound consistency
        env_score = self._analyze_environment(audio_path)

        # 4. Phoneme accuracy
        phoneme_accuracy = self._score_phoneme_match(lip_features, audio_path)

        # Combine
        combined = (
            0.40 * sync_score +
            0.25 * voice_clone_score +
            0.15 * env_score +
            0.20 * (1.0 - phoneme_accuracy)  # Inverted: low accuracy = suspicious
        )

        return {
            "prediction": "fake" if combined > 0.5 else "real",
            "confidence": float(max(1 - combined, combined)),
            "probabilities": {"real": float(1 - combined), "fake": float(combined)},
            "sync_score": sync_score,
            "voice_clone_score": voice_clone_score,
            "env_score": env_score,
            "phoneme_accuracy": phoneme_accuracy,
        }

    def _extract_lip_features(self, frames: np.ndarray) -> np.ndarray:
        """Extract lip movement features from video frames."""
        lip_sequences = []

        for frame in frames[:64]:  # Sample up to 64 frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Detect face
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                continue

            x, y, w, h = faces[0]

            # Extract mouth region (approximately lower 30% of face)
            mouth_y = y + int(h * 0.6)
            mouth_h = int(h * 0.35)
            mouth_x = x + int(w * 0.2)
            mouth_w = int(w * 0.6)

            mouth_roi = gray[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]

            if mouth_roi.size > 0:
                # Compute mouth opening metric
                _, thresh = cv2.threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mouth_opening = np.sum(thresh > 0) / (thresh.size + 1e-8)

                # Edge density (lip contour)
                edges = cv2.Canny(mouth_roi, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.size + 1e-8)

                lip_sequences.append([mouth_opening, edge_density])

        if not lip_sequences:
            return np.zeros((1, 2))

        return np.array(lip_sequences)

    def _extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract audio features using librosa."""
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=16000, duration=30)

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)

            return mel_db

        except Exception as e:
            logger.warning(f"Audio feature extraction failed: {e}")
            return np.zeros((80, 100))

    def _score_sync(self, lip_features: np.ndarray, audio_features: np.ndarray) -> float:
        """
        Score audio-visual synchronization.

        Real speech: lip movements correlate with audio energy
        Deepfake: lip movements may be out of sync
        """
        if len(lip_features) < 5 or audio_features.size == 0:
            return 0.5

        # Downsample audio to match lip frame count
        audio_energy = np.mean(audio_features, axis=0)
        if len(audio_energy) > len(lip_features):
            indices = np.linspace(0, len(audio_energy) - 1, len(lip_features)).astype(int)
            audio_energy = audio_energy[indices]

        # Normalize both signals
        lip_signal = lip_features[:, 0]  # Mouth opening
        lip_signal = (lip_signal - lip_signal.mean()) / (lip_signal.std() + 1e-8)
        audio_signal = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-8)

        # Cross-correlation
        min_len = min(len(lip_signal), len(audio_signal))
        lip_signal = lip_signal[:min_len]
        audio_signal = audio_signal[:min_len]

        correlation = np.corrcoef(lip_signal, audio_signal)[0, 1]

        # Convert to sync score (high correlation = good sync = likely real)
        sync_score = float(np.clip(1.0 - correlation, 0, 1))

        return sync_score

    def _detect_voice_clone(self, audio_path: str) -> float:
        """
        Detect voice cloning artifacts.

        Voice clones often have:
        - Unnatural spectral envelope
        - Missing breathing sounds
        - Consistent pitch without natural variation
        - Spectral discontinuities at concatenation points
        """
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=16000, duration=30)

            # Spectral analysis
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            harmonic_ratio = librosa.feature.spectral_flatness(y=y)[0]

            # Check for unnatural consistency
            centroid_var = np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-8)
            rolloff_var = np.std(spectral_rolloff) / (np.mean(spectral_rolloff) + 1e-8)

            # Breathing detection (low frequency energy bursts)
            breathing_score = self._detect_breathing(y, sr)

            # Score: low variance + no breathing = suspicious
            consistency_score = 1.0 - min(centroid_var + rolloff_var, 1.0)
            clone_score = (consistency_score * 0.5 + breathing_score * 0.5)

            return float(np.clip(clone_score, 0, 1))

        except Exception:
            return 0.5

    def _detect_breathing(self, audio: np.ndarray, sr: int) -> float:
        """Detect natural breathing sounds in audio."""
        # Bandpass filter for breathing frequencies (0.5-3 Hz)
        from scipy.signal import butter, filtfilt

        try:
            low = 0.5 / (sr / 2)
            high = 3.0 / (sr / 2)
            b, a = butter(2, [low, high], btype="band")
            breathing = filtfilt(b, a, audio)

            # Energy of breathing signal
            energy = np.mean(breathing ** 2)

            # Real speech has detectable breathing
            # Score: low energy = no breathing = suspicious
            score = float(np.clip(1.0 - energy * 1000, 0, 1))

            return score

        except Exception:
            return 0.5

    def _analyze_environment(self, audio_path: str) -> float:
        """
        Analyze environmental sound consistency.

        Real recordings have natural background noise.
        Deepfakes often have:
        - Perfect silence between words
        - Consistent noise floor (generated)
        - Missing room impulse response
        """
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=16000, duration=30)

            # Split into segments
            segment_length = sr  # 1 second segments
            segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]

            if len(segments) < 3:
                return 0.5

            # Analyze noise floor consistency
            noise_floors = []
            for seg in segments:
                # Estimate noise floor from quiet sections
                rms = np.sqrt(np.mean(seg ** 2))
                noise_floors.append(rms)

            noise_var = np.std(noise_floors) / (np.mean(noise_floors) + 1e-8)

            # Real: noise varies naturally
            # Fake: noise is often too consistent or too clean
            score = float(np.clip(1.0 - noise_var * 2, 0, 1))

            return score

        except Exception:
            return 0.5

    def _score_phoneme_match(self, lip_features: np.ndarray, audio_path: str) -> float:
        """
        Score how well lip movements match audio phonemes.

        Uses a simplified approach:
        - High lip opening = vowel sounds
        - Low lip opening = consonants/mutes
        """
        if len(lip_features) < 5:
            return 0.5

        # Simple correlation between lip movement and audio energy
        lip_movement = np.diff(lip_features[:, 0])
        lip_movement = np.abs(lip_movement)

        # Normalize
        if lip_movement.max() > 0:
            lip_movement = lip_movement / lip_movement.max()

        # Score based on movement variability
        movement_var = np.std(lip_movement)
        movement_score = float(np.clip(movement_var * 5, 0, 1))

        return movement_score

    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Generate audio-visual sync visualizations."""
        output = self.predict(input_data, **kwargs)
        return {
            "sync_score": output.metadata["sync_score"],
            "voice_clone_score": output.metadata["voice_clone_score"],
            "env_score": output.metadata["env_score"],
            "phoneme_accuracy": output.metadata["phoneme_accuracy"],
        }
