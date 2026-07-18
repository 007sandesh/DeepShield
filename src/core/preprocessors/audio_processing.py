"""
Audio Processing for video analysis.

Extracts and processes audio from video files for
audio-visual synchronization analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class AudioData:
    """Container for extracted audio data."""

    waveform: np.ndarray  # Raw audio samples
    sample_rate: int  # Sample rate (Hz)
    duration: float  # Duration in seconds
    spectrogram: Optional[np.ndarray] = None  # Mel spectrogram
    mel_features: Optional[np.ndarray] = None  # Extracted mel features
    phonemes: Optional[list[dict]] = None  # Phoneme timestamps


class AudioProcessor:
    """
    Audio extraction and processing pipeline.

    Features:
    - Extract audio from video files
    - Convert to mel spectrograms
    - Compute audio features
    - Phoneme extraction (using Whisper)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        duration: Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration

    def extract_from_video(self, video_path: str) -> Optional[AudioData]:
        """
        Extract audio track from video file.

        Args:
            video_path: Path to video file

        Returns:
            AudioData or None if extraction fails
        """
        try:
            import subprocess
            import tempfile

            # Extract audio using ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                tmp_path,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                logger.warning(f"ffmpeg extraction failed: {result.stderr}")
                return None

            # Load extracted audio
            audio_data = self._load_audio(tmp_path)

            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

            return audio_data

        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None

    def _load_audio(self, audio_path: str) -> AudioData:
        """Load and process audio file."""
        try:
            import librosa

            # Load audio
            y, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                duration=self.duration,
            )

            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels, fmax=8000
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Additional features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            duration = len(y) / sr

            return AudioData(
                waveform=y,
                sample_rate=sr,
                duration=duration,
                spectrogram=mel_spec,
                mel_features=mel_db,
            )

        except ImportError:
            logger.warning("librosa not available, using basic audio loading")
            return self._basic_load(audio_path)

    def _basic_load(self, audio_path: str) -> AudioData:
        """Basic audio loading without librosa."""
        import wave

        with wave.open(audio_path, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0

        return AudioData(
            waveform=audio,
            sample_rate=sr,
            duration=n_frames / sr,
        )

    def extract_phonemes(self, audio_data: AudioData) -> list[dict]:
        """
        Extract phoneme timestamps using Whisper.

        Returns list of phonemes with start/end times.
        """
        try:
            import whisper

            model = whisper.load_model("base")

            # Transcribe with word timestamps
            result = model.transcribe(
                audio_data.waveform,
                language=None,
                word_timestamps=True,
            )

            phonemes = []
            for segment in result["segments"]:
                for word_info in segment.get("words", []):
                    phonemes.append({
                        "word": word_info["word"],
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "probability": word_info.get("probability", 0.0),
                    })

            return phonemes

        except Exception as e:
            logger.warning(f"Phoneme extraction failed: {e}")
            return []

    def compute_features(self, audio_data: AudioData) -> dict[str, np.ndarray]:
        """
        Compute audio features for analysis.

        Returns dict of feature arrays.
        """
        features = {}

        try:
            import librosa

            y = audio_data.waveform
            sr = audio_data.sample_rate

            # RMS energy
            features["rms"] = librosa.feature.rms(y=y)[0]

            # Zero crossing rate
            features["zcr"] = librosa.feature.zero_crossing_rate(y)[0]

            # Spectral contrast
            features["spectral_contrast"] = librosa.feature.spectral_contrast(
                y=y, sr=sr
            )

            # Chroma features
            features["chroma"] = librosa.feature.chroma_stft(y=y, sr=sr)

        except ImportError:
            # Basic features without librosa
            features["rms"] = np.sqrt(np.mean(audio_data.waveform ** 2))

        return features
