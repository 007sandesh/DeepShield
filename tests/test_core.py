"""
DeepShield Test Suite.

Comprehensive tests for all detection layers and pipelines.
"""

import numpy as np
import pytest
import cv2


# ─── Fixtures ──────────────────────────────────────────────
@pytest.fixture
def sample_face_image():
    """Generate a synthetic face-like image for testing."""
    img = np.random.randint(80, 180, (256, 256, 3), dtype=np.uint8)
    # Add some structure (simulate face-like features)
    cv2.circle(img, (128, 100), 40, (200, 180, 160), -1)  # Face oval
    cv2.circle(img, (110, 90), 8, (50, 50, 50), -1)  # Left eye
    cv2.circle(img, (146, 90), 8, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(img, (128, 130), (15, 8), 0, 0, 180, (100, 80, 80), 2)  # Mouth
    return img


@pytest.fixture
def sample_video_frames():
    """Generate synthetic video frames for testing."""
    frames = []
    for i in range(16):
        img = np.random.randint(80, 180, (256, 256, 3), dtype=np.uint8)
        # Add slight variation between frames
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if i % 4 == 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
        frames.append(img)
    return np.stack(frames)


@pytest.fixture
def sample_audio_path(tmp_path):
    """Generate a synthetic audio file for testing."""
    import wave
    import struct

    filepath = tmp_path / "test_audio.wav"
    sample_rate = 16000
    duration = 2.0
    n_samples = int(sample_rate * duration)

    # Generate sine wave
    samples = []
    for i in range(n_samples):
        value = int(16000 * np.sin(2 * np.pi * 440 * i / sample_rate))
        samples.append(struct.pack('<h', max(-32768, min(32767, value))))

    with wave.open(str(filepath), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b''.join(samples))

    return str(filepath)


# ─── Unit Tests ────────────────────────────────────────────
class TestFaceDetector:
    """Test face detection and extraction."""

    def test_face_extractor_initialization(self):
        from src.core.preprocessors.face_extraction import FaceExtractor
        extractor = FaceExtractor()
        assert extractor is not None
        assert extractor.target_size == (256, 256)

    def test_face_extraction(self, sample_face_image):
        from src.core.preprocessors.face_extraction import FaceExtractor
        extractor = FaceExtractor(min_confidence=0.5)
        faces = extractor.extract_from_image(sample_face_image)
        # Should detect at least something (even if synthetic)
        assert isinstance(faces, list)

    def test_quality_assessment(self, sample_face_image):
        from src.core.preprocessors.face_extraction import FaceExtractor
        extractor = FaceExtractor()
        quality = extractor._assess_quality(sample_face_image)
        assert 0.0 <= quality <= 1.0


class TestFrameSampler:
    """Test frame sampling strategies."""

    def test_sampler_initialization(self):
        from src.core.preprocessors.frame_sampling import FrameSampler, SamplingStrategy
        sampler = FrameSampler(strategy=SamplingStrategy.UNIFORM)
        assert sampler.target_fps == 10.0
        assert sampler.max_frames == 32


class TestFrequencyAnalyzer:
    """Test frequency domain analysis."""

    def test_fft_analysis(self, sample_face_image):
        from src.core.models.frequency_analyzer import FrequencyAnalyzer
        analyzer = FrequencyAnalyzer()
        gray = cv2.cvtColor(sample_face_image, cv2.COLOR_BGR2GRAY)
        score, vis = analyzer._analyze_fft(gray)
        assert 0.0 <= score <= 1.0
        assert vis is not None

    def test_dct_analysis(self, sample_face_image):
        from src.core.models.frequency_analyzer import FrequencyAnalyzer
        analyzer = FrequencyAnalyzer()
        gray = cv2.cvtColor(sample_face_image, cv2.COLOR_BGR2GRAY)
        score, vis = analyzer._analyze_dct(gray)
        assert 0.0 <= score <= 1.0

    def test_band_analysis(self, sample_face_image):
        from src.core.models.frequency_analyzer import FrequencyAnalyzer
        analyzer = FrequencyAnalyzer()
        gray = cv2.cvtColor(sample_face_image, cv2.COLOR_BGR2GRAY)
        bands = analyzer._analyze_bands(gray)
        assert "low" in bands
        assert "mid_low" in bands
        assert "mid_high" in bands


class TestModelRegistry:
    """Test model registry system."""

    def test_list_models(self):
        from src.core.models.base import ModelRegistry
        models = ModelRegistry.list_models()
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_register_model(self):
        from src.core.models.base import BaseModel, ModelRegistry, ModelType

        @ModelRegistry.register
        class TestModel(BaseModel):
            model_type = ModelType.FORGERY_DETECTION
            model_name = "test_model_dummy"

            def load_weights(self, weights_path=None):
                self._is_loaded = True

            def predict(self, input_data, **kwargs):
                pass

            def get_explainability(self, input_data, **kwargs):
                return {}

        models = ModelRegistry.list_models()
        assert "test_model_dummy" in models


class TestEnsemble:
    """Test ensemble classifier."""

    def test_ensemble_initialization(self):
        from src.core.models.ensemble import EnsembleClassifier
        ensemble = EnsembleClassifier()
        assert ensemble is not None
        assert ensemble._is_loaded

    def test_default_weights(self):
        from src.core.models.ensemble import EnsembleClassifier
        ensemble = EnsembleClassifier()
        weights = ensemble._model_weights
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_fallback_prediction(self):
        from src.core.models.ensemble import EnsembleClassifier
        ensemble = EnsembleClassifier()
        result = ensemble._fallback_prediction()
        assert result["prediction"] == "uncertain"
        assert result["confidence"] == 0.0


# ─── Integration Tests ─────────────────────────────────────
@pytest.mark.integration
class TestImagePipeline:
    """Integration tests for image detection pipeline."""

    def test_pipeline_initialization(self):
        from src.core.pipelines.image_pipeline import ImageDetectionPipeline
        pipeline = ImageDetectionPipeline(models=["frequency_analyzer"])
        pipeline.initialize()
        assert pipeline._initialized

    def test_detection_with_synthetic(self, sample_face_image):
        from src.core.pipelines.image_pipeline import ImageDetectionPipeline
        pipeline = ImageDetectionPipeline(models=["frequency_analyzer"])
        pipeline.initialize()
        result = pipeline.detect(sample_face_image)
        assert result.prediction in ("REAL", "FAKE")
        assert 0.0 <= result.confidence <= 1.0


# ─── Performance Tests ─────────────────────────────────────
@pytest.mark.slow
class TestPerformance:
    """Performance benchmark tests."""

    def test_inference_speed(self, sample_face_image):
        """Ensure inference completes within time budget."""
        from src.core.models.frequency_analyzer import FrequencyAnalyzer
        import time

        analyzer = FrequencyAnalyzer()

        start = time.perf_counter()
        for _ in range(10):
            analyzer.predict(sample_face_image)
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 1.0, f"Inference too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
