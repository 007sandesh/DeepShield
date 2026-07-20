"""
Image Detection Pipeline.

End-to-end pipeline for analyzing single images:
Image → Face Extraction → Multi-Model Analysis → Ensemble → Result
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelRegistry
from src.core.preprocessors.face_extraction import FaceExtractor, ExtractedFace
import src.core.models  # noqa: F401 — register detection models


@dataclass
class DetectionResult:
    """Complete detection result with all analysis layers."""

    # Core prediction
    prediction: str  # "REAL" or "FAKE"
    confidence: float  # Probability of the predicted class (0.0 - 1.0)
    probabilities: dict[str, float] = field(
        default_factory=lambda: {"real": 0.5, "fake": 0.5}
    )

    # Dual scores
    ai_gen_score: Optional[float] = None  # AI-generated image score
    face_forgery_score: Optional[float] = None  # Face-swap forgery score

    # Per-model breakdown
    model_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Face metadata
    faces_detected: int = 0
    best_face_quality: float = 0.0

    # Explainability
    explanation: str = ""
    heatmap: Optional[np.ndarray] = None
    attention_maps: Optional[list] = None

    # Performance
    total_time_ms: float = 0.0
    model_times_ms: dict[str, float] = field(default_factory=dict)

    # Metadata
    input_type: str = "image"
    input_shape: tuple = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        real_p = float(self.probabilities.get("real", 1.0 - self.confidence))
        fake_p = float(self.probabilities.get("fake", self.confidence))
        # Keep probs normalized and consistent with the predicted label
        total = real_p + fake_p
        if total > 0:
            real_p, fake_p = real_p / total, fake_p / total
        result = {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "probability": {
                "real": round(real_p, 4),
                "fake": round(fake_p, 4),
            },
            "faces_detected": self.faces_detected,
            "best_face_quality": round(self.best_face_quality, 4),
            "explanation": self.explanation,
            "model_results": self.model_results,
            "performance": {
                "total_time_ms": round(self.total_time_ms, 2),
                "model_times_ms": self.model_times_ms,
            },
        }
        if self.ai_gen_score is not None:
            result["ai_gen_score"] = round(self.ai_gen_score, 4)
        if self.face_forgery_score is not None:
            result["face_forgery_score"] = round(self.face_forgery_score, 4)
        return result


class ImageDetectionPipeline:
    """
    Complete image analysis pipeline.

    Flow:
    1. Load and validate image
    2. Extract faces (highest quality)
    3. Run all detection models on best face
    4. Combine results via ensemble
    5. Generate explanation
    """

    def __init__(
        self,
        models: Optional[list[str]] = None,
        device: str = "auto",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.face_extractor = FaceExtractor()
        self.confidence_threshold = confidence_threshold
        self._models: dict[str, BaseModel] = {}
        self._ensemble = None
        self._device = device
        self._initialized = False

        self._model_names = models or [
            "ai_image_detector",
            "forgery_detector",
        ]

    def initialize(self) -> None:
        """Initialize all models (lazy loading)."""
        if self._initialized:
            return

        logger.info("Initializing Image Detection Pipeline...")

        # Load individual models
        for name in self._model_names:
            try:
                model = ModelRegistry.create(name, device=self._device)
                self._models[name] = model
                logger.info(f"  ✓ {name}")
            except Exception as e:
                logger.warning(f"  ✗ {name}: {e}")

        # Load ensemble
        try:
            self._ensemble = ModelRegistry.create("ensemble", device=self._device)
            for name, model in self._models.items():
                self._ensemble.register_model(name, model)
            logger.info("  ✓ ensemble")
        except Exception as e:
            logger.warning(f"  ✗ ensemble: {e}")

        self._initialized = True
        logger.info(
            f"Pipeline initialized with {len(self._models)} models"
        )

    def detect(
        self,
        input_data: str | np.ndarray,
        extract_best_face: bool = True,
    ) -> DetectionResult:
        """
        Analyze an image for deepfake detection.

        Args:
            input_data: File path (str) or image array (np.ndarray)
            extract_best_face: Whether to extract and analyze best face

        Returns:
            DetectionResult with full analysis
        """
        start_time = time.perf_counter()

        if not self._initialized:
            self.initialize()

        # Load image
        if isinstance(input_data, str):
            image = cv2.imread(input_data)
            if image is None:
                raise ValueError(f"Cannot load image: {input_data}")
        else:
            image = input_data

        logger.info(f"Analyzing image: {image.shape}")

        # Extract faces
        faces = []
        if extract_best_face:
            faces = self.face_extractor.extract_from_image(image, max_faces=1)

        if not faces:
            logger.warning("No faces detected, analyzing full image")
            # Fallback: resize full image
            resized = cv2.resize(image, (256, 256))
            faces = [ExtractedFace(
                image=resized,
                bbox=(0, 0, image.shape[1], image.shape[0]),
                confidence=0.5,
                quality_score=0.5,
            )]

        best_face = faces[0]
        logger.info(
            f"Best face: quality={best_face.quality_score:.2f}, "
            f"confidence={best_face.confidence:.2f}"
        )

        # Run models — cache raw ModelOutput objects for ensemble to avoid re-inference
        model_results_raw: dict[str, ModelOutput] = {}
        model_results = {}
        model_times = {}

        for name, model in self._models.items():
            try:
                # Generative-AI detectors work best on the full frame
                model_input = image if name == "ai_image_detector" else best_face.image
                output = model.predict(model_input)
                model_results_raw[name] = output
                model_results[name] = output.to_dict()
                model_times[name] = output.inference_time_ms

                logger.info(
                    f"  {name}: {output.prediction} "
                    f"({output.confidence:.2f}) "
                    f"[{output.inference_time_ms:.1f}ms]"
                )

            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                model_results[name] = {"error": str(e)}

        # Ensemble prediction — pass cached model results to avoid double inference
        probabilities = {"real": 0.5, "fake": 0.5}
        ai_gen_score = None
        face_forgery_score = None
        if self._ensemble:
            try:
                ensemble_output = self._ensemble.predict(
                    best_face.image,
                    model_results=model_results_raw,  # cached — no re-run
                    full_image=image,
                )
                probabilities = {
                    "real": float(ensemble_output.probabilities.get("real", 0.5)),
                    "fake": float(ensemble_output.probabilities.get("fake", 0.5)),
                }
                # Extract dual scores from ensemble metadata
                meta = ensemble_output.metadata or {}
                ai_gen_score = meta.get("ai_gen_score")
                face_forgery_score = meta.get("face_forgery_score")
                final_prediction, final_confidence, probabilities = self._resolve_prediction(
                    probabilities
                )
            except Exception as e:
                logger.error(f"Ensemble failed: {e}")
                final_prediction, final_confidence, probabilities = self._majority_vote(
                    model_results
                )
        else:
            final_prediction, final_confidence, probabilities = self._majority_vote(
                model_results
            )

        # Generate explanation
        explanation = self._generate_explanation(
            final_prediction, final_confidence, model_results, best_face
        )

        # Generate heatmap
        heatmap = self._generate_heatmap(best_face.image, model_results)

        total_time = (time.perf_counter() - start_time) * 1000

        return DetectionResult(
            prediction=final_prediction,
            confidence=final_confidence,
            probabilities=probabilities,
            ai_gen_score=ai_gen_score,
            face_forgery_score=face_forgery_score,
            model_results=model_results,
            faces_detected=len(faces),
            best_face_quality=best_face.quality_score,
            explanation=explanation,
            heatmap=heatmap,
            total_time_ms=total_time,
            model_times_ms=model_times,
            input_type="image",
            input_shape=image.shape,
        )

    def _resolve_prediction(
        self, probabilities: dict[str, float]
    ) -> tuple[str, float, dict[str, float]]:
        """Pick the stronger class and set confidence to that class probability."""
        real_p = float(probabilities.get("real", 0.5))
        fake_p = float(probabilities.get("fake", 0.5))
        total = real_p + fake_p
        if total > 0:
            real_p, fake_p = real_p / total, fake_p / total
        else:
            real_p = fake_p = 0.5

        if fake_p >= real_p:
            return "FAKE", fake_p, {"real": real_p, "fake": fake_p}
        return "REAL", real_p, {"real": real_p, "fake": fake_p}

    def _majority_vote(
        self, model_results: dict[str, dict]
    ) -> tuple[str, float, dict[str, float]]:
        """Fallback: average fake probabilities from successful model results."""
        fake_scores: list[float] = []

        for result in model_results.values():
            if "error" in result or "prediction" not in result:
                continue

            probs = result.get("probabilities") or {}
            if "fake" in probs:
                fake_scores.append(float(probs["fake"]))
                continue

            conf = float(result.get("confidence", 0.5))
            pred = str(result["prediction"]).upper()
            fake_scores.append(conf if pred == "FAKE" else 1.0 - conf)

        if not fake_scores:
            return "UNCERTAIN", 0.0, {"real": 0.5, "fake": 0.5}

        avg_fake = float(np.mean(fake_scores))
        return self._resolve_prediction({"real": 1.0 - avg_fake, "fake": avg_fake})

    def _generate_explanation(
        self,
        prediction: str,
        confidence: float,
        model_results: dict[str, dict],
        face: ExtractedFace,
    ) -> str:
        """Generate human-readable explanation."""
        parts = []

        if prediction == "FAKE":
            parts.append(f"This image appears to be AI-generated (confidence: {confidence:.0%}).")

            # Identify which models flagged it
            flagged = []
            for name, result in model_results.items():
                if result.get("prediction", "").upper() == "FAKE":
                    flagged.append(name.replace("_", " ").title())

            if flagged:
                parts.append(f"Detection signals from: {', '.join(flagged)}.")

            # Add specific insights
            if "frequency_analyzer" in model_results:
                freq_score = model_results["frequency_analyzer"].get("metadata", {}).get("fft_score", 0)
                if freq_score > 0.5:
                    parts.append("Frequency analysis detected GAN spectral artifacts.")

            if "biological_signals" in model_results:
                bio = model_results["biological_signals"].get("metadata", {})
                if not bio.get("has_rppg_signal", True):
                    parts.append("No natural blood flow (rPPG) signal detected in skin.")

            if "temporal_analyzer" in model_results:
                temp = model_results["temporal_analyzer"].get("metadata", {})
                if temp.get("jitter_score", 0) > 0.5:
                    parts.append("Facial landmark jitter detected across frames.")

        else:
            parts.append(f"This image appears to be authentic (confidence: {confidence:.0%}).")
            parts.append("No significant deepfake artifacts detected across analysis layers.")

        return " ".join(parts)

    def _generate_heatmap(
        self, face: np.ndarray, model_results: dict
    ) -> Optional[np.ndarray]:
        """Generate combined heatmap from model explainability."""
        try:
            # Try attention network heatmap
            if "attention_network" in model_results:
                attn = model_results["attention_network"]
                if "explainability" in attn and "manipulation_map" in attn["explainability"]:
                    return np.array(attn["explainability"]["manipulation_map"])

            # Try frequency analysis
            if "frequency_analyzer" in model_results:
                freq = model_results["frequency_analyzer"]
                if "explainability" in freq and "fft_magnitude" in freq["explainability"]:
                    return np.array(freq["explainability"]["fft_magnitude"])

        except Exception:
            pass

        return None
