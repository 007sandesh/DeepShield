"""
AI-generated image detector.

Uses a vision classifier trained on synthetic vs authentic photos
(diffusion / SDXL / ChatGPT-style generations), complementary to
face-swap deepfake detectors.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelType, ModelRegistry


@ModelRegistry.register
class AIImageDetector(BaseModel):
    """
    Detect AI-generated (synthetic) images.

    Default backbone: Organika/sdxl-detector (labels: artificial / human).
    Operates on the full frame (not only the face crop).
    """

    model_type = ModelType.FORGERY_DETECTION
    model_name = "ai_image_detector"

    DEFAULT_MODEL_ID = "Organika/sdxl-detector"

    def __init__(
        self,
        device: str = "auto",
        model_id: str = DEFAULT_MODEL_ID,
        **kwargs,
    ) -> None:
        super().__init__(device=device)
        self.model_id = model_id
        self._pipeline = None
        self.load_weights()

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load HuggingFace image-classification pipeline."""
        import time

        start = time.perf_counter()
        try:
            from transformers import pipeline

            device_idx = 0 if self.device == "cuda" else -1
            self._pipeline = pipeline(
                "image-classification",
                model=self.model_id,
                device=device_idx,
            )
            self._is_loaded = True
            self._load_time_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"AI image detector loaded ({self.model_id}) in {self._load_time_ms:.1f}ms"
            )
        except Exception as e:
            self._is_loaded = False
            logger.error(f"Failed to load AI image detector: {e}")
            raise

    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Classify an image as AI-generated or authentic.

        Args:
            input_data: BGR image (H, W, 3) — preferably full frame
        """
        result, elapsed = self._measure_inference(self._classify, input_data)
        return ModelOutput(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            metadata={
                "model_id": self.model_id,
                "raw_scores": result["raw_scores"],
            },
            inference_time_ms=elapsed,
        )

    def _classify(self, image: np.ndarray) -> dict[str, Any]:
        from PIL import Image

        if image is None:
            raise ValueError("Empty image")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        pil_image = Image.fromarray(rgb)

        scores = self._pipeline(pil_image)
        # Normalize label variants to fake/real
        fake_score = 0.0
        real_score = 0.0
        raw: dict[str, float] = {}
        for item in scores:
            label = str(item["label"]).lower()
            score = float(item["score"])
            raw[label] = score
            if label in {"artificial", "ai", "fake", "generated", "synthetic"}:
                fake_score = max(fake_score, score)
            elif label in {"human", "real", "authentic", "photo"}:
                real_score = max(real_score, score)

        # If only one side was matched, derive the other
        if fake_score == 0.0 and real_score == 0.0 and scores:
            # Fallback: highest score label assumed "fake-like" if unknown
            top = max(scores, key=lambda x: x["score"])
            if str(top["label"]).lower() in {"artificial", "ai", "fake", "generated"}:
                fake_score = float(top["score"])
                real_score = 1.0 - fake_score
            else:
                real_score = float(top["score"])
                fake_score = 1.0 - real_score
        elif fake_score > 0.0 and real_score == 0.0:
            real_score = 1.0 - fake_score
        elif real_score > 0.0 and fake_score == 0.0:
            fake_score = 1.0 - real_score

        total = fake_score + real_score
        if total > 0:
            fake_score /= total
            real_score /= total

        prediction = "fake" if fake_score >= real_score else "real"
        confidence = max(fake_score, real_score)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {"real": real_score, "fake": fake_score},
            "raw_scores": raw,
        }

    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        output = self.predict(input_data, **kwargs)
        return {
            "raw_scores": output.metadata.get("raw_scores", {}),
            "probabilities": output.probabilities,
        }
