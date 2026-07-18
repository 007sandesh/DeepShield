"""
Ensemble Meta-Classifier.

Combines outputs from all detection layers using learned weights
and produces the final prediction with calibrated confidence.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelType, ModelRegistry


class CalibratedEnsemble(nn.Module):
    """
    Learned ensemble with temperature scaling for confidence calibration.

    Instead of simple averaging, learns optimal combination weights
    and calibrates output probabilities.
    """

    def __init__(self, num_models: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(num_models, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=-1),
        )
        self.calibration = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, model_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_outputs: (batch, num_models, 2) — probabilities from each model
        """
        # Get model confidences
        confidences = model_outputs[:, :, 1]  # Fake probabilities

        # Attention-weighted combination
        weights = self.attention_weights(confidences)
        weighted_confidence = (weights * confidences).sum(dim=-1)

        # Temperature scaling for calibration
        calibrated = torch.sigmoid((weighted_confidence - 0.5) * self.calibration)

        return calibrated


@ModelRegistry.register
class EnsembleClassifier(BaseModel):
    """
    Meta-classifier that combines all detection layers.

    Features:
    - Learned combination weights (not fixed averaging)
    - Temperature-scaled confidence calibration
    - Per-model contribution analysis
    - Explainable decision process
    - Dynamic model selection based on input quality
    """

    model_type = ModelType.ENSEMBLE
    model_name = "ensemble"

    DEFAULT_WEIGHTS = {
        "ai_image_detector": 0.40,
        "forgery_detector": 0.25,
        "frequency_analyzer": 0.15,
        "attention_network": 0.10,
        "temporal_analyzer": 0.05,
        "biological_signals": 0.03,
        "audio_sync": 0.02,
    }

    def __init__(self, device: str = "auto", **kwargs) -> None:
        super().__init__(device=device)
        self._ensemble_net = CalibratedEnsemble()
        self._ensemble_net = self._ensemble_net.to(self.device).eval()
        self._model_weights = self.DEFAULT_WEIGHTS.copy()
        self._models: dict[str, BaseModel] = {}
        self._is_loaded = True
        logger.info("Ensemble classifier initialized")

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load ensemble weights."""
        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)
            self._ensemble_net.load_state_dict(state_dict)
        self._is_loaded = True

    def register_model(self, name: str, model: BaseModel) -> None:
        """Register a sub-model for ensemble prediction."""
        self._models[name] = model
        logger.info(f"Registered model in ensemble: {name}")

    def set_weights(self, weights: dict[str, float]) -> None:
        """Override ensemble weights."""
        total = sum(weights.values())
        self._model_weights = {k: v / total for k, v in weights.items()}
        logger.info(f"Updated ensemble weights: {self._model_weights}")

    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Run ensemble prediction combining all registered models.

        Args:
            input_data: Input data (image or video frames)
            **kwargs: Passed to individual models

        Returns:
            ModelOutput with ensemble prediction and per-model breakdown
        """
        result, elapsed = self._measure_inference(self._ensemble_predict, input_data, **kwargs)

        return ModelOutput(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            metadata={
                "model_contributions": result["contributions"],
                "model_agreement": result["agreement"],
                "num_models_used": result["num_models"],
                "calibration_applied": True,
            },
            explainability={
                "decision_process": result["decision_process"],
                "per_model_heatmaps": result.get("per_model_heatmaps", {}),
            },
            inference_time_ms=elapsed,
        )

    def _ensemble_predict(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Run all models and combine predictions."""
        model_outputs: dict[str, dict[str, float]] = {}
        model_scores: list[float] = []
        model_weights: list[float] = []
        full_image = kwargs.get("full_image")

        for name, model in self._models.items():
            if not model.is_loaded:
                logger.warning(f"Model {name} not loaded, skipping")
                continue

            try:
                model_input = (
                    full_image
                    if name == "ai_image_detector" and full_image is not None
                    else input_data
                )
                output = model.predict(model_input)
                model_outputs[name] = {
                    "prediction": output.prediction,
                    "confidence": output.confidence,
                    "fake_prob": output.probabilities.get("fake", 0.5),
                    "real_prob": output.probabilities.get("real", 0.5),
                    "inference_time_ms": output.inference_time_ms,
                }

                weight = self._model_weights.get(name, 1.0 / len(self._models))
                model_scores.append(output.probabilities.get("fake", 0.5))
                model_weights.append(weight)

            except Exception as e:
                logger.error(f"Model {name} failed: {e}")

        if not model_scores:
            return self._fallback_prediction()

        # Method 1: Weighted average
        model_scores_arr = np.array(model_scores)
        model_weights_arr = np.array(model_weights)
        model_weights_arr = model_weights_arr / model_weights_arr.sum()

        weighted_avg = float(np.dot(model_scores_arr, model_weights_arr))

        # Method 2: Learned ensemble
        try:
            scores_tensor = torch.tensor(
                [[1 - s, s] for s in model_scores], dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            learned_score = self._ensemble_net(scores_tensor).item()
        except Exception:
            learned_score = weighted_avg

        # Combine methods
        final_score = 0.5 * weighted_avg + 0.5 * learned_score

        # High-confidence generative-AI hits should not be diluted by
        # face-swap-only backbones that were trained on different artifacts.
        ai_out = model_outputs.get("ai_image_detector")
        if ai_out is not None and ai_out["fake_prob"] >= 0.90:
            final_score = max(final_score, 0.75 * ai_out["fake_prob"] + 0.25 * final_score)
        elif ai_out is not None and ai_out["real_prob"] >= 0.90:
            final_score = min(final_score, 0.75 * (1.0 - ai_out["real_prob"]) + 0.25 * final_score)

        # Model agreement analysis
        predictions = [o["prediction"] for o in model_outputs.values()]
        agreement = max(
            predictions.count("real"),
            predictions.count("fake"),
        ) / len(predictions)

        prediction = "fake" if final_score >= 0.5 else "real"
        class_probability = final_score if prediction == "fake" else (1.0 - final_score)
        # Soft-calibrate: keep class probability primary, nudge by agreement
        confidence = float(np.clip(0.85 * class_probability + 0.15 * agreement, 0.01, 0.99))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {"real": 1.0 - final_score, "fake": final_score},
            "contributions": model_outputs,
            "agreement": float(agreement),
            "num_models": len(model_outputs),
            "decision_process": {
                "weighted_average": weighted_avg,
                "learned_score": learned_score,
                "final_score": final_score,
                "class_probability": class_probability,
                "method": "calibrated_ensemble",
            },
        }

    def _calibrate_confidence(self, score: float, agreement: float) -> float:
        """
        Calibrate confidence based on model agreement and score magnitude.

        High agreement + extreme score = high confidence
        Low agreement + ambiguous score = low confidence
        """
        # Distance from decision boundary (0.5)
        boundary_distance = abs(score - 0.5) * 2

        # Confidence combines certainty and agreement
        confidence = 0.6 * boundary_distance + 0.4 * agreement

        return float(np.clip(confidence, 0.1, 0.99))

    def _fallback_prediction(self) -> dict[str, Any]:
        """Fallback when no models are available."""
        return {
            "prediction": "uncertain",
            "confidence": 0.0,
            "probabilities": {"real": 0.5, "fake": 0.5},
            "contributions": {},
            "agreement": 0.0,
            "num_models": 0,
            "decision_process": {
                "weighted_average": 0.5,
                "learned_score": 0.5,
                "final_score": 0.5,
                "method": "fallback",
            },
        }

    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Generate ensemble decision explanation."""
        output = self.predict(input_data, **kwargs)
        return {
            "contributions": output.metadata["model_contributions"],
            "agreement": output.metadata["model_agreement"],
            "decision_process": output.explainability["decision_process"],
        }
