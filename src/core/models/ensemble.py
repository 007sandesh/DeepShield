"""
Ensemble Meta-Classifier.

Production fusion (research-backed):
  - Domain specialists: AI-gen detector (full frame) + face-forgery ensemble (face crop)
  - Final FAKE score = soft OR / max across domains (DeepDect / MIT ensemble practice)
  - Never let "AI says real" suppress a face-forgery FAKE signal
  - Untrained models (attention_network, CalibratedEnsemble) gated out by default
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

    NOTE: This module has no pre-trained weights and is considered
    **experimental**. It is NOT used in the default prediction path —
    only the weighted-average path is active unless use_learned=True.
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

    Production path (default):
      Dual-domain soft-OR of AI-gen + face-forgery specialists.
      Frequency/temporal are advisory only (low weight / gated).

    Experimental: Learned CalibratedEnsemble (use_learned=True).
    """

    model_type = ModelType.ENSEMBLE
    model_name = "ensemble"

    # Advisory weights — used only when domain specialists are missing.
    # Primary path uses domain OR of ai_image_detector + forgery_detector.
    DEFAULT_WEIGHTS = {
        "ai_image_detector": 0.50,
        "forgery_detector": 0.50,
        "frequency_analyzer": 0.0,
        "temporal_analyzer": 0.0,
        "attention_network": 0.0,
        "biological_signals": 0.0,
        "audio_sync": 0.0,
    }

    def __init__(
        self,
        device: str = "auto",
        use_learned: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(device=device)
        self._use_learned = use_learned
        if use_learned:
            self._ensemble_net = CalibratedEnsemble()
            self._ensemble_net = self._ensemble_net.to(self.device).eval()
        else:
            self._ensemble_net = None
        self._model_weights = self.DEFAULT_WEIGHTS.copy()
        self._models: dict[str, BaseModel] = {}
        self._is_loaded = True
        logger.info(f"Ensemble classifier initialized (use_learned={use_learned})")

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load ensemble weights (no-op unless use_learned=True)."""
        if weights_path and self._use_learned:
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

    def predict(
        self,
        input_data: np.ndarray,
        model_results: Optional[dict[str, ModelOutput]] = None,
        **kwargs,
    ) -> ModelOutput:
        """
        Run ensemble prediction.

        Args:
            input_data: Input data (image or video frames) — used only when
                        model_results is not provided (legacy path).
            model_results: **Pre-computed model outputs** to combine.
                           Passing this avoids re-running every model (fixes
                           the double-inference problem).
            **kwargs: Passed to individual models if model_results is None.

        Returns:
            ModelOutput with ensemble prediction and per-model breakdown
        """
        result, elapsed = self._measure_inference(
            self._ensemble_predict, input_data, model_results, **kwargs,
        )

        return ModelOutput(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            metadata={
                "model_contributions": result["contributions"],
                "model_agreement": result["agreement"],
                "num_models_used": result["num_models"],
                "calibration_applied": self._use_learned,
                "ai_gen_score": result["ai_gen_score"],
                "face_forgery_score": result["face_forgery_score"],
            },
            explainability={
                "decision_process": result["decision_process"],
            },
            inference_time_ms=elapsed,
        )

    def _ensemble_predict(
        self,
        input_data: np.ndarray,
        model_results: Optional[dict[str, ModelOutput]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Combine predictions — either from pre-computed results or by running models."""
        model_outputs: dict[str, dict[str, float]] = {}
        model_scores: list[float] = []
        model_weights: list[float] = []
        full_image = kwargs.get("full_image")

        if model_results is not None:
            # ── Use pre-computed results (no re-inference) ──
            for name, output in model_results.items():
                if not isinstance(output, ModelOutput):
                    continue
                weight = self._model_weights.get(name, 0.0)
                if weight <= 0.0:
                    continue
                fake_prob = float(output.probabilities.get("fake", 0.5))
                model_outputs[name] = {
                    "prediction": output.prediction,
                    "confidence": output.confidence,
                    "fake_prob": fake_prob,
                    "real_prob": float(output.probabilities.get("real", 1.0 - fake_prob)),
                    "inference_time_ms": output.inference_time_ms,
                }
                if name == "forgery_detector" and isinstance(output.metadata, dict):
                    thr = output.metadata.get("decision_threshold")
                    if thr is not None:
                        model_outputs[name]["decision_threshold"] = float(thr)
                model_scores.append(fake_prob)
                model_weights.append(weight)
        else:
            # ── Legacy: run models (deprecated — kept for backward compat) ──
            for name, model in self._models.items():
                weight = self._model_weights.get(name, 0.0)
                if weight <= 0.0:
                    logger.debug(f"Skipping gated model: {name} (weight={weight})")
                    continue
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
                        "fake_prob": float(output.probabilities.get("fake", 0.5)),
                        "real_prob": float(output.probabilities.get("real", 0.5)),
                        "inference_time_ms": output.inference_time_ms,
                    }
                    model_scores.append(float(output.probabilities.get("fake", 0.5)))
                    model_weights.append(weight)

                except Exception as e:
                    logger.error(f"Model {name} failed: {e}")

        if not model_scores:
            return self._fallback_prediction()

        # ── Dual-domain soft OR (production path) ──
        # Complementary specialists: generative-AI vs face-swap forgery.
        # Soft OR = 1 - Π(1 - p_i)  — either domain can prove FAKE.
        # Never dampen toward REAL when AI-gen says "authentic photo".
        ai_out = model_outputs.get("ai_image_detector")
        forgery_out = model_outputs.get("forgery_detector")
        ai_gen_score = float(ai_out["fake_prob"]) if ai_out is not None else None
        face_forgery_score = (
            float(forgery_out["fake_prob"]) if forgery_out is not None else None
        )

        domain_scores = [
            s for s in (ai_gen_score, face_forgery_score) if s is not None
        ]
        if domain_scores:
            AI_FAKE_THRESHOLD = 0.50
            FORGERY_FAKE_THRESHOLD = 0.35
            if forgery_out is not None and "decision_threshold" in forgery_out:
                FORGERY_FAKE_THRESHOLD = float(forgery_out["decision_threshold"])

            ai_hit = ai_gen_score is not None and ai_gen_score >= AI_FAKE_THRESHOLD
            forgery_hit = (
                face_forgery_score is not None
                and face_forgery_score >= FORGERY_FAKE_THRESHOLD
            )

            if ai_hit or forgery_hit:
                # Soft-OR of the triggering specialists only
                active = []
                if ai_hit:
                    active.append(ai_gen_score)
                if forgery_hit:
                    active.append(face_forgery_score)
                survive_real = 1.0
                for s in active:
                    survive_real *= 1.0 - float(np.clip(s, 0.0, 1.0))
                final_score = float(max(1.0 - survive_real, max(active)))
            else:
                # Neither specialist confident → lean REAL using max soft signal
                # scaled below the decision boundary.
                peak = max(domain_scores)
                final_score = float(min(peak, 0.49))

            method = "domain_dual_threshold_or"
            weighted_avg = float(np.mean(domain_scores))
            learned_score = weighted_avg
        else:
            # Fallback: weighted average of whatever models remain
            model_scores_arr = np.array(model_scores)
            model_weights_arr = np.array(model_weights)
            model_weights_arr = model_weights_arr / model_weights_arr.sum()
            weighted_avg = float(np.dot(model_scores_arr, model_weights_arr))
            learned_score = weighted_avg
            if self._use_learned and self._ensemble_net is not None and len(model_scores) >= 2:
                try:
                    scores_tensor = torch.tensor(
                        [[1 - s, s] for s in model_scores], dtype=torch.float32
                    ).unsqueeze(0).to(self.device)
                    learned_score = self._ensemble_net(scores_tensor).item()
                except Exception:
                    pass
            final_score = (
                0.5 * weighted_avg + 0.5 * learned_score
                if self._use_learned
                else weighted_avg
            )
            method = "weighted_average" if not self._use_learned else "calibrated_ensemble"

        # ── Agreement ──
        predictions = [o["prediction"] for o in model_outputs.values()]
        agreement = max(
            predictions.count("real"),
            predictions.count("fake"),
        ) / len(predictions) if predictions else 0.0

        prediction = "fake" if final_score >= 0.5 else "real"
        class_probability = final_score if prediction == "fake" else (1.0 - final_score)
        confidence = float(np.clip(0.85 * class_probability + 0.15 * agreement, 0.01, 0.99))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {"real": 1.0 - final_score, "fake": final_score},
            "contributions": model_outputs,
            "agreement": float(agreement),
            "num_models": len(model_outputs),
            "ai_gen_score": ai_gen_score,
            "face_forgery_score": face_forgery_score if face_forgery_score is not None else 0.5,
            "decision_process": {
                "weighted_average": weighted_avg,
                "learned_score": learned_score,
                "final_score": final_score,
                "method": method,
                "domain_scores": {
                    "ai_gen": ai_gen_score,
                    "face_forgery": face_forgery_score,
                },
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
            "ai_gen_score": None,
            "face_forgery_score": 0.5,
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
