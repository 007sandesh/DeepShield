"""
Video Detection Pipeline.

End-to-end pipeline for analyzing video files:
Video → Frame Sampling → Face Extraction → Temporal Analysis → Ensemble → Result
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
from src.core.preprocessors.face_extraction import FaceExtractor
from src.core.preprocessors.frame_sampling import FrameSampler, SamplingStrategy
from src.core.preprocessors.audio_processing import AudioProcessor
from src.core.pipelines.image_pipeline import DetectionResult


class VideoDetectionPipeline:
    """
    Complete video analysis pipeline.

    Flow:
    1. Extract audio track
    2. Sample frames (adaptive/uniform)
    3. Extract faces from each frame
    4. Run temporal analysis across frames
    5. Run per-frame detection (frequency, forgery, attention)
    6. Analyze audio-visual sync
    7. Combine all signals via ensemble
    8. Generate per-frame timeline + overall result
    """

    def __init__(
        self,
        models: Optional[list[str]] = None,
        device: str = "auto",
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        max_frames: int = 32,
        target_fps: float = 10.0,
    ) -> None:
        self.face_extractor = FaceExtractor()
        self.frame_sampler = FrameSampler(
            strategy=sampling_strategy,
            target_fps=target_fps,
            max_frames=max_frames,
        )
        self.audio_processor = AudioProcessor()
        self._models: dict[str, BaseModel] = {}
        self._ensemble = None
        self._device = device
        self._initialized = False
        self._max_frames = max_frames

        self._model_names = models or [
            "forgery_detector",
            "frequency_analyzer",
            "attention_network",
            "temporal_analyzer",
            "audio_sync",
        ]

    def initialize(self) -> None:
        """Initialize all models."""
        if self._initialized:
            return

        logger.info("Initializing Video Detection Pipeline...")

        for name in self._model_names:
            try:
                model = ModelRegistry.create(name, device=self._device)
                self._models[name] = model
                logger.info(f"  ✓ {name}")
            except Exception as e:
                logger.warning(f"  ✗ {name}: {e}")

        try:
            self._ensemble = ModelRegistry.create("ensemble", device=self._device)
            for name, model in self._models.items():
                self._ensemble.register_model(name, model)
        except Exception as e:
            logger.warning(f"  ✗ ensemble: {e}")

        self._initialized = True
        logger.info(f"Video pipeline initialized with {len(self._models)} models")

    def detect(self, video_path: str) -> dict[str, Any]:
        """
        Analyze a video for deepfake detection.

        Args:
            video_path: Path to video file

        Returns:
            Comprehensive detection result with per-frame analysis
        """
        start_time = time.perf_counter()

        if not self._initialized:
            self.initialize()

        video_path = str(Path(video_path))
        logger.info(f"Analyzing video: {video_path}")

        # 1. Extract audio
        logger.info("Step 1/6: Extracting audio...")
        audio_data = self.audio_processor.extract_from_video(video_path)
        has_audio = audio_data is not None

        # 2. Sample frames
        logger.info("Step 2/6: Sampling frames...")
        frames, video_fps = self.frame_sampler.sample(video_path, self._max_frames)
        logger.info(f"  Sampled {len(frames)} frames @ {video_fps:.1f} FPS")

        # 3. Extract faces from each frame
        logger.info("Step 3/6: Extracting faces...")
        all_faces = []
        frame_face_map = []  # Track which face belongs to which frame

        for idx, frame in enumerate(frames):
            faces = self.face_extractor.extract_from_image(frame, max_faces=1)
            if faces:
                all_faces.append(faces[0])
                frame_face_map.append(idx)

        logger.info(f"  Extracted {len(all_faces)} faces from {len(frames)} frames")

        if not all_faces:
            logger.warning("No faces detected in video")
            return self._empty_result(video_path)

        # 4. Temporal analysis (across all face frames)
        logger.info("Step 4/6: Running temporal analysis...")
        temporal_result = None
        if "temporal_analyzer" in self._models and len(all_faces) > 2:
            face_frames = np.stack([f.image for f in all_faces])
            try:
                temporal_result = self._models["temporal_analyzer"].predict(
                    face_frames, fps=video_fps
                )
                logger.info(
                    f"  Temporal: {temporal_result.prediction} "
                    f"({temporal_result.confidence:.2f})"
                )
            except Exception as e:
                logger.error(f"  Temporal analysis failed: {e}")

        # 5. Per-frame analysis
        logger.info("Step 5/6: Running per-frame analysis...")
        frame_results = []
        model_times_accum = {}

        for idx, face in enumerate(all_faces):
            frame_result = {"frame_idx": frame_face_map[idx], "models": {}}

            for name, model in self._models.items():
                if name in ("temporal_analyzer", "audio_sync"):
                    continue  # Skip temporal-only models for per-frame

                try:
                    output = model.predict(face.image)
                    frame_result["models"][name] = {
                        "prediction": output.prediction,
                        "confidence": output.confidence,
                        "probabilities": output.probabilities,
                        "time_ms": output.inference_time_ms,
                    }

                    # Accumulate timing
                    if name not in model_times_accum:
                        model_times_accum[name] = 0.0
                    model_times_accum[name] += output.inference_time_ms

                except Exception as e:
                    frame_result["models"][name] = {"error": str(e)}

            frame_results.append(frame_result)

        # 6. Audio-visual sync analysis
        logger.info("Step 6/6: Analyzing audio-visual sync...")
        audio_sync_result = None
        if has_audio and "audio_sync" in self._models:
            try:
                # Use middle frames for audio sync
                mid_start = len(all_faces) // 3
                mid_end = 2 * len(all_faces) // 3
                mid_frames = np.stack([f.image for f in all_faces[mid_start:mid_end]])

                audio_sync_result = self._models["audio_sync"].predict(
                    mid_frames,
                    audio_path=str(audio_data.waveform),  # Simplified
                    fps=video_fps,
                )
                logger.info(
                    f"  Audio sync: {audio_sync_result.prediction} "
                    f"({audio_sync_result.confidence:.2f})"
                )
            except Exception as e:
                logger.error(f"  Audio sync failed: {e}")

        # Combine results
        final_result = self._combine_results(
            frame_results=frame_results,
            temporal_result=temporal_result,
            audio_sync_result=audio_sync_result,
            total_frames=len(frames),
            analyzed_frames=len(all_faces),
            video_fps=video_fps,
            has_audio=has_audio,
        )

        total_time = (time.perf_counter() - start_time) * 1000
        final_result["performance"]["total_time_ms"] = round(total_time, 2)
        final_result["performance"]["model_times_ms"] = {
            k: round(v, 2) for k, v in model_times_accum.items()
        }

        logger.info(
            f"Video analysis complete: {final_result['prediction']} "
            f"({final_result['confidence']:.2f}) in {total_time:.0f}ms"
        )

        return final_result

    def _combine_results(
        self,
        frame_results: list[dict],
        temporal_result: Optional[ModelOutput],
        audio_sync_result: Optional[ModelOutput],
        total_frames: int,
        analyzed_frames: int,
        video_fps: float,
        has_audio: bool,
    ) -> dict[str, Any]:
        """Combine all analysis results into final prediction."""

        # Aggregate per-frame predictions
        frame_fake_probs = []
        for fr in frame_results:
            for model_name, model_result in fr.get("models", {}).items():
                if "probabilities" in model_result:
                    frame_fake_probs.append(model_result["probabilities"].get("fake", 0.5))

        # Per-frame average
        avg_frame_prob = np.mean(frame_fake_probs) if frame_fake_probs else 0.5

        # Temporal score
        temporal_prob = 0.5
        temporal_meta = {}
        if temporal_result:
            temporal_prob = temporal_result.probabilities.get("fake", 0.5)
            temporal_meta = temporal_result.metadata

        # Audio sync score
        audio_prob = 0.5
        audio_meta = {}
        if audio_sync_result:
            audio_prob = audio_sync_result.probabilities.get("fake", 0.5)
            audio_meta = audio_sync_result.metadata

        # Weighted combination
        if has_audio:
            final_prob = (
                0.35 * avg_frame_prob +
                0.35 * temporal_prob +
                0.30 * audio_prob
            )
        else:
            final_prob = (
                0.55 * avg_frame_prob +
                0.45 * temporal_prob
            )

        prediction = "FAKE" if final_prob > 0.5 else "REAL"
        confidence = max(final_prob, 1 - final_prob)

        # Generate timeline
        timeline = []
        for fr in frame_results:
            frame_fake = []
            for model_result in fr.get("models", {}).values():
                if "probabilities" in model_result:
                    frame_fake.append(model_result["probabilities"].get("fake", 0.5))

            frame_prob = np.mean(frame_fake) if frame_fake else 0.5
            timeline.append({
                "frame": fr["frame_idx"],
                "timestamp": round(fr["frame_idx"] / video_fps, 2),
                "fake_probability": round(float(frame_prob), 4),
                "label": "FAKE" if frame_prob > 0.5 else "REAL",
            })

        # Identify suspicious segments
        suspicious_segments = self._find_suspicious_segments(timeline)

        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "probability": {
                "real": round(1 - final_prob, 4),
                "fake": round(final_prob, 4),
            },
            "video_info": {
                "total_frames": total_frames,
                "analyzed_frames": analyzed_frames,
                "fps": video_fps,
                "duration_seconds": round(total_frames / video_fps, 2),
                "has_audio": has_audio,
            },
            "temporal_analysis": temporal_meta,
            "audio_analysis": audio_meta,
            "timeline": timeline,
            "suspicious_segments": suspicious_segments,
            "model_agreement": self._compute_agreement(frame_results),
            "performance": {},
        }

    def _find_suspicious_segments(
        self, timeline: list[dict], threshold: float = 0.6
    ) -> list[dict]:
        """Find contiguous suspicious segments."""
        segments = []
        current_segment = None

        for entry in timeline:
            if entry["fake_probability"] > threshold:
                if current_segment is None:
                    current_segment = {
                        "start_frame": entry["frame"],
                        "start_time": entry["timestamp"],
                        "end_frame": entry["frame"],
                        "end_time": entry["timestamp"],
                        "max_probability": entry["fake_probability"],
                    }
                else:
                    current_segment["end_frame"] = entry["frame"]
                    current_segment["end_time"] = entry["timestamp"]
                    current_segment["max_probability"] = max(
                        current_segment["max_probability"],
                        entry["fake_probability"],
                    )
            else:
                if current_segment is not None:
                    current_segment["duration"] = round(
                        current_segment["end_time"] - current_segment["start_time"], 2
                    )
                    if current_segment["duration"] > 0.1:
                        segments.append(current_segment)
                    current_segment = None

        if current_segment is not None:
            current_segment["duration"] = round(
                current_segment["end_time"] - current_segment["start_time"], 2
            )
            segments.append(current_segment)

        return segments

    def _compute_agreement(self, frame_results: list[dict]) -> dict[str, Any]:
        """Compute model agreement statistics."""
        agreement_stats = {}

        for fr in frame_results:
            for model_name, model_result in fr.get("models", {}).items():
                if model_name not in agreement_stats:
                    agreement_stats[model_name] = {"real": 0, "fake": 0, "errors": 0}

                if "prediction" in model_result:
                    pred = model_result["prediction"].upper()
                    if pred in agreement_stats[model_name]:
                        agreement_stats[model_name][pred] += 1
                else:
                    agreement_stats[model_name]["errors"] += 1

        return agreement_stats

    def _empty_result(self, video_path: str) -> dict[str, Any]:
        """Return empty result when no faces are detected."""
        return {
            "prediction": "UNCERTAIN",
            "confidence": 0.0,
            "error": "No faces detected in video",
            "video_path": video_path,
        }
