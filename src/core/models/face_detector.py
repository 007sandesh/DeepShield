"""
Face Detection and Alignment module.

Uses RetinaFace (primary) or MTCNN (fallback) for robust face detection
with landmark alignment and quality scoring.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelType, ModelRegistry


@ModelRegistry.register
class FaceDetector(BaseModel):
    """
    Multi-backend face detector with alignment.

    Features:
    - RetinaFace for production accuracy
    - MTCNN as lightweight fallback
    - 5-point landmark alignment
    - Face quality scoring
    - Batch processing support
    """

    model_type = ModelType.FACE_DETECTION
    model_name = "face_detector"

    def __init__(
        self,
        backend: str = "retinaface",
        device: str = "auto",
        min_confidence: float = 0.9,
        target_size: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__(device=device)
        self.backend = backend
        self.min_confidence = min_confidence
        self.target_size = target_size
        self._detector = None
        self.load_weights()

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load face detection model."""
        import time
        start = time.perf_counter()

        if self.backend == "retinaface":
            from retinaface import RetinaFace as RF
            self._detector = RF
            logger.info("Loaded RetinaFace detector")

        elif self.backend == "mtcnn":
            from facenet_pytorch import MTCNN
            self._detector = MTCNN(
                keep_all=True,
                device=str(self.device),
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                post_process=True,
            )
            logger.info("Loaded MTCNN detector")

        elif self.backend == "mediapipe":
            import mediapipe as mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.min_confidence,
            )
            logger.info("Loaded MediaPipe detector")

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self._is_loaded = True
        self._load_time_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Face detector loaded in {self._load_time_ms:.1f}ms")

    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Detect faces in image.

        Args:
            input_data: BGR image (H, W, C) as numpy array
            **kwargs:
                align: bool = True — Apply alignment after detection

        Returns:
            ModelOutput with detected faces in metadata
        """
        align = kwargs.get("align", True)

        result, elapsed = self._measure_inference(
            self._detect_faces, input_data, align
        )

        faces = result["faces"]
        num_faces = len(faces)

        return ModelOutput(
            prediction="detected" if num_faces > 0 else "no_face",
            confidence=max((f["confidence"] for f in faces), default=0.0),
            metadata={
                "num_faces": num_faces,
                "faces": faces,
                "image_shape": input_data.shape,
            },
            inference_time_ms=elapsed,
        )

    def _detect_faces(
        self, image: np.ndarray, align: bool
    ) -> dict[str, Any]:
        """Internal face detection logic."""
        faces = []

        if self.backend == "retinaface":
            import tempfile
            import os

            # RetinaFace expects file path or RGB array
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self._detector.detect_faces(rgb)

            if detections:
                for det in detections:
                    if det["score"] < self.min_confidence:
                        continue

                    bbox = det["facial_area"]
                    landmarks = det["landmarks"]

                    face_img = self._extract_face(image, bbox, align, landmarks)
                    quality = self._assess_quality(face_img)

                    faces.append({
                        "bbox": bbox,
                        "confidence": det["score"],
                        "landmarks": landmarks,
                        "face_image": face_img,
                        "quality_score": quality,
                        "aligned": align,
                    })

        elif self.backend == "mtcnn":
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            boxes, probs, landmarks = self._detector(tensor)

            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes[0], probs[0])):
                    if prob < self.min_confidence:
                        continue

                    bbox = box.cpu().numpy().astype(int).tolist()
                    face_img = self._extract_face(image, bbox, align)
                    quality = self._assess_quality(face_img)

                    faces.append({
                        "bbox": bbox,
                        "confidence": float(prob),
                        "landmarks": landmarks[0][i].cpu().numpy().tolist() if landmarks else None,
                        "face_image": face_img,
                        "quality_score": quality,
                        "aligned": align,
                    })

        return {"faces": faces}

    def _extract_face(
        self,
        image: np.ndarray,
        bbox: list[int],
        align: bool,
        landmarks: Optional[dict] = None,
    ) -> np.ndarray:
        """Extract and optionally align face from image."""
        h, w = image.shape[:2]

        # Add padding
        pad_w = int((bbox[2] - bbox[0]) * 0.1)
        pad_h = int((bbox[3] - bbox[1]) * 0.1)
        x1 = max(0, bbox[0] - pad_w)
        y1 = max(0, bbox[1] - pad_h)
        x2 = min(w, bbox[2] + pad_w)
        y2 = min(h, bbox[3] + pad_h)

        face = image[y1:y2, x1:x2]

        if align and landmarks:
            face = self._align_face(image, landmarks, target_size=self.target_size)

        # Resize to target size
        face = cv2.resize(face, self.target_size, interpolation=cv2.INTER_LANCZOS4)

        return face

    def _align_face(
        self,
        image: np.ndarray,
        landmarks: dict,
        target_size: tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """Align face using eye landmarks for rotation normalization."""
        if not landmarks or "left_eye" not in landmarks or "right_eye" not in landmarks:
            return image

        left_eye = np.array(landmarks["left_eye"])
        right_eye = np.array(landmarks["right_eye"])

        # Calculate angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate center between eyes
        center = ((left_eye + right_eye) / 2).astype(int)

        # Rotation matrix
        M = cv2.getRotationMatrix2D(
            (center[0], center[1]), angle, scale=1.0
        )

        # Apply rotation
        aligned = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return aligned

    def _assess_quality(self, face: np.ndarray) -> float:
        """
        Assess face quality score (0.0 - 1.0).

        Factors: brightness, contrast, blur, size.
        """
        if face.size == 0:
            return 0.0

        # Brightness (good range: 80-180)
        brightness = np.mean(face)
        brightness_score = 1.0 - abs(brightness - 130) / 130

        # Contrast (higher is better)
        contrast = np.std(face)
        contrast_score = min(contrast / 50.0, 1.0)

        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        blur_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)

        # Weighted combination
        quality = (
            0.3 * brightness_score +
            0.3 * contrast_score +
            0.4 * blur_score
        )

        return float(np.clip(quality, 0.0, 1.0))

    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Generate face detection visualization."""
        output = self.predict(input_data, **kwargs)
        vis = input_data.copy()

        for face in output.metadata.get("faces", []):
            bbox = face["bbox"]
            conf = face["confidence"]
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(
                vis, f"{conf:.2f}", (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        return {
            "visualization": vis,
            "faces": output.metadata["faces"],
            "num_faces": output.metadata["num_faces"],
        }
