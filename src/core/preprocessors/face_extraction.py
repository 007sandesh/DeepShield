"""
Face Extraction and Preprocessing.

Handles face detection, alignment, cropping, and normalization
for downstream analysis models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger


@dataclass
class ExtractedFace:
    """Container for an extracted face with metadata."""

    image: np.ndarray  # Aligned face image (H, W, 3)
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[dict[str, tuple[int, int]]] = None
    quality_score: float = 0.0
    source_frame_idx: int = 0

    @property
    def is_high_quality(self) -> bool:
        return self.quality_score > 0.7


class FaceExtractor:
    """
    Production-grade face extraction pipeline.

    Steps:
    1. Detect faces in frame
    2. Filter by confidence and quality
    3. Align using facial landmarks
    4. Crop and resize to target dimensions
    5. Normalize pixel values
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        min_confidence: float = 0.9,
        min_quality: float = 0.5,
        padding_ratio: float = 0.2,
    ) -> None:
        self.target_size = target_size
        self.min_confidence = min_confidence
        self.min_quality = min_quality
        self.padding_ratio = padding_ratio
        self._detector = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize face detector."""
        try:
            from retinaface import RetinaFace
            self._detector = "retinaface"
            logger.info("FaceExtractor: Using RetinaFace")
        except ImportError:
            logger.warning("RetinaFace not available, using OpenCV Haar Cascade")
            self._detector = "opencv"
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def extract_from_image(
        self, image: np.ndarray, max_faces: int = 5
    ) -> list[ExtractedFace]:
        """
        Extract all faces from a single image.

        Args:
            image: BGR image (H, W, 3)
            max_faces: Maximum faces to extract

        Returns:
            List of ExtractedFace objects, sorted by confidence
        """
        faces = []

        if self._detector == "retinaface":
            faces = self._extract_retinaface(image, max_faces)
        else:
            faces = self._extract_opencv(image, max_faces)

        # Sort by confidence (best first)
        faces.sort(key=lambda f: f.confidence, reverse=True)

        return faces[:max_faces]

    def extract_from_video(
        self,
        video_path: str,
        max_frames: int = 32,
        target_fps: float = 10.0,
    ) -> list[ExtractedFace]:
        """
        Extract best faces from video.

        Samples frames at target_fps, extracts faces,
        and returns the highest quality faces.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / target_fps))

        all_faces: list[ExtractedFace] = []
        frame_idx = 0

        while cap.isOpened() and len(all_faces) < max_frames * 3:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                faces = self.extract_from_image(frame, max_faces=3)
                for face in faces:
                    face.source_frame_idx = frame_idx
                all_faces.extend(faces)

            frame_idx += 1

        cap.release()

        # Sort by quality and return best faces
        all_faces.sort(key=lambda f: f.quality_score, reverse=True)

        logger.info(
            f"Extracted {len(all_faces)} faces from {total_frames} frames "
            f"(keeping top {max_frames})"
        )

        return all_faces[:max_frames]

    def _extract_retinaface(
        self, image: np.ndarray, max_faces: int
    ) -> list[ExtractedFace]:
        """Extract faces using RetinaFace."""
        from retinaface import RetinaFace

        faces = []
        detections = RetinaFace.detect_faces(image)

        if not isinstance(detections, dict):
            return faces

        for key, det in detections.items():
            if det["score"] < self.min_confidence:
                continue

            bbox = tuple(det["facial_area"])
            landmarks = det.get("landmarks", {})

            # Extract and align face
            face_img = self._crop_and_align(image, bbox, landmarks)
            quality = self._assess_quality(face_img)

            if quality < self.min_quality:
                continue

            faces.append(ExtractedFace(
                image=face_img,
                bbox=bbox,
                confidence=det["score"],
                landmarks=landmarks,
                quality_score=quality,
            ))

            if len(faces) >= max_faces:
                break

        return faces

    def _extract_opencv(
        self, image: np.ndarray, max_faces: int
    ) -> list[ExtractedFace]:
        """Fallback face extraction using OpenCV."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self._cascade.detectMultiScale(gray, 1.3, 5)

        faces = []
        for (x, y, w, h) in detections:
            bbox = (x, y, x + w, y + h)
            face_img = self._crop_and_align(image, bbox)
            quality = self._assess_quality(face_img)

            if quality < self.min_quality:
                continue

            faces.append(ExtractedFace(
                image=face_img,
                bbox=bbox,
                confidence=0.9,  # OpenCV doesn't provide confidence
                quality_score=quality,
            ))

            if len(faces) >= max_faces:
                break

        return faces

    def _crop_and_align(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        landmarks: Optional[dict] = None,
    ) -> np.ndarray:
        """Crop face with padding and optional alignment."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add padding
        pad_w = int((x2 - x1) * self.padding_ratio)
        pad_h = int((y2 - y1) * self.padding_ratio)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        face = image[y1:y2, x1:x2]

        # Align if landmarks available
        if landmarks and "left_eye" in landmarks and "right_eye" in landmarks:
            face = self._align_face(image, landmarks)

        # Resize to target
        face = cv2.resize(face, self.target_size, interpolation=cv2.INTER_LANCZOS4)

        return face

    def _align_face(
        self, image: np.ndarray, landmarks: dict
    ) -> np.ndarray:
        """Align face based on eye positions."""
        left_eye = np.array(landmarks["left_eye"])
        right_eye = np.array(landmarks["right_eye"])

        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Center between eyes
        center = ((left_eye + right_eye) / 2)
        center_pt = (float(center[0]), float(center[1]))

        # Rotate
        M = cv2.getRotationMatrix2D(center_pt, float(angle), 1.0)
        aligned = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return aligned

    def _assess_quality(self, face: np.ndarray) -> float:
        """Assess face quality (0.0 - 1.0)."""
        if face.size == 0:
            return 0.0

        # Brightness
        brightness = np.mean(face)
        brightness_ok = 1.0 - abs(brightness - 128) / 128

        # Contrast
        contrast = np.std(face) / 64.0

        # Blur (Laplacian variance)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(blur / 300.0, 1.0)

        # Size check
        h, w = face.shape[:2]
        size_score = min(h, w) / 256.0

        quality = (
            0.25 * brightness_ok +
            0.25 * blur_score +
            0.25 * min(contrast, 1.0) +
            0.25 * size_score
        )

        return float(np.clip(quality, 0.0, 1.0))
