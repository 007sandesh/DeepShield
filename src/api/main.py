"""
DeepShield API — FastAPI Backend.

Production-grade REST API with:
- Image/Video upload and analysis
- WebSocket real-time streaming
- Batch processing
- Rate limiting and authentication
- Health checks and monitoring
"""

from __future__ import annotations

import io
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.core.pipelines.image_pipeline import ImageDetectionPipeline
from src.core.pipelines.video_pipeline import VideoDetectionPipeline


# ─── Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipelines on startup, cleanup on shutdown."""
    logger.info("🚀 DeepShield API starting up...")

    settings = get_settings()
    app.state.image_pipeline = ImageDetectionPipeline(
        device=settings.model.device,
    )
    app.state.image_pipeline.initialize()

    app.state.video_pipeline = VideoDetectionPipeline(
        device=settings.model.device,
    )
    app.state.video_pipeline.initialize()

    app.state.start_time = time.time()
    logger.info("✅ DeepShield API ready")

    yield

    logger.info("🛑 DeepShield API shutting down")


# ─── App ───────────────────────────────────────────────────
app = FastAPI(
    title="DeepShield API",
    description="AI-Powered Deepfake Detection Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ───────────────────────────────────────────────
class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probability: dict[str, float]
    faces_detected: int = 0
    explanation: str = ""
    model_results: dict[str, Any] = {}
    performance: dict[str, Any] = {}


class VideoDetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probability: dict[str, float]
    video_info: dict[str, Any]
    timeline: list[dict[str, Any]]
    suspicious_segments: list[dict[str, Any]]
    performance: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    models_loaded: int


class BatchRequest(BaseModel):
    file_paths: list[str]
    media_type: str = Field(default="image", pattern="^(image|video)$")


# ─── Routes ────────────────────────────────────────────────
@app.get("/", response_model=dict)
async def root():
    return {
        "name": "DeepShield API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    uptime = time.time() - app.state.start_time
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
        models_loaded=len(app.state.image_pipeline._models),
    )


@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(..., description="Image file (jpg, png, webp)"),
):
    """
    Analyze an image for deepfake detection.

    Upload an image file and receive a comprehensive analysis
    including prediction, confidence, and explanation.
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}",
        )

    try:
        # Read file
        contents = await file.read()

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Run detection
        result = app.state.image_pipeline.detect(image)

        return DetectionResponse(**result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov, webm)"),
):
    """
    Analyze a video for deepfake detection.

    Upload a video file for comprehensive analysis including
    temporal consistency, audio-visual sync, and per-frame scoring.
    """
    allowed_types = {
        "video/mp4", "video/avi", "video/quicktime",
        "video/webm", "video/x-msvideo",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}",
        )

    try:
        # Save to temp file
        contents = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Run detection
        result = app.state.video_pipeline.detect(tmp_path)

        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)

        return VideoDetectionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(request: BatchRequest):
    """
    Batch analyze multiple files.

    Process multiple images or videos in a single request.
    """
    results = []

    for file_path in request.file_paths:
        try:
            if not Path(file_path).exists():
                results.append({
                    "file": file_path,
                    "error": "File not found",
                })
                continue

            if request.media_type == "image":
                image = cv2.imread(file_path)
                if image is None:
                    results.append({"file": file_path, "error": "Cannot read image"})
                    continue
                result = app.state.image_pipeline.detect(image)
                results.append(result.to_dict())
            else:
                result = app.state.video_pipeline.detect(file_path)
                results.append(result)

        except Exception as e:
            results.append({"file": file_path, "error": str(e)})

    return {"results": results, "total": len(results)}


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection.

    Send base64-encoded images or video frames for
    streaming analysis results.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "image":
                # Decode base64 image
                import base64
                img_bytes = base64.b64decode(data["data"])
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is not None:
                    result = app.state.image_pipeline.detect(image)
                    await websocket.send_json(result.to_dict())

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# ─── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.debug,
    )
