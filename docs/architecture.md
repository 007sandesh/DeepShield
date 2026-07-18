# DeepShield Architecture Document 🏗️

## Overview

DeepShield is a **multi-modal deepfake detection platform** built with a layered architecture that combines signal processing, computer vision, and deep learning to achieve state-of-the-art detection accuracy.

---

## 🧠 System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │  Web UI  │  │    CLI   │  │  Python  │  │  External Apps    │   │
│  │ (Next.js)│  │  (Typer) │  │   SDK    │  │  (REST/WebSocket)  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬──────────┘   │
└───────┼──────────────┼──────────────┼────────────────┼──────────────┘
        │              │              │                │
┌───────┴──────────────┴──────────────┴────────────────┴──────────────┐
│                      API LAYER (FastAPI)                             │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │   Routes   │  │ Middleware  │  │ WebSocket  │  │  Rate Limit  │  │
│  │ /detect/*  │  │ Auth, CORS │  │  /ws/*     │  │  Retry/Pool  │  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                      ORCHESTRATION LAYER                             │
│                                                                      │
│  ┌────────────────────┐  ┌────────────────────┐                     │
│  │ ImagePipeline      │  │ VideoPipeline      │                     │
│  │                    │  │                    │                     │
│  │ • Face extraction  │  │ • Frame sampling   │                     │
│  │ • Multi-model      │  │ • Audio extraction  │                     │
│  │ • Ensemble fusion  │  │ • Temporal analysis  │                     │
│  │ • Explainability   │  │ • AV sync analysis  │                     │
│  └────────┬───────────┘  └────────┬───────────┘                     │
└───────────┼────────────────────────┼────────────────────────────────┘
            │                        │
┌───────────┴────────────────────────┴────────────────────────────────┐
│                      DETECTION ENGINE                                 │
│                                                                      │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │  Forgery   │ │Frequency │ │ Temporal │ │Biological│ │  Audio  │ │
│  │ Detector   │ │Analyzer  │ │ Analyzer │ │ Signals  │ │  Sync   │ │
│  │ (CNNs)     │ │(FFT/DCT) │ │  (LSTM)  │ │ (rPPG)   │ │(Whisper)│ │
│  └─────┬──────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘ │
│        └──────────────┴────────────┴────────────┴────────────┘      │
│                                  │                                   │
│  ┌───────────────────────────────┴───────────────────────────────┐  │
│  │                  Ensemble Meta-Classifier                      │  │
│  │  • Learned attention weights  • Temperature calibration       │  │
│  └───────────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                            │
│                                                                      │
│  ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ PostgreSQL│  │   Redis    │  │ Docker   │  │ Prometheus +   │  │
│  │ (Results) │  │  (Cache)   │  │ (K8s)    │  │ Grafana (Obs)  │  │
│  └───────────┘  └────────────┘  └──────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📡 Data Flow

### Image Detection Flow

```
Image Upload
    │
    ▼
┌──────────────┐
│  Validate    │  ← File type, size, integrity checks
│  & Load      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Face        │  ← Detect faces, select best quality
│  Extraction  │       Align, crop to 256x256
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────┐
│  Parallel Model Execution          │
│                                    │
│  ┌─────────┐  ┌──────────┐         │
│  │Forgery  │  │Frequency │         │
│  │Detector │  │Analyzer  │         │
│  └────┬────┘  └────┬─────┘         │
│       │             │               │
│  ┌────┴────┐  ┌────┴─────┐         │
│  │Attention│  │Biological│         │
│  │Network  │  │ Signals  │         │
│  └─────────┘  └──────────┘         │
└────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  Ensemble    │  ← Weighted combination
│  Fusion      │       Confidence calibration
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Explain     │  ← SHAP, GradCAM, heatmaps
│  & Return    │       Natural language explanation
└──────────────┘
```

### Video Detection Flow

```
Video Upload
    │
    ├────────────┐
    │            │
    ▼            ▼
┌──────────┐ ┌──────────┐
│ Frame    │ │ Audio    │
│ Sampling │ │ Extract  │
│ (32 fps) │ │ (ffmpeg) │
└────┬─────┘ └────┬─────┘
     │            │
     ▼            │
┌──────────┐      │
│ Face     │      │
│ Extract  │      │
│ (per fr) │      │
└────┬─────┘      │
     │            │
     ▼            ▼
┌─────────────────────────┐
│  Temporal Analysis      │ ← LSTM across face sequence
│  Frequency Analysis     │ ← Per frame
│  Attention Analysis     │ ← Per frame
│  Audio Sync Analysis    │ ← Cross-modal matching
└──────────┬──────────────┘
           │
           ▼
┌──────────────────┐
│  Ensemble +      │
│  Timeline        │ ← Per-frame confidence scores
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Results +       │ ← Suspicious segments
│  Export          │     Overall verdict
└──────────────────┘
```

---

## 💾 Data Schema

### API Request/Response Models

```python
# ─── Image Detection ───
POST /detect/image
Request:  multipart/form-data (file: image)
Response: {
    "prediction": "REAL" | "FAKE",
    "confidence": 0.9721,
    "probability": {"real": 0.0279, "fake": 0.9721},
    "faces_detected": 1,
    "best_face_quality": 0.89,
    "explanation": "Multiple detection layers indicate AI generation...",
    "model_results": {
        "forgery_detector": {
            "prediction": "fake",
            "confidence": 0.98,
            "inference_time_ms": 12.3
        },
        "frequency_analyzer": {
            "prediction": "fake",
            "confidence": 0.87,
            "metadata": {
                "fft_score": 0.82,
                "dct_score": 0.65,
                "band_energy": {...}
            },
            "inference_time_ms": 8.1
        },
        "attention_network": {
            "prediction": "fake",
            "confidence": 0.95,
            "inference_time_ms": 18.2
        },
        "biological_signals": {
            "prediction": "real",
            "confidence": 0.56,
            "inference_time_ms": 21.5
        }
    },
    "performance": {
        "total_time_ms": 72.1,
        "model_times_ms": {...}
    },
    "heatmap": "...base64_encoded_heatmap..."
}
```

### Database Schema

```sql
-- Detection Results
CREATE TABLE detection_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    file_hash VARCHAR(64),
    file_name VARCHAR(512),
    file_type VARCHAR(32),  -- 'image' or 'video'
    file_size_bytes BIGINT,
    
    prediction VARCHAR(16),
    confidence FLOAT,
    
    model_results JSONB,      -- Full per-model breakdown
    metadata JSONB,           -- Input metadata
    
    inference_time_ms FLOAT,
    explainability JSONB,     -- Optional heatmap data (base64)
    
    INDEX idx_created_at (created_at),
    INDEX idx_prediction (prediction)
);

-- Analysis History
CREATE TABLE analysis_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    detection_id UUID REFERENCES detection_results(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    feedback VARCHAR(32),      -- 'correct', 'incorrect', 'unsure'
    
    INDEX idx_user_id (user_id)
);

-- Batch Jobs
CREATE TABLE batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(32) DEFAULT 'pending',  -- pending, running, completed, failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    total_files INT,
    completed_files INT DEFAULT 0,
    results JSONB
);
```

### Model Weight Schema

```
models/weights/
├── xception_ffpp.pt              [85MB]   XceptionNet on FF++
├── efficientnet_celebdf.pt       [45MB]   EfficientNet-B4 on Celeb-DF
├── shape_predictor_68.dat        [100MB]  dlib landmarks
├── vit_face_forensics.pt         [330MB]  ViT base weights
├── whisper_base.pt               [142MB]  Whisper base
└── model_registry.json           [~1KB]   Manifest
```

---

## 🔒 Security Architecture

### Rate Limiting
```yaml
Detection API:   100 requests/minute/user
Batch API:       10 requests/minute/user
WebSocket:       60 messages/minute/connection
```

### Authentication
- API Key via `X-API-Key` header
- JWT-based for web UI
- Rate limiting per IP as fallback

### Input Validation
- File type whitelist: jpg, png, webp, mp4, avi, mov
- Max file size: 100MB (image), 500MB (video)
- Virus scanning for uploaded content
- Exif data stripping

---

## 📈 Monitoring & Observability

### Metrics (Prometheus)
```
deepshield_detections_total{type, prediction}
deepshield_detection_duration_ms{type, model}
deepshield_model_errors_total{model}
deepshield_api_requests_total{endpoint, status}
deepshield_active_connections
```

### Dashboards (Grafana)
1. **Detection Overview** — Total detections, accuracy, latency
2. **Model Performance** — Per-model latency, error rates, throughput
3. **System Health** — CPU, GPU, memory, disk usage
4. **User Analytics** — Usage patterns, popular endpoints

### Logging (Structured JSON)
```json
{
    "timestamp": "2024-01-15T12:00:00Z",
    "level": "INFO",
    "event": "detection_complete",
    "detection_id": "uuid",
    "file_type": "image",
    "file_size": 245760,
    "models_used": 4,
    "prediction": "FAKE",
    "confidence": 0.97,
    "inference_time_ms": 72.1,
    "total_time_ms": 85.3
}
```

---

## 🚀 Deployment Architecture

### Single Server (Dev/Small)
```
[Client] → [FastAPI + Models] → [SQLite]
```

### Production
```
                    ┌───────────┐
                    │  Nginx    │  ← SSL termination, rate limit, static files
                    │  LB       │
                    └─────┬─────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
        ┌─────┴────┐ ┌─────┴────┐ ┌─────┴────┐
        │ FastAPI  │ │ FastAPI  │ │ FastAPI  │  ← Horizontally scaled
        │ Worker   │ │ Worker   │ │ Worker   │
        │ (GPU)    │ │ (GPU)    │ │ (GPU)    │
        └──────────┘ └──────────┘ └──────────┘
              │           │           │
              └───────────┼───────────┘
                          │
                    ┌─────┴─────┐
                    │  Redis    │  ← Session cache, rate limit
                    │  Cluster  │
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │PostgreSQL │  ← Persistent results
                    │  + TimescaleDB
                    └───────────┘
```

---

## 🧪 Testing Strategy

| Layer | Tests | Tools |
|-------|-------|-------|
| **Unit** | Model initialization, preprocessing, inference | pytest, pytest-cov |
| **Integration** | Full pipeline end-to-end, API endpoints | pytest, httpx, docker |
| **Performance** | Latency benchmarks, memory profiling | pytest-benchmark, memory-profiler |
| **Robustness** | Adversarial attacks, edge cases | Custom test suite |
| **Regression** | Model output consistency | Snapshot testing |
