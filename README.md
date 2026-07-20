<div align="center">

# 🛡️ DeepShield

### Reliable Deepfake Detection You Can Run Locally

**Multi-model face forgery analysis with clear explanations — built for researchers, builders, and anyone who wants to verify what they see.**

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge)

<br>

<div align="center">
<table>
<tr>
<td>
<pre>
 _____                  _____ _     _      _     _
|  __ \                / ____| |   (_)    | |   | |
| |  | | ___  ___ _ __| (___ | |__  _  ___| | __| |
| |  | |/ _ \/ _ \ '_ \\___ \| '_ \| |/ _ \ |/ _` |
| |__| |  __/  __/ |_) |___) | | | | |  __/ | (_| |
|_____/ \___|\___| .__/_____/|_| |_|_|\___|_|\__,_|
                 | |
                 |_|
</pre>
</td>
</tr>
</table>
</div>

[![Focus](https://img.shields.io/badge/Focus-Face%20Deepfakes%20%2B%20AI%20Gen-blueviolet?style=for-the-badge)](#-what-deepshield-detects-today)
[![Models](https://img.shields.io/badge/Models-EfficientNet%20%2B%20AI%20Detector%20%2B%20Ensemble-success?style=for-the-badge)](#-how-it-works)
[![CLI](https://img.shields.io/badge/Interface-CLI%20%2B%20API-orange?style=for-the-badge)](#-quick-start)

</div>

---

## Why DeepShield?

Deepfakes are getting better — DeepShield gives you a practical way to check images for **face manipulation, forgery, and AI-generated content**. Multiple specialized models examine the same media, then an ensemble combines their signals into one clear verdict: **REAL** or **FAKE**, with confidence and a short explanation.

| Strength | What you get |
|----------|----------------|
| **Face deepfakes** | EfficientNet-B4 weights trained on Celeb-DF / FaceForensics-style face-swap data |
| **AI-generated images** | HuggingFace `Organika/sdxl-detector` classifies synthetic vs authentic stills |
| **Multi-signal analysis** | Spatial forgery, frequency cues, and AI-gen scoring — not a single black-box score |
| **Explainable output** | Human-readable explanations and per-model breakdowns |
| **Easy to run** | One CLI command after downloading weights |
| **Open & extensible** | MIT license, FastAPI surface, Docker-friendly layout |

---

## What DeepShield Detects Today

DeepShield handles **two detection domains** with dedicated backends:

### 1. Face-Swap / Deepfake Still Detection (face crop)
Face-centric deepfakes — swaps, reenactment, and related facial forgery patterns:
- EfficientNet-B4 loaded with Celeb-DF / FaceForensics-trained checkpoint
- Frequency analysis for spectral artifacts

### 2. AI-Generated Image Detection (full frame)
Fully synthetic stills from generative-AI models:
- HuggingFace `Organika/sdxl-detector` — classifies ChatGPT Images, Midjourney, Stable Diffusion / SDXL, and similar generators

**Both scores are surfaced in every result** — an image can be flagged as AI-generated even if the face crop looks authentic, and vice versa.

---

## Key Capabilities

<table>
<tr>
<td width="50%">

### Multi-Layer Detection
Independent analyzers run together, then fuse through a weighted ensemble.

- **AI Image Detector** — Synthetic vs authentic stills (full frame)
- **Forgery Detector** — EfficientNet-B4 face forgery scoring (face crop)
- **Frequency Analysis** — Spectral / DCT-style artifact cues
- **Ensemble** — Calibrated Real / Fake probabilities

*Untrained models (attention_network, etc.) are gated out to prevent noise in the final score.*

</td>
<td width="50%">

### Clear Results
Every run aims to be actionable, not just a number.

- Real / Fake label with confidence
- **Dual scores**: AI-Gen probability + Face Forgery probability
- Per-model votes and timings
- Plain-language explanation
- Optional JSON export for tooling

</td>
</tr>
<tr>
<td>

### Ready to Use
Practical interfaces for local and server workflows.

- **CLI** — `deepshield detect …`
- **FastAPI** — REST endpoints for apps
- **Docker** — Compose-based deployment layout
- **CI** — GitHub Actions workflow included

</td>
<td>

### Built to Grow
Designed as a platform, not a one-off script.

- Pluggable model registry
- Face detection + alignment preprocessing
- Configurable device (`cpu` / `cuda` / `auto`)
- Benchmark script for accuracy tracking

</td>
</tr>
</table>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DeepShield Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Image                                                    │
│       │                                                          │
│       ├──────────────────────────────────────────────────┐       │
│       │                                                  │       │
│       ▼                                                  ▼       │
│  ┌──────────────────┐                        ┌────────────────┐  │
│  │  FACE EXTRACTION  │                        │  FULL FRAME    │  │
│  │  RetinaFace/CV2   │                        │                │  │
│  └────────┬─────────┘                        └───────┬────────┘  │
│           │                                          │           │
│           ▼                                          ▼           │
│  ┌────────────────────────┐              ┌────────────────────┐  │
│  │  Forgery (EfficientNet)│              │  AI-Gen Detector   │  │
│  │  Frequency Analysis    │              │  (Organika/sdxl)   │  │
│  └────────────┬───────────┘              └────────┬───────────┘  │
│               │                                   │              │
│               └──────────────┬────────────────────┘              │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────┐                                            │
│  │     ENSEMBLE      │  Weighted fusion + AI override            │
│  └────────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  Output: prediction · confidence · AI-gen score · forgery score  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/007sandesh/DeepShield.git
cd DeepShield

python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -e ".[dev]"
```

### 2. Download model weights

```bash
python scripts/download_models.py
# or lighter set:
python scripts/download_models.py --light
```

Weights land in `models/weights/` (gitignored — download once per machine).

### 3. Detect

```bash
# Module form (always works after install)
python -m src.cli.main detect path/to/image.jpg --verbose

# Or via entry point, if on PATH
deepshield detect path/to/image.jpg -o result.json

# Apply a confidence threshold (below = UNCERTAIN)
deepshield detect path/to/image.jpg --threshold 0.7

# Suppress explanation output
deepshield detect path/to/image.jpg --no-explain
```

### 4. Optional: API server

```bash
deepshield serve
# Docs: http://localhost:8000/docs
# Metrics: http://localhost:8000/metrics
```

### Docker

```bash
docker-compose up -d
```

### Python API

```python
from src.core.pipelines.image_pipeline import ImageDetectionPipeline

pipeline = ImageDetectionPipeline(device="auto")
pipeline.initialize()

result = pipeline.detect("path/to/image.jpg")
print(result.prediction, result.confidence, result.explanation)
print(f"AI-gen score: {result.ai_gen_score}")
print(f"Face forgery score: {result.face_forgery_score}")
```

---

## How It Works

1. **Face extraction** — RetinaFace (with OpenCV fallback) finds and aligns faces.
2. **AI-gen classification** — `Organika/sdxl-detector` (HuggingFace) runs on the full frame to detect synthetic media.
3. **Forgery scoring** — EfficientNet-B4 loads Celeb-DF-style checkpoint weights for face Real/Fake classification.
4. **Frequency cues** — Looks for unnatural smoothness / spectral patterns common in forged faces.
5. **Ensemble** — Combines model probabilities; high-confidence AI-gen hits get an override boost.

**Model weights status:**
- `efficientnet_celebdf.pt` — Celeb-DF + FF++ trained checkpoint (~98.5% val accuracy in checkpoint metadata)
- `Organika/sdxl-detector` — HuggingFace model, loaded at runtime (no local download needed)
- `vit_face_forensics.pt` — **⚠ Generic ViT-B/16 base weights, NOT forensics-finetuned.** Used only when explicitly selected via `backbone="vit"`.
- `xception_ffpp.pt` — FaceForensics++ checkpoint; architecture may not fully match (loaded with `strict=False`).

Your real-world results depend on media quality, face size, compression, and deepfake type — always review the per-model breakdown when stakes are high.

---

## Benchmarking

Run the benchmark script against your own data:

```bash
python scripts/benchmark.py --real-dir ./benchmark_data/real \
                            --fake-dir ./benchmark_data/fake \
                            --ai-dir ./benchmark_data/ai_generated \
                            --output benchmark_results.json
```

### Domain adaptation (modern face-swaps)

Off-the-shelf Celeb-DF weights often miss InsightFace / `inswapper` swaps. Industry practice is **transfer learning with a real-face replay buffer**:

```bash
# 1) Put train pairs in benchmark_data/finetune/{real,fake}
# 2) Fine-tune last EfficientNet stages + head
python scripts/finetune_faceswap.py --epochs 10 --device cpu

# Writes models/weights/efficientnet_inswapper.pt (auto-loaded by the ensemble)
```

Expected directory layout:
```
benchmark_data/
├── real/           # Authentic face images
├── fake/           # Face-swap deepfakes
└── ai_generated/   # ChatGPT, Midjourney, SDXL stills, etc.
```

The script reports overall accuracy, FPR (false positives on real faces), and FNR (missed fakes).

---

## Roadmap

| Status | Item |
|--------|------|
| ✅ **Done** | Face deepfake image detection (EfficientNet + ensemble) |
| ✅ **Done** | AI-generated image detection (Organika/sdxl-detector via HuggingFace) |
| ✅ **Done** | Dual AI-gen / face-forgery score surfacing |
| ✅ **Done** | Untrained model gating (attention_network excluded from vote) |
| ✅ **Done** | Industry domain adaptation for InsightFace / inswapper face-swaps |
| ✅ **Done** | Soft-vote forgery ensemble + calibrated thresholds |
| ✅ **Done** | CLI `--threshold` and `--no-explain` flags |
| ✅ **Done** | API `/metrics` endpoint |
| 🔧 **Next** | Stronger video end-to-end packaging and richer explainability UI |
| 🔧 **Next** | Broader adversarial evaluation and production hardening |

---

## Project Structure

```
DeepShield/
├── src/
│   ├── core/
│   │   ├── models/           # Detectors + ensemble
│   │   ├── pipelines/        # Image & video pipelines
│   │   └── preprocessors/    # Face / frame / audio prep
│   ├── api/                  # FastAPI app
│   ├── cli/                  # Rich CLI
│   └── config/               # Settings
├── scripts/
│   ├── download_models.py    # Weight downloader
│   └── benchmark.py          # Accuracy benchmark (FPR/FNR)
├── tests/
├── docker/
├── docs/
├── docker-compose.yml
└── pyproject.toml
```

---

## Testing

```bash
pytest tests/ -v
# or
make test
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML | PyTorch, timm, transformers |
| Vision | OpenCV, RetinaFace, MediaPipe |
| API / CLI | FastAPI, Typer, Rich |
| Packaging | Docker, GitHub Actions |

---

## Contributing

1. Fork the repo  
2. Create a branch (`git checkout -b feature/your-idea`)  
3. Commit and push  
4. Open a Pull Request  

Issues and PRs that improve accuracy, UX, or documentation are especially welcome.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Face forgery research dataset  
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) — Challenging deepfake benchmark  
- [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) — Face detection  
- [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector) — AI-generated image classification  
- Community EfficientNet / Xception deepfake checkpoints on Hugging Face  

---

<div align="center">

**Built to help people trust what they see — face deepfakes and generative AI imagery.**

[GitHub](https://github.com/007sandesh/DeepShield)

</div>
