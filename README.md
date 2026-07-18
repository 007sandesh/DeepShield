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

[![Focus](https://img.shields.io/badge/Focus-Face%20Deepfakes-blueviolet?style=for-the-badge)](#-what-deepshield-detects-today)
[![Models](https://img.shields.io/badge/Models-EfficientNet%20%2B%20Ensemble-success?style=for-the-badge)](#-how-it-works)
[![CLI](https://img.shields.io/badge/Interface-CLI%20%2B%20API-orange?style=for-the-badge)](#-quick-start)

</div>

---

## Why DeepShield?

Deepfakes are getting better — DeepShield gives you a practical way to check images and videos for **face manipulation and forgery**. Multiple specialized models examine the same media, then an ensemble combines their signals into one clear verdict: **REAL** or **FAKE**, with confidence and a short explanation.

| Strength | What you get |
|----------|----------------|
| **Strong on face deepfakes** | EfficientNet-B4 weights trained on Celeb-DF / FaceForensics-style face-swap data |
| **Multi-signal analysis** | Spatial forgery, frequency cues, and attention — not a single black-box score |
| **Explainable output** | Human-readable explanations and per-model breakdowns |
| **Easy to run** | One CLI command after downloading weights |
| **Open & extensible** | MIT license, FastAPI surface, Docker-friendly layout |

---

## What DeepShield Detects Today

DeepShield is tuned for **face-centric deepfakes** — swaps, reenactment, and related facial forgery patterns common in research benchmarks and real-world clips.

**Well suited for:**
- Face-swap and face-reenactment deepfakes
- Portraits / talking-head style media with a clear face
- Offline forensic checks via CLI or API

**Coming soon — AI-generated image detection:**
Dedicated coverage for fully synthetic stills (e.g. ChatGPT Images, Midjourney, Stable Diffusion / SDXL). That capability is on the roadmap so DeepShield can flag both classic deepfakes **and** generative AI media with the same clarity.

---

## Key Capabilities

<table>
<tr>
<td width="50%">

### Multi-Layer Detection
Independent analyzers run together, then fuse through a weighted ensemble.

- **Forgery Detector** — EfficientNet-B4 face forgery scoring
- **Frequency Analysis** — Spectral / DCT-style artifact cues
- **Attention Network** — Patch-level suspicion signals
- **Ensemble** — Calibrated Real / Fake probabilities

*Video pipelines also include temporal, biological, and audio-sync modules.*

</td>
<td width="50%">

### Clear Results
Every run aims to be actionable, not just a number.

- Real / Fake label with confidence
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
- Tests for core pipeline behavior

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
│  Input: Image / Video                                            │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────┐                                            │
│  │  PREPROCESSING    │  Face detect → Align → Crop               │
│  └────────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           DETECTION LAYERS (parallel)                     │    │
│  │                                                           │    │
│  │  Forgery (EfficientNet) · Frequency · Attention           │    │
│  │  (+ Temporal / Biological / Audio-Sync for video)         │    │
│  └───────────────────────────┬─────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────┐                                            │
│  │     ENSEMBLE      │  Weighted fusion + confidence             │
│  └────────┬─────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  Output: prediction · confidence · explanation · model details   │
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
```

### 4. Optional: API server

```bash
deepshield serve
# Docs: http://localhost:8000/docs
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
```

---

## How It Works

1. **Face extraction** — RetinaFace (with OpenCV fallback) finds and aligns faces.
2. **Forgery scoring** — EfficientNet-B4 loads Celeb-DF-style checkpoint weights for face Real/Fake classification.
3. **Frequency cues** — Looks for unnatural smoothness / spectral patterns common in forged faces.
4. **Attention** — Spot-checks patch-level inconsistency.
5. **Ensemble** — Combines model probabilities into a calibrated final verdict.

Upstream EfficientNet Celeb-DF checkpoint reports strong validation accuracy on face-forgery benchmarks (~98.5% val accuracy in the included weights metadata). Your real-world results depend on media quality, face size, compression, and deepfake type — always review the per-model breakdown when stakes are high.

---

## Roadmap

| Status | Item |
|--------|------|
| **Now** | Face deepfake image detection (EfficientNet + ensemble) |
| **Now** | CLI, FastAPI scaffold, Docker layout, unit tests |
| **Next** | Dedicated **AI-generated image** detection (ChatGPT / Midjourney / SDXL-class stills) |
| **Next** | Stronger video end-to-end packaging and richer explainability UI |
| **Later** | Broader adversarial evaluation and production hardening |

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
├── scripts/download_models.py
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
|-------|--------|
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
- Community EfficientNet / Xception deepfake checkpoints on Hugging Face  

---

<div align="center">

**Built to help people trust what they see — starting with face deepfakes, expanding next to generative AI imagery.**

[GitHub](https://github.com/007sandesh/DeepShield)

</div>
