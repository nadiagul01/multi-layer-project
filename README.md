<div align="center">

# Explainable Image Captioning with BLIP

*Multimodal AI • Visual Explanations • Production MLOps Pipeline*

<a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
<a href="https://huggingface.co/Salesforce/blip-image-captioning-base"><img src="https://img.shields.io/badge/Model-BLIP-FFD21E?logo=huggingface&logoColor=black" alt="BLIP"></a>
<a href="https://github.com/nadiagul01/multi-layer-project/actions"><img src="https://img.shields.io/github/actions/workflow/status/nadiagul01/multi-layer-project/ci.yml?label=CI%2FCD&logo=github" alt="CI/CD"></a>
<a href="https://huggingface.co/spaces/Nadiagul/blip-image-captioning"><img src="https://img.shields.io/badge/Live_Demo-Hugging_Face-blue?logo=huggingface&logoColor=white" alt="Live Demo"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>

**Upload any image → receive a natural-language caption + Grad-CAM visual explanation**

</div>

---

## Live Demo

**Try it now → [huggingface.co/spaces/Nadiagul/blip-image-captioning](https://huggingface.co/spaces/Nadiagul/blip-image-captioning)**

A public web app hosted on Hugging Face Spaces — upload any image and get a caption instantly, running right in your browser. The full Grad-CAM explainability view is available in the local Gradio demo (see *Getting Started*).

---

## Overview

An end-to-end **multimodal AI pipeline** that generates image captions using [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) (Bootstrapping Language-Image Pre-training) and provides **visual explanations** via Grad-CAM heatmaps — showing which image regions influenced each caption.

Built with a full **MLOps production stack**: experiment tracking (MLflow), data versioning (DVC), REST API (FastAPI), containerization (Docker), and CI/CD (GitHub Actions).

### Key Highlights

| Metric | Value |
|--------|-------|
| **METEOR Score** | 47.58 (+10.7 over zero-shot baseline) |
| **Dataset** | Flickr8k — 8,091 images, 40,455 captions |
| **XAI Methods** | Grad-CAM, LIME-style Perturbation, Cross-Attention |
| **MLflow Runs** | 6 tracked experiments + registered model v1 |
| **Unit Tests** | 20 (passing) |

---

## System Architecture

<div align="center">
  <img src="slides_images/architecture_diagram.jpg" alt="System Architecture" width="840">
</div>

The **model pipeline** encodes the image with a frozen BLIP Vision Transformer, generates a caption with the fine-tuned text decoder, and taps the encoder for Grad-CAM and LIME explanations. Around it sit a **serving layer** (FastAPI, Gradio, Docker) and an **MLOps layer** (MLflow, DVC, GitHub Actions, pytest).

---

## Features

- **Image Captioning** — Upload any image, receive a natural-language description.
- **Visual Explanations** — Grad-CAM heatmaps show which image regions influenced the caption.
- **Experiment Tracking** — MLflow dashboard with 6+ tracked runs and a registered production model.
- **XAI Analysis** — Grad-CAM, LIME-style perturbation, and correct-vs-incorrect comparison.
- **Production-Ready** — FastAPI REST API, Dockerized, with automated pytest testing.
- **Data Versioning** — DVC tracks the dataset and model artifacts.

---

## Getting Started

### Option 1 — Run locally

```bash
git clone https://github.com/nadiagul01/multi-layer-project.git
cd multi-layer-project

python -m venv venv
source venv/bin/activate            # macOS/Linux
# venv\Scripts\activate             # Windows

pip install -r requirements-hf.txt
pip install fastapi uvicorn python-multipart

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Option 2 — Run with Docker

```bash
docker build -t blip-captioning-api .
docker run -p 8000:8000 blip-captioning-api
```

Open **http://localhost:8000/docs**.

### Option 3 — Local Gradio demo (with Grad-CAM)

```bash
python app/gradio_app.py
# open http://localhost:7860
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | API information and usage guide |
| `/health` | `GET` | Health check — verify the model is loaded |
| `/caption` | `POST` | Upload image → caption + Grad-CAM overlay |

**Example**

```bash
curl -X POST "http://localhost:8000/caption" -F "file=@photo.jpg"
```

```json
{
  "caption": "a man with a backpack",
  "gradcam_overlay": "/outputs/gradcam_087a2c65.png",
  "model": "Salesforce/blip-image-captioning-base",
  "device": "cpu"
}
```

---

## Testing

```bash
pytest tests/test_pipeline.py -v -m "not slow"   # fast tests (~2 min)
pytest tests/test_pipeline.py -v                 # all tests
```

---

## MLflow — Experiment Tracking & Registry

```bash
cd notebooks/week2
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000
```

| Run | Type | METEOR |
|-----|------|--------|
| `zero_shot_blip_baseline` | Baseline | 36.88 |
| `exp1_baseline` (lr 5e-5, 200 imgs) | Fine-tuned | 46.55 |
| `exp2_high_lr` (lr 1e-4, 200 imgs) | Fine-tuned | 49.42 |
| `exp3_large_subset` (lr 5e-5, 500 imgs) | Fine-tuned | 47.58 |

**Selected baseline:** `exp3_large_subset` — chosen for consistency and lowest training loss. Registered as `blip-flickr8k-captioning` **v1**.

---

## Explainability (XAI)

| Method | What it shows | Applied to |
|--------|---------------|------------|
| **Grad-CAM** | Important image regions (heatmap) | Vision encoder (ViT) |
| **Perturbation (LIME-style)** | Region importance via occlusion | Full pipeline |
| **Cross-Attention** | Word-to-region alignment | Decoder |

**Key finding:** focused attention on the main subject correlates with accurate captions (METEOR ~83), while diffuse or misplaced attention leads to poor captions (METEOR ~5).

---

## Weekly Progress

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| **1** | Data Preparation | EDA, text/image preprocessing, vocabulary building |
| **2** | Model Development | Zero-shot BLIP, fine-tuning, MLflow experiments, +10.7 METEOR |
| **3** | Explainability | Grad-CAM, LIME perturbation, correct-vs-incorrect analysis |
| **4** | Production Pipeline | Refactor to `src/`, DVC, MLflow registry, pytest, FastAPI, Docker |
| **5** | CI/CD & Deployment | GitHub Actions, Streamlit/Gradio demo, Hugging Face Spaces, ONNX export |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Model | BLIP (`Salesforce/blip-image-captioning-base`) |
| Framework | PyTorch, Hugging Face Transformers |
| API | FastAPI, Uvicorn |
| Containerization | Docker |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Testing | pytest |
| Metrics | BLEU, ROUGE-1, ROUGE-L, METEOR |
| XAI | Grad-CAM, Perturbation/LIME |

---

## Author

**Nadia Gul** — [GitHub](https://github.com/nadiagul01)

Built as part of an AI/ML focused on multimodal deep learning, explainability, and production deployment.
