<p align="center">
  <h1 align="center">🖼️ Explainable Image Captioning with BLIP</h1>
  <p align="center">
    <em>Multimodal AI • Visual Explanations • Production MLOps Pipeline</em>
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/Salesforce/blip-image-captioning-base"><img src="https://img.shields.io/badge/🤗_Model-BLIP-FFD21E" alt="BLIP"></a>
  <a href="https://github.com/nadiagul01/multi-layer-project/actions"><img src="https://img.shields.io/github/actions/workflow/status/nadiagul01/multi-layer-project/ci.yml?label=CI%2FCD&logo=github" alt="CI/CD"></a>
  <a href="https://huggingface.co/spaces/Nadiagul/blip-captioning"><img src="https://img.shields.io/badge/🤗_Spaces-Live_Demo-blue" alt="HF Spaces"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <strong>Upload any image → receive a natural-language caption + Grad-CAM visual explanation</strong>
</p>

---

## 🎯 Overview

An end-to-end **multimodal AI pipeline** that generates image captions using [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) (Bootstrapping Language-Image Pre-training) and provides **visual explanations** via Grad-CAM heatmaps — showing which image regions influenced each caption.

Built with a full **MLOps production stack**: experiment tracking (MLflow), data versioning (DVC), REST API (FastAPI), containerization (Docker), and CI/CD (GitHub Actions).

### Key Highlights

| Metric | Value |
|--------|-------|
| **METEOR Score** | 47.58 (+29% over zero-shot baseline) |
| **Dataset** | Flickr8k — 8,091 images, 40,455 captions |
| **XAI Methods** | Grad-CAM, LIME-style Perturbation, Cross-Attention |
| **MLflow Runs** | 6 tracked experiments + registered model v1 |
| **Unit Tests** | 20 (all passing) |
| **ONNX Export** | 1.12× faster inference |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌───────────────────────────┐    ┌──────────────────────┐  │
│  │  Input    │    │      BLIP MODEL            │    │    Outputs           │  │
│  │  Image    │───▶│  ┌─────────────────────┐  │───▶│  • Caption           │  │
│  │  (JPEG/   │    │  │  Vision Encoder      │  │    │  • Grad-CAM Overlay  │  │
│  │   PNG)    │    │  │  (ViT - frozen)      │  │    │  • Confidence Score  │  │
│  └──────────┘    │  └──────────┬────────────┘  │    └──────────────────────┘  │
│                  │             │                │                             │
│                  │  ┌──────────▼────────────┐  │    ┌──────────────────────┐  │
│                  │  │  Text Decoder          │  │    │  XAI Layer           │  │
│                  │  │  (fine-tuned)          │  │◄───│  • Grad-CAM          │  │
│                  │  └──────────────────────┘  │    │  • LIME Perturbation  │  │
│                  └───────────────────────────┘    │  • Attention Maps     │  │
│                                                    └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                          SERVING LAYER                                      │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────────────┐ │
│  │  FastAPI    │  │  Gradio  │  │ Streamlit│  │   Docker Container        │ │
│  │  REST API   │  │  Web UI  │  │  Web UI  │  │   (Python 3.11-slim)      │ │
│  │  /caption   │  │  :7860   │  │  :8501   │  │   Port 8000               │ │
│  └────────────┘  └──────────┘  └──────────┘  └───────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                          MLOPS LAYER                                        │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────────────┐ │
│  │  MLflow     │  │   DVC    │  │  pytest  │  │   GitHub Actions CI/CD    │ │
│  │  6 runs +   │  │  Data +  │  │  20 unit │  │   lint → test → build    │ │
│  │  Registry   │  │  Model   │  │  tests   │  │   Docker on merge        │ │
│  └────────────┘  └──────────┘  └──────────┘  └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

Setup
```bash
git clone https://github.com/nadiagul01/multi-layer-project.git
cd multi-layer-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Open **http://localhost:8000/docs** for the interactive Swagger UI.
```

### Gradio Interactive UI

```bash
pip install gradio
python app/gradio_app.py
# Open http://localhost:7860
### FastAPI:
pip install fastapi uvicorn python-multipart
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```

### Option 4: Live Demo

🔗 **[Try it on Gradio →]()**

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | API information and usage guide |
| `/health` | `GET` | Health check — verify model is loaded |
| `/caption` | `POST` | Upload image → caption + Grad-CAM overlay |

### Generate a Caption

```bash
curl -X POST "http://localhost:8000/caption" -F "file=@photo.jpg"
```

**Response:**
```json
{
    "caption": "a man with a backpack walking on a trail",
    "gradcam_overlay": "/outputs/gradcam_087a2c65.png",
    "model": "Salesforce/blip-image-captioning-base",
    "device": "cpu"
}
```

---

## 📊 Experiment Results

All experiments tracked with **MLflow** (6 runs + registered production model).

Metric	Zero-Shot	Fine-Tuned	Improvement
BLEU	20.37	22.87	+2.50
ROUGE-1	50.73	53.59	+2.86
METEOR	36.88	47.58	+10.70

Selected configuration: lr=5e-5, 500 training images, beam search (5 beams)

---

## 🔍 Explainability (XAI)

Three complementary XAI methods were applied to understand how the model generates captions:

| Method | What It Reveals | Applied To |
|--------|----------------|------------|
| **Grad-CAM** | Heatmap of important image regions | Vision Encoder (ViT) |
| **LIME-style Perturbation** | Region importance via patch occlusion | Full pipeline |
| **Cross-Attention** | Word-to-region alignment | Decoder attention |

### Key Findings

- ✅ **Correct captions**: Model focuses on the main subject → METEOR ~83
- 🎯 **Insight**: Focused attention on the primary subject strongly correlates with caption quality

---

## 🧪 Testing

```bash
# Run fast tests (20 tests, ~2 min)
pytest tests/test_pipeline.py -v -m "not slow"

# Run ALL tests including model loading
pytest tests/test_pipeline.py -v
```

Test coverage includes:
- Data loader: file loading, caption parsing, dataset splits, edge cases
- Inference: caption generation, evaluation metrics, decoding strategies
- Model: loading, freezing, device handling

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Model** | BLIP (`Salesforce/blip-image-captioning-base`) |
| **Framework** | PyTorch, Hugging Face Transformers |
| **Evaluation** | BLEU, ROUGE-1, ROUGE-L, METEOR |
| **XAI** | Grad-CAM, LIME-style Perturbation |
| **API** | FastAPI + Uvicorn |
| **Web UI** | Gradio, Streamlit |
| **Containerization** | Docker (Python 3.11-slim) |
| **Experiment Tracking** | MLflow (6 runs + model registry v1) |
| **Data Versioning** | DVC |
| **Testing** | pytest (20 unit tests) |
| **CI/CD** | GitHub Actions (lint + test + Docker build) |
| **Deployment** | Hugging Face Spaces, Render |
| **Model Export** | ONNX (1.12× faster inference) |

---
CI/CD
GitHub Actions runs on every push: flake8 linting, pytest, Docker build (on merge to main)

ONNX Export
bash
python scripts/export_onnx.py
# Result: 1.12x faster inference (1079ms vs 1211ms)
Technologies
PyTorch, Transformers, BLIP, NLTK, FastAPI, Gradio, Docker, MLflow, DVC, pytest, GitHub Actions, ONNX
## 🚢 Deployment

### Hugging Face Spaces

The Gradio app is deployed at: **[huggingface.co/spaces/Nadiagul/blip-captioning](https://huggingface.co/spaces/Nadiagul/blip-captioning)**

### Docker

```bash
docker build -t blip-captioning-api .
docker run -p 8000:8000 blip-captioning-api
```

## 👩‍💻 Author

**Nadia Gul**  
AI/ML  — National University of Computer and Emerging Sciences (NUCES/FAST), Islamabad  
Mentor: **Dr. Mateen Yaqoob**

[![GitHub](https://img.shields.io/badge/GitHub-nadiagul01-181717?logo=github)](https://github.com/nadiagul01)
[![Hugging Face](https://img.shields.io/badge/🤗_HuggingFace-Nadiagul-FFD21E)](https://huggingface.co/Nadiagul)

---

<p align="center">
  <em>Built with ❤️ as part of AI/ML internship focusing on multimodal deep learning, explainability, and production deployment.</em>
</p>
