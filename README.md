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

---

## 📁 Project Structure

```
multi-layer-project/
│
├── app/                              # API & Web interfaces
│   ├── main.py                       #   FastAPI REST API (/caption, /health)
│   ├── gradio_app.py                 #   Gradio interactive UI
│   └── outputs/                      #   Generated Grad-CAM overlay images
│
├── src/                              # Modular Python codebase
│   ├── data_loader.py                #   Dataset loading, splits, PyTorch Dataset
│   ├── model.py                      #   BLIP model loading, freezing, checkpoints
│   ├── inference.py                  #   Caption generation & BLEU/ROUGE/METEOR eval
│   └── xai.py                        #   Grad-CAM for ViT & perturbation analysis
│
├── notebooks/                        # Weekly experiment notebooks
│   ├── week1/                        #   EDA, preprocessing, library setup
│   ├── week2/                        #   Zero-shot BLIP, fine-tuning, MLflow
│   └── week3/                        #   XAI: Grad-CAM, LIME, comparison analysis
│
├── tests/
│   └── test_pipeline.py              # 20 pytest unit tests
│
├── scripts/
│   ├── register_model.py             # MLflow model registry automation
│   └── export_onnx.py                # ONNX model export for optimized inference
│
├── data/                             # Dataset (DVC-tracked, not in git)
│   ├── flickr8k/                     #   8,091 images + captions
│   ├── processed/                    #   Preprocessed outputs
│   └── xai_examples/                 #   15 curated XAI examples
│
├── blip-image-captioning/            # Hugging Face Spaces deployment
│   ├── app.py                        #   Gradio app for HF Spaces
│   ├── requirements.txt              #   HF Spaces dependencies
│   └── README.md                     #   HF Spaces metadata
│
├── .github/workflows/ci.yml          # GitHub Actions CI/CD pipeline
├── Dockerfile                        # Docker container build recipe
├── render.yaml                       # Render deployment configuration
├── dvc.yaml                          # DVC pipeline definition
├── requirements.txt                  # Main Python dependencies
├── requirements-docker.txt           # Docker-specific deps (CPU PyTorch)
├── requirements-render.txt           # Render-specific deps (CPU PyTorch)
├── streamlit_app.py                  # Streamlit web interface
└── presentation.html                 # Final presentation slides
```

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/nadiagul01/multi-layer-project.git
cd multi-layer-project

# Create virtual environment
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS/Linux

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers Pillow matplotlib numpy fastapi uvicorn python-multipart

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Option 2: Run with Docker

```bash
# Build the Docker image (~5 min first time)
docker build -t blip-captioning-api .

# Run the container
docker run -p 8000:8000 blip-captioning-api
```

### Option 3: Gradio Interactive UI

```bash
python app/gradio_app.py
# Open http://localhost:7860
```

### Option 4: Live Demo

🔗 **[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/Nadiagul/blip-captioning)**

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

| Run | Configuration | METEOR | Notes |
|-----|--------------|--------|-------|
| `zero_shot_blip_baseline` | No fine-tuning | 36.88 | Baseline |
| `exp1_baseline` | lr=5e-5, 200 images | 46.55 | First fine-tune |
| `exp2_high_lr` | lr=1e-4, 200 images | 49.42 | Higher learning rate |
| `exp3_large_subset` | lr=5e-5, 500 images | 47.58 | ✅ **Selected baseline** |
| `blip_text_decoder_finetune` | Initial training | — | Loss tracked |
| `week3_xai_analysis` | XAI artifacts | — | Grad-CAM + comparison |

> **Selected model:** `exp3_large_subset` — chosen for overall consistency, lowest training loss, and robust generalization on unseen images.

### View MLflow Dashboard

```bash
cd notebooks/week2
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

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
- ❌ **Incorrect captions**: Attention is diffuse/misplaced (background instead of subject) → METEOR ~5
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

## 📅 Weekly Progress

| Week | Focus | Key Deliverables |
|------|-------|-----------------|
| **Week 1** | Data Preparation | Flickr8k EDA, text/image preprocessing, vocabulary building, Git workflow |
| **Week 2** | Model Development | Zero-shot BLIP → fine-tuning, 6 MLflow experiments, +29% METEOR improvement |
| **Week 3** | Explainability | Grad-CAM heatmaps, LIME perturbation, correct vs incorrect analysis |
| **Week 4** | Production Pipeline | Code refactoring, DVC, MLflow registry, pytest (20 tests), FastAPI, Docker |
| **Week 5** | Deployment & CI/CD | GitHub Actions, Gradio/Streamlit UIs, HF Spaces deployment, ONNX export |

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

## 🚢 Deployment

### Hugging Face Spaces

The Gradio app is deployed at: **[huggingface.co/spaces/Nadiagul/blip-captioning](https://huggingface.co/spaces/Nadiagul/blip-captioning)**

To deploy your own copy:
1. Create a new Space on Hugging Face (SDK: Gradio)
2. Upload the contents of `blip-image-captioning/` to the Space repository
3. The Space will auto-build and deploy

### Render

```bash
# Using render.yaml (Blueprint deployment)
# Connect your GitHub repo to Render → it reads render.yaml automatically

# Or manually:
# Build: pip install -r requirements-render.txt
# Start: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Docker

```bash
docker build -t blip-captioning-api .
docker run -p 8000:8000 blip-captioning-api
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Nadia Gul**  
AI/ML Internship — National University of Computer and Emerging Sciences (NUCES/FAST), Islamabad  
Mentor: **Dr. Mateen Yaqoob**

[![GitHub](https://img.shields.io/badge/GitHub-nadiagul01-181717?logo=github)](https://github.com/nadiagul01)
[![Hugging Face](https://img.shields.io/badge/🤗_HuggingFace-Nadiagul-FFD21E)](https://huggingface.co/Nadiagul)

---

<p align="center">
  <em>Built with ❤️ as part of AI/ML internship focusing on multimodal deep learning, explainability, and production deployment.</em>
</p>
