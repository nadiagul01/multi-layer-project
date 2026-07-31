# multi-layer-project

# 🖼️ BLIP Image Captioning with Explainability

> Automated image caption generation using **BLIP** (Bootstrapping Language-Image Pre-training) on the **Flickr8k** dataset, with built-in **Grad-CAM visual explanations** — served via **FastAPI** and containerized with **Docker**.

---

## ✨ Features

- **Image Captioning** — Upload any image, receive a natural-language description
- **Visual Explanations** — Grad-CAM heatmaps show which image regions influenced the caption
- **Experiment Tracking** — MLflow dashboard with 6+ tracked runs and a registered production model
- **XAI Analysis** — Grad-CAM, LIME-style perturbation, and correct vs incorrect comparison
- **Production-Ready** — FastAPI REST API, Dockerized, with automated pytest testing
- **Data Versioning** — DVC tracks dataset and model artifacts

---

## 📁 Project Structure

```
multi-layer-project/
│
├── app/                          # FastAPI application
│   ├── main.py                   #   API endpoints: /caption, /health
│   └── outputs/                  #   Generated Grad-CAM overlay images
│
├── src/                          # Modular Python codebase
│   ├── data_loader.py            #   Dataset loading, splitting, PyTorch Dataset
│   ├── model.py                  #   BLIP model loading, freezing, checkpointing
│   ├── inference.py              #   Caption generation (greedy/beam) & evaluation
│   └── xai.py                    #   Grad-CAM for ViT & perturbation analysis
│
├── notebooks/                    # Weekly experiment notebooks
│   ├── week1/                    #   EDA, preprocessing, library testing
│   ├── week2/                    #   Zero-shot BLIP, fine-tuning, MLflow experiments
│   └── week3/                    #   XAI: Grad-CAM, LIME, comparison analysis
│
├── tests/
│   └── test_pipeline.py          # 20 pytest unit tests (data loader + inference)
│
├── scripts/
│   └── register_model.py         # MLflow model registry script
│
├── data/                         # Dataset (DVC-tracked, not in git)
│   ├── flickr8k/                 #   8,091 images + captions
│   ├── processed/                #   Preprocessed outputs
│   └── xai_examples/             #   15 curated XAI examples
│
├── Dockerfile                    # Container build recipe
├── requirements.txt              # Python dependencies (local development)
├── requirements-docker.txt       # Python dependencies (Docker — CPU-only PyTorch)
├── dvc.yaml                      # DVC pipeline definition
└── README.md
```

---

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/nadiagul01/multi-layer-project.git
cd multi-layer-project

# Create virtual environment
python -m venv venv
venv\Scripts\activate              # Windows
source venv/bin/activate           # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** to access the interactive Swagger UI.

### Option 2: Run with Docker

```bash
# Build the Docker image (~5 min first time)
docker build -t blip-captioning-api .

# Run the container
docker run -p 8000:8000 blip-captioning-api
```

Open **http://localhost:8000/docs** — the API is running inside the container.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | API information and usage guide |
| `/health` | `GET` | Health check — verify model is loaded |
| `/caption` | `POST` | Upload image → receive caption + Grad-CAM overlay |

### Example — Generate a Caption

**Request:**
```bash
curl -X POST "http://localhost:8000/caption" -F "file=@photo.jpg"
```

**Response:**
```json
{
    "caption": "a man with a backpack",
    "gradcam_overlay": "/outputs/gradcam_087a2c65.png",
    "model": "Salesforce/blip-image-captioning-base",
    "device": "cpu"
}
```

The Grad-CAM overlay image can be viewed at: `http://localhost:8000/outputs/gradcam_087a2c65.png`

---

## 🧪 Running Tests

```bash
# Run all fast tests (20 tests, ~2 min)
pytest tests/test_pipeline.py -v -m "not slow"

# Run ALL tests including model loading (slower)
pytest tests/test_pipeline.py -v
```

---

## 📊 MLflow — Experiment Tracking & Model Registry

```bash
cd notebooks/week2
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open **http://localhost:5000** to view:

- **Runs tab** — 6 tracked experiments (zero-shot baseline + fine-tuning configurations)
- **Models tab** — Registered production model: `blip-flickr8k-captioning` v1

### Tracked Experiments

| Run | Type | Key Metric |
|-----|------|------------|
| `zero_shot_blip_baseline` | Baseline | METEOR: 36.88 |
| `exp1_baseline` | Fine-tuned (lr=5e-5, 200 imgs) | METEOR: 46.55 |
| `exp2_high_lr` | Fine-tuned (lr=1e-4, 200 imgs) | METEOR: 49.42 |
| `exp3_large_subset` | Fine-tuned (lr=5e-5, 500 imgs) | METEOR: 47.58 |
| `blip_text_decoder_finetune` | Initial fine-tune | Training loss tracked |
| `week3_xai_analysis` | XAI artifacts | Grad-CAM + comparison |

**Selected baseline:** `exp3_large_subset` — chosen for overall consistency and lowest training loss.

---

## 🔍 Explainability (XAI)

Three complementary methods were applied to understand model predictions:

| Method | What It Shows | Applied To |
|--------|--------------|------------|
| **Grad-CAM** | Heatmap of important image regions | Vision encoder (ViT) |
| **Perturbation (LIME-style)** | Region importance via occlusion | Full pipeline |
| **Cross-Attention** | Word-to-region alignment | Decoder (attempted) |

**Key Finding:** Focused attention on the main subject correlates with accurate captions (METEOR ~83), while diffuse or misplaced attention leads to poor captions (METEOR ~5).

---

## 📅 Weekly Progress

| Week | Focus | Key Deliverables |
|------|-------|-----------------|
| **Week 1** | Data Preparation | Flickr8k EDA, text/image preprocessing, vocabulary building |
| **Week 2** | Model Development | Zero-shot BLIP, fine-tuning, 3 MLflow experiments, +10 METEOR improvement |
| **Week 3** | Explainability (XAI) | Grad-CAM heatmaps, LIME perturbation, correct vs incorrect analysis |
| **Week 4** | Production Pipeline | Code refactoring, DVC, MLflow registry, pytest, FastAPI, Docker |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Model** | BLIP (Salesforce/blip-image-captioning-base) |
| **Framework** | PyTorch, Hugging Face Transformers |
| **API** | FastAPI, Uvicorn |
| **Containerization** | Docker |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **Testing** | pytest (20 unit tests) |
| **Evaluation Metrics** | BLEU, ROUGE-1, ROUGE-L, METEOR |
| **XAI** | Grad-CAM, Perturbation/LIME |

---

## 👩‍💻 Author

**Nadia Gul** — [GitHub](https://github.com/nadiagul01)

Built as part of AI/ML internship project focusing on multimodal deep learning, explainability, and production deployment.