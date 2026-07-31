"""
FastAPI Image Captioning Service

Endpoints:
    POST /caption     — Upload an image, get a caption + Grad-CAM overlay
    GET  /health      — Health check (verify model is loaded)

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Test with curl:
    curl -X POST "http://localhost:8000/caption" -F "file=@path/to/image.jpg"
"""

import io
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from transformers import BlipForConditionalGeneration, BlipProcessor
import matplotlib
matplotlib.use("Agg")  # non-interactive backend 
import matplotlib.pyplot as plt


# ─── Configuration ───
MODEL_NAME = os.getenv("MODEL_NAME", "Salesforce/blip-image-captioning-base")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_DIR = Path("app/outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── App Initialization ───
app = FastAPI(
    title="BLIP Image Captioning API",
    description="Generate captions and Grad-CAM visual explanations for images using BLIP.",
    version="1.0.0",
)

# Serve generated overlay images as static files
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


# ─── Model Loading (runs once at startup) ───
print(f"Loading BLIP model on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded successfully.")


# ─── Grad-CAM for ViT ───
class GradCAMViT:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = model.vision_model.encoder.layers[-1].layer_norm1
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, pixel_values):
        self.model.zero_grad()
        pixel_values = pixel_values.to(DEVICE).requires_grad_(True)
        with torch.enable_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)
            pooled = outputs.last_hidden_state[:, 0, :]
            score = pooled.sum()
            score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224))

        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self.activations).sum(dim=-1)
        cam = cam[:, 1:]
        cam = F.relu(cam)
        grid_size = int(cam.shape[1] ** 0.5)
        cam = cam.reshape(1, 1, grid_size, grid_size)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


gradcam = GradCAMViT(model)


# ─── Helper Functions ───
def save_gradcam_overlay(original_image, heatmap, output_path, alpha=0.4):
    """Save original image with Grad-CAM heatmap overlay."""
    img = original_image.resize((224, 224))
    img_array = np.array(img) / 255.0
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    overlay = img_array * (1 - alpha) + heatmap_colored * alpha
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close()


# ─── API Endpoints ───

@app.get("/health")
def health_check():
    """Check if the API and model are running."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(DEVICE),
    }


@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    """
    Upload an image and receive:
    - Generated caption (beam search, 5 beams)
    - Grad-CAM overlay image path showing where the model focused

    Usage:
        curl -X POST "http://localhost:8000/caption" -F "file=@photo.jpg"
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Generate caption
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.inference_mode():
            output_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=30,
                num_beams=5,
                early_stopping=True,
            )
        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()

        # Generate Grad-CAM heatmap
        heatmap = gradcam.generate_heatmap(inputs["pixel_values"])

        # Save overlay image
        overlay_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
        overlay_path = OUTPUTS_DIR / overlay_filename
        save_gradcam_overlay(image, heatmap, overlay_path)

        return JSONResponse({
            "caption": caption,
            "gradcam_overlay": f"/outputs/{overlay_filename}",
            "model": MODEL_NAME,
            "device": str(DEVICE),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/")
def root():
    """API information."""
    return {
        "name": "BLIP Image Captioning API",
        "version": "1.0.0",
        "endpoints": {
            "POST /caption": "Upload image -> get caption + Grad-CAM overlay",
            "GET /health": "Health check",
        },
        "usage": 'curl -X POST "http://localhost:8000/caption" -F "file=@image.jpg"',
    }
