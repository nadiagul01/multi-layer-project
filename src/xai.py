"""
xai.py — Explainability utilities: Grad-CAM for ViT and perturbation analysis.

Usage:
    from src.xai import GradCAMViT, compute_perturbation_importance, create_overlay
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from .inference import generate_caption


class GradCAMViT:
    """Grad-CAM adapted for Vision Transformer (ViT) encoders.
    
    Args:
        model: BLIP model
        target_layer: Layer to hook (typically model.vision_model.encoder.layers[-1].layer_norm1)
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        self.gradients = None
        
        if target_layer is None:
            target_layer = model.vision_model.encoder.layers[-1].layer_norm1
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap for given pixel values.
        
        Returns:
            numpy array (224, 224) normalized to 0-1
        """
        device = next(self.model.parameters()).device
        self.model.zero_grad()
        pixel_values = pixel_values.to(device).requires_grad_(True)
        
        with torch.enable_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)
            pooled = outputs.last_hidden_state[:, 0, :]
            score = pooled.sum()
            score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224))

        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self.activations).sum(dim=-1)
        cam = cam[:, 1:]  # remove CLS token
        cam = F.relu(cam)
        
        grid_size = int(cam.shape[1] ** 0.5)
        cam = cam.reshape(1, 1, grid_size, grid_size)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def compute_perturbation_importance(model, processor, image_path, 
                                    original_caption: str,
                                    grid_size: int = 7) -> np.ndarray:
    """LIME-style perturbation analysis: occlude image patches and measure caption change.
    
    Args:
        model: BLIP model
        processor: BLIP processor
        image_path: Path to image
        original_caption: The caption generated without occlusion
        grid_size: Number of patches per side (7 = 49 total patches)
        
    Returns:
        numpy array (grid_size, grid_size) with importance scores 0-1
    """
    device = next(model.parameters()).device
    
    with Image.open(image_path) as img:
        rgb = img.convert("RGB").resize((224, 224))
    img_array = np.array(rgb)
    
    patch_size = 224 // grid_size
    importance_map = np.zeros((grid_size, grid_size))

    for row in range(grid_size):
        for col in range(grid_size):
            occluded = img_array.copy()
            r_start, c_start = row * patch_size, col * patch_size
            occluded[r_start:r_start+patch_size, c_start:c_start+patch_size] = 128
            
            occ_img = Image.fromarray(occluded)
            inputs = processor(images=occ_img, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=30, num_beams=1, do_sample=False)
            occ_caption = processor.decode(out[0], skip_special_tokens=True).strip()
            
            orig_words = set(original_caption.lower().split())
            occ_words = set(occ_caption.lower().split())
            change = len(orig_words.symmetric_difference(occ_words)) / max(len(orig_words | occ_words), 1)
            importance_map[row, col] = change

    if importance_map.max() > importance_map.min():
        importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min())

    return importance_map


def create_overlay(image_path, heatmap: np.ndarray, alpha: float = 0.4) -> tuple:
    """Create a heatmap overlay on the original image.
    
    Returns:
        (original_array, overlay_array) both normalized 0-1
    """
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    overlay = img_array * (1 - alpha) + heatmap_colored * alpha
    overlay = np.clip(overlay, 0, 1)
    
    return img_array, overlay
