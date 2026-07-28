"""
model.py — BLIP model loading and configuration.

Usage:
    from src.model import load_blip_model, freeze_vision_encoder, get_device
"""

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


DEFAULT_MODEL_NAME = "Salesforce/blip-image-captioning-base"


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_blip_model(model_name: str = DEFAULT_MODEL_NAME, device: torch.device = None):
    """Load pretrained BLIP processor and model.
    
    Args:
        model_name: Hugging Face model identifier or local checkpoint path
        device: torch device (auto-detected if None)
        
    Returns:
        (processor, model) tuple
    """
    if device is None:
        device = get_device()
    
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    
    return processor, model


def freeze_vision_encoder(model) -> int:
    """Freeze all parameters in the vision encoder (for text-decoder-only fine-tuning).
    
    Returns:
        Number of trainable parameters remaining
    """
    for param in model.vision_model.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return trainable


def save_checkpoint(model, processor, save_dir: str):
    """Save model and processor for deployment."""
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Checkpoint saved to {save_dir}")


def load_checkpoint(checkpoint_dir: str, device: torch.device = None):
    """Load a saved checkpoint (for deployment or continued training)."""
    if device is None:
        device = get_device()
    processor = BlipProcessor.from_pretrained(checkpoint_dir)
    model = BlipForConditionalGeneration.from_pretrained(checkpoint_dir).to(device)
    return processor, model
