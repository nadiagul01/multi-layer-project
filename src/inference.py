"""
inference.py — Caption generation and evaluation utilities.

Usage:
    from src.inference import generate_caption, evaluate_model
"""

from pathlib import Path

import evaluate
import nltk
import torch
from PIL import Image
from tqdm import tqdm


# Decoding strategy configurations
DECODING_STRATEGIES = {
    "greedy": {"max_new_tokens": 30, "num_beams": 1, "do_sample": False},
    "beam":   {"max_new_tokens": 30, "num_beams": 5, "early_stopping": True},
}


def ensure_nltk_resources():
    """Download required NLTK data if not present."""
    for resource in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)


def generate_caption(model, processor, image_path: str | Path, 
                     strategy: str = "beam", device: torch.device = None) -> str:
    """Generate a caption for a single image.
    
    Args:
        model: BLIP model
        processor: BLIP processor
        image_path: Path to the image file
        strategy: 'greedy' or 'beam'
        device: torch device
        
    Returns:
        Generated caption string
    """
    if device is None:
        device = next(model.parameters()).device
    
    config = DECODING_STRATEGIES[strategy]
    
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    
    inputs = {k: v.to(device) for k, v in processor(images=rgb, return_tensors="pt").items()}
    
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **config)
    
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def evaluate_model(model, processor, image_names: list[str], image_dir: str | Path,
                   references: dict, strategy: str = "beam",
                   device: torch.device = None) -> dict:
    """Evaluate model on a set of images using BLEU, ROUGE, and METEOR.
    
    Args:
        model: BLIP model
        processor: BLIP processor
        image_names: List of image filenames to evaluate
        image_dir: Path to Images directory
        references: Dict mapping image name -> list of reference captions
        strategy: Decoding strategy ('greedy' or 'beam')
        device: torch device
        
    Returns:
        dict with 'metrics' (BLEU, ROUGE-1, ROUGE-L, METEOR) and 'predictions' list
    """
    ensure_nltk_resources()
    image_dir = Path(image_dir)
    
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    
    model.eval()
    predictions, refs_list = [], []
    
    for img_name in tqdm(image_names, desc=f"Evaluating ({strategy})"):
        caption = generate_caption(model, processor, image_dir / img_name, strategy, device)
        predictions.append(caption)
        refs_list.append(references[img_name])
    
    bleu = bleu_metric.compute(predictions=predictions, references=refs_list)
    rouge = rouge_metric.compute(predictions=predictions, references=refs_list)
    meteor = meteor_metric.compute(predictions=predictions, references=refs_list)
    
    metrics = {
        "BLEU": round(bleu["bleu"] * 100, 2),
        "ROUGE-1": round(float(rouge["rouge1"]) * 100, 2),
        "ROUGE-L": round(float(rouge["rougeL"]) * 100, 2),
        "METEOR": round(meteor["meteor"] * 100, 2),
    }
    
    return {"metrics": metrics, "predictions": predictions}
