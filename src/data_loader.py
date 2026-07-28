"""
data_loader.py — Data loading and preprocessing utilities for Flickr8k captioning.

Usage:
    from src.data_loader import load_captions, get_image_path, Flickr8kCaptionDataset
"""

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_captions(caption_file: str | Path) -> pd.DataFrame:
    """Load and clean the Flickr8k captions CSV file.
    
    Args:
        caption_file: Path to captions.txt
        
    Returns:
        DataFrame with columns ['image', 'caption']
    """
    captions = pd.read_csv(caption_file)
    captions.columns = [str(c).strip().lower().replace(" ", "_") for c in captions.columns]
    return captions


def get_image_dir(data_root: str | Path) -> Path:
    """Find the Images directory (handles 'Images' vs 'images' naming)."""
    data_root = Path(data_root)
    image_dir = data_root / "Images"
    if not image_dir.exists():
        image_dir = data_root / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found under {data_root}")
    return image_dir


def get_available_images(captions_df: pd.DataFrame, image_dir: str | Path) -> list[str]:
    """Return list of image filenames that exist on disk."""
    image_dir = Path(image_dir)
    return [img for img in captions_df["image"].drop_duplicates().tolist()
            if (image_dir / img).exists()]


def get_references(captions_df: pd.DataFrame, image_names: list[str]) -> dict[str, list[str]]:
    """Build a dict mapping image filename -> list of reference captions."""
    return {
        img: captions_df.loc[captions_df["image"] == img, "caption"].astype(str).tolist()
        for img in image_names
    }


def split_dataset(all_images: list[str], val_size: int = 300, 
                  train_size: int = 1000, seed: int = 42) -> dict:
    """Split images into train/val/test sets with no overlap.
    
    Returns:
        dict with keys 'train', 'val', 'test' containing image name lists
    """
    random.seed(seed)
    val_images = random.sample(all_images, min(val_size, len(all_images)))
    remaining = [img for img in all_images if img not in set(val_images)]
    random.seed(seed)
    random.shuffle(remaining)
    train_images = remaining[:min(train_size, len(remaining))]
    
    return {
        "train": train_images,
        "val": val_images,
        "test": val_images,  # same as val for this project
    }


class Flickr8kCaptionDataset(Dataset):
    """PyTorch Dataset for Flickr8k image-caption pairs.
    
    Args:
        image_names: List of image filenames
        captions_df: DataFrame with 'image' and 'caption' columns
        image_dir: Path to the Images directory
        random_caption: If True, randomly pick one of 5 captions per image (data augmentation)
    """

    def __init__(self, image_names, captions_df, image_dir, random_caption=True):
        self.image_names = image_names
        self.captions_df = captions_df
        self.image_dir = Path(image_dir)
        self.random_caption = random_caption

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        with Image.open(self.image_dir / image_name) as image:
            rgb_image = image.convert("RGB")
        options = self.captions_df.loc[
            self.captions_df["image"] == image_name, "caption"
        ].astype(str).tolist()
        text = random.choice(options) if self.random_caption else options[0]
        return rgb_image, text
