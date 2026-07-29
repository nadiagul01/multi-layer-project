"""
tests/test_pipeline.py — Unit tests for data_loader.py and inference.py

Run with: pytest tests/test_pipeline.py -v

These tests verify that the modular code in src/ works correctly:
- data_loader: loads captions, finds images, builds references, splits dataset
- inference: generates captions, returns correct types
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

# Add project root to path so src/ can be imported
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_captions,
    get_image_dir,
    get_available_images,
    get_references,
    split_dataset,
)

# ──────────────────────────────────────────────
# Configuration 
# ──────────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "data" / "flickr8k"
CAPTION_FILE = DATA_ROOT / "captions.txt"


# ──────────────────────────────────────────────
# TESTS FOR data_loader.py
# ──────────────────────────────────────────────

class TestLoadCaptions:
    """Tests for the load_captions function."""

    def test_returns_dataframe(self):
        """load_captions should return a pandas DataFrame."""
        df = load_captions(CAPTION_FILE)
        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"

    def test_has_required_columns(self):
        """DataFrame should have 'image' and 'caption' columns."""
        df = load_captions(CAPTION_FILE)
        assert "image" in df.columns, "Missing 'image' column"
        assert "caption" in df.columns, "Missing 'caption' column"

    def test_not_empty(self):
        """DataFrame should contain data (Flickr8k has 40,455 rows)."""
        df = load_captions(CAPTION_FILE)
        assert len(df) > 0, "DataFrame should not be empty"
        assert len(df) > 40000, f"Expected ~40,455 rows, got {len(df)}"

    def test_no_null_values(self):
        """No null values in image or caption columns."""
        df = load_captions(CAPTION_FILE)
        assert df["image"].notna().all(), "Found null image names"
        assert df["caption"].notna().all(), "Found null captions"


class TestGetImageDir:
    """Tests for the get_image_dir function."""

    def test_returns_valid_path(self):
        """Should return a Path that exists."""
        image_dir = get_image_dir(DATA_ROOT)
        assert image_dir.exists(), f"Image directory not found: {image_dir}"

    def test_contains_images(self):
        """Directory should contain image files."""
        image_dir = get_image_dir(DATA_ROOT)
        images = list(image_dir.glob("*.jpg"))
        assert len(images) > 0, "No .jpg files found in image directory"

    def test_invalid_path_raises(self):
        """Should raise FileNotFoundError for invalid path."""
        with pytest.raises(FileNotFoundError):
            get_image_dir("/nonexistent/path")


class TestGetAvailableImages:
    """Tests for the get_available_images function."""

    def test_returns_list(self):
        """Should return a list of strings."""
        df = load_captions(CAPTION_FILE)
        image_dir = get_image_dir(DATA_ROOT)
        images = get_available_images(df, image_dir)
        assert isinstance(images, list), "Should return a list"
        assert all(isinstance(img, str) for img in images), "All items should be strings"

    def test_correct_count(self):
        """Flickr8k should have ~8,091 available images."""
        df = load_captions(CAPTION_FILE)
        image_dir = get_image_dir(DATA_ROOT)
        images = get_available_images(df, image_dir)
        assert len(images) > 8000, f"Expected ~8,091 images, got {len(images)}"


class TestGetReferences:
    """Tests for the get_references function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        df = load_captions(CAPTION_FILE)
        image_dir = get_image_dir(DATA_ROOT)
        images = get_available_images(df, image_dir)[:5]
        refs = get_references(df, images)
        assert isinstance(refs, dict), "Should return a dict"

    def test_five_references_per_image(self):
        """Each image should have 5 reference captions."""
        df = load_captions(CAPTION_FILE)
        image_dir = get_image_dir(DATA_ROOT)
        images = get_available_images(df, image_dir)[:10]
        refs = get_references(df, images)
        for img, captions in refs.items():
            assert len(captions) == 5, f"{img} has {len(captions)} captions, expected 5"

    def test_references_are_strings(self):
        """All references should be non-empty strings."""
        df = load_captions(CAPTION_FILE)
        image_dir = get_image_dir(DATA_ROOT)
        images = get_available_images(df, image_dir)[:5]
        refs = get_references(df, images)
        for img, captions in refs.items():
            for cap in captions:
                assert isinstance(cap, str) and len(cap) > 0, f"Empty caption for {img}"


class TestSplitDataset:
    """Tests for the split_dataset function."""

    def test_returns_correct_keys(self):
        """Should return dict with train, val, test keys."""
        images = [f"img_{i}.jpg" for i in range(100)]
        splits = split_dataset(images, val_size=20, train_size=50)
        assert "train" in splits, "Missing 'train' key"
        assert "val" in splits, "Missing 'val' key"
        assert "test" in splits, "Missing 'test' key"

    def test_no_overlap(self):
        """Train and val sets should not overlap."""
        images = [f"img_{i}.jpg" for i in range(100)]
        splits = split_dataset(images, val_size=20, train_size=50)
        overlap = set(splits["train"]) & set(splits["val"])
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping images between train and val"

    def test_correct_sizes(self):
        """Sets should have requested sizes."""
        images = [f"img_{i}.jpg" for i in range(100)]
        splits = split_dataset(images, val_size=20, train_size=50)
        assert len(splits["val"]) == 20, f"Val size: {len(splits['val'])}, expected 20"
        assert len(splits["train"]) == 50, f"Train size: {len(splits['train'])}, expected 50"

    def test_reproducible(self):
        """Same seed should produce same splits."""
        images = [f"img_{i}.jpg" for i in range(100)]
        split1 = split_dataset(images, seed=42)
        split2 = split_dataset(images, seed=42)
        assert split1["val"] == split2["val"], "Same seed should produce same val set"


# ──────────────────────────────────────────────
# TESTS FOR inference.py (lightweight — no model loading)
# ──────────────────────────────────────────────

class TestDecodingStrategies:
    """Tests for decoding configuration."""

    def test_strategies_exist(self):
        """Both greedy and beam strategies should be defined."""
        from src.inference import DECODING_STRATEGIES
        assert "greedy" in DECODING_STRATEGIES, "Missing greedy strategy"
        assert "beam" in DECODING_STRATEGIES, "Missing beam strategy"

    def test_greedy_config(self):
        """Greedy should use num_beams=1."""
        from src.inference import DECODING_STRATEGIES
        assert DECODING_STRATEGIES["greedy"]["num_beams"] == 1

    def test_beam_config(self):
        """Beam should use num_beams=5."""
        from src.inference import DECODING_STRATEGIES
        assert DECODING_STRATEGIES["beam"]["num_beams"] == 5

    def test_max_tokens(self):
        """Both strategies should limit to 30 tokens."""
        from src.inference import DECODING_STRATEGIES
        for name, config in DECODING_STRATEGIES.items():
            assert config["max_new_tokens"] == 30, f"{name} max_new_tokens != 30"


# ──────────────────────────────────────────────
# INTEGRATION TEST 
# Mark with @pytest.mark.slow so it can be skipped
# Run with: pytest tests/test_pipeline.py -v -m "not slow"
# ──────────────────────────────────────────────

@pytest.mark.slow
class TestModelInference:
    """Integration tests that load the actual BLIP model (slow)."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        """Load model once for all tests in this class."""
        from src.model import load_blip_model
        self.processor, self.model = load_blip_model()
        self.model.eval()

    def test_generate_caption_returns_string(self):
        """generate_caption should return a non-empty string."""
        from src.inference import generate_caption
        image_dir = get_image_dir(DATA_ROOT)
        df = load_captions(CAPTION_FILE)
        images = get_available_images(df, image_dir)
        caption = generate_caption(self.model, self.processor, image_dir / images[0])
        assert isinstance(caption, str), "Caption should be a string"
        assert len(caption) > 0, "Caption should not be empty"

    def test_generate_caption_both_strategies(self):
        """Both greedy and beam should produce valid captions."""
        from src.inference import generate_caption
        image_dir = get_image_dir(DATA_ROOT)
        df = load_captions(CAPTION_FILE)
        images = get_available_images(df, image_dir)
        for strategy in ["greedy", "beam"]:
            caption = generate_caption(self.model, self.processor,
                                       image_dir / images[0], strategy=strategy)
            assert isinstance(caption, str) and len(caption) > 0, f"{strategy} failed"
