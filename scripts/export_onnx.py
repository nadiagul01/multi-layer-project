"""
ONNX Export Script — Export BLIP vision encoder to ONNX format

ONNX (Open Neural Network Exchange) is a universal model format.
Benefits: faster inference, works across frameworks (PyTorch, TensorFlow, mobile)

Run: python scripts/export_onnx.py

Output: exports/blip_vision_encoder.onnx
"""

import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ─── Setup ───
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)
ONNX_PATH = EXPORT_DIR / "blip_vision_encoder.onnx"

MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEVICE = torch.device("cpu")  # ONNX export works on CPU

print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded.")

# ─── Create dummy input ───
dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=dummy_image, return_tensors="pt")
pixel_values = inputs["pixel_values"]

print(f"Input shape: {pixel_values.shape}")

# ─── Export vision encoder to ONNX ───
print("Exporting vision encoder to ONNX...")
start = time.time()

torch.onnx.export(
    model.vision_model,                    # model to export
    (pixel_values,),                       # dummy input
    str(ONNX_PATH),                        # output path
    input_names=["pixel_values"],          # input tensor name
    output_names=["last_hidden_state"],    # output tensor name
    dynamic_axes={                          # allow variable batch size
        "pixel_values": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"},
    },
    opset_version=14,                       # ONNX opset version
    do_constant_folding=True,              # optimize constants
)

export_time = time.time() - start
file_size = ONNX_PATH.stat().st_size / (1024 * 1024)

print(f"\nExport successful!")
print(f"  File: {ONNX_PATH}")
print(f"  Size: {file_size:.1f} MB")
print(f"  Time: {export_time:.1f} seconds")

# ─── Verify ONNX model ───
try:
    import onnx
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("  ONNX validation: PASSED")
except ImportError:
    print("  (install 'onnx' package for validation: pip install onnx)")
except Exception as e:
    print(f"  ONNX validation error: {e}")

# ─── Benchmark: PyTorch vs ONNX ───
try:
    import onnxruntime as ort

    # PyTorch inference time
    with torch.inference_mode():
        start = time.time()
        for _ in range(10):
            _ = model.vision_model(pixel_values=pixel_values)
        pytorch_time = (time.time() - start) / 10

    # ONNX inference time
    session = ort.InferenceSession(str(ONNX_PATH))
    numpy_input = pixel_values.numpy()
    start = time.time()
    for _ in range(10):
        _ = session.run(None, {"pixel_values": numpy_input})
    onnx_time = (time.time() - start) / 10

    speedup = pytorch_time / onnx_time
    print(f"\n  Benchmark (10 runs average):")
    print(f"    PyTorch: {pytorch_time*1000:.1f} ms")
    print(f"    ONNX:    {onnx_time*1000:.1f} ms")
    print(f"    Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

except ImportError:
    print("\n  (install 'onnxruntime' for benchmark: pip install onnxruntime)")

print(f"\nDone! ONNX model saved to: {ONNX_PATH}")
