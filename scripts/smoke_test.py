"""
scripts/smoke_test.py
──────────────────────
Verifies the entire CLIP pipeline works end-to-end before running any app code.

Tests:
  1. Model loads without error
  2. Text encoding produces a 512-dim normalized vector
  3. Image encoding produces a 512-dim normalized vector
  4. A cat text query scores HIGHER against a cat image than a dog text query
     (core multimodal alignment check)

Run with:
  source .venv/bin/activate
  python scripts/smoke_test.py
"""

import io
import sys

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

print("=" * 60)
print("CLIP Smoke Test")
print("=" * 60)

# ── 1. Load model ─────────────────────────────────────────────────
print("\n[1/4] Loading CLIP model...")
try:
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print("      ✅ Model loaded")
except Exception as e:
    print(f"      ❌ FAILED: {e}")
    sys.exit(1)

# ── 2. Text encoding ──────────────────────────────────────────────
print("\n[2/4] Testing text encoding...")
texts = ["a cat sitting on a mat", "a dog running on a beach"]
inputs = processor(text=texts, return_tensors="pt", padding=True)
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

assert text_features.shape == (2, 512), f"Expected (2, 512), got {text_features.shape}"
norms = text_features.norm(dim=-1)
assert torch.allclose(norms, torch.ones(2), atol=1e-5), "Vectors not normalized!"
print(f"      ✅ Text vectors shape: {list(text_features.shape)} (normalized)")

# ── 3. Image encoding ─────────────────────────────────────────────
print("\n[3/4] Testing image encoding (downloading a cat photo)...")
try:
    # Public domain cat image from Wikimedia Commons
    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg"
    img_bytes = requests.get(img_url, timeout=10).content
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img_features = model.get_image_features(**img_inputs)
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    assert img_features.shape == (1, 512), f"Expected (1, 512), got {img_features.shape}"
    print(f"      ✅ Image vector shape: {list(img_features.shape)} (normalized)")
except Exception as e:
    print(f"      ❌ FAILED: {e}")
    sys.exit(1)

# ── 4. Cross-modal alignment check ───────────────────────────────
print("\n[4/4] Verifying cross-modal alignment...")
# Compute similarity of both text vectors against the cat image
sims = (text_features @ img_features.T).squeeze()
cat_sim = sims[0].item()
dog_sim = sims[1].item()

print(f"      Similarity [cat text ↔ cat image]: {cat_sim:.4f}")
print(f"      Similarity [dog text ↔ cat image]: {dog_sim:.4f}")

if cat_sim > dog_sim:
    print(f"\n✅ CLIP alignment verified! Cat query ({cat_sim:.4f}) > Dog query ({dog_sim:.4f})")
    print("   The model correctly scores a cat description closer to a cat image.\n")
else:
    print(f"\n❌ Alignment check FAILED: expected cat > dog, got {cat_sim:.4f} <= {dog_sim:.4f}")
    sys.exit(1)

print("=" * 60)
print("All checks passed — ready to build! 🚀")
print("=" * 60)
