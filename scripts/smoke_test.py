"""
scripts/smoke_test.py
──────────────────────
Verifies the entire CLIP pipeline works end-to-end before running any app code.

Tests:
  1. Model loads without error
  2. Text encoding produces a 512-dim normalized vector
  3. Image encoding produces a 512-dim normalized vector (synthetic image — no network)
  4. Semantic similarity: related texts score higher than unrelated texts

Run with:
  source .venv/bin/activate
  python scripts/smoke_test.py
"""

import sys

import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

print("=" * 60)
print("CLIP Smoke Test")
print("=" * 60)


# ── Helper: encode text inline (no app import needed here) ────────

def _encode_text(text: str, model, processor) -> torch.Tensor:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)


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


# ── 3. Image encoding (synthetic image — no network needed) ───────
print("\n[3/4] Testing image encoding (synthetic image)...")
try:
    # Create a 224×224 RGB image — CLIP accepts any PIL RGB image
    image = Image.new("RGB", (224, 224), color=(180, 140, 100))
    draw = ImageDraw.Draw(image)
    draw.ellipse([60, 60, 164, 164], fill=(210, 160, 90))
    draw.ellipse([80, 30, 120, 70], fill=(200, 150, 80))

    img_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img_features = model.get_image_features(**img_inputs)
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    assert img_features.shape == (1, 512), f"Expected (1, 512), got {img_features.shape}"
    img_norm = img_features.norm(dim=-1).item()
    assert abs(img_norm - 1.0) < 1e-4, f"Image vector not normalized: {img_norm}"
    print(f"      ✅ Image vector shape: {list(img_features.shape)} (normalized)")
    print(f"      (Synthetic image used — real photo tests in tests/test_embeddings.py)")
except Exception as e:
    print(f"      ❌ FAILED: {e}")
    sys.exit(1)


# ── 4. Semantic similarity check ──────────────────────────────────
print("\n[4/4] Verifying semantic embedding space...")

vec_cat    = _encode_text("a cat sitting on a mat", model, processor).squeeze()
vec_kitten = _encode_text("a kitten resting on a rug", model, processor).squeeze()
vec_rocket = _encode_text("a rocket launching into space", model, processor).squeeze()

sim_related   = float(torch.dot(vec_cat, vec_kitten))
sim_unrelated = float(torch.dot(vec_cat, vec_rocket))

print(f"      Similarity [cat ↔ kitten]:  {sim_related:.4f}  (should be HIGH)")
print(f"      Similarity [cat ↔ rocket]:  {sim_unrelated:.4f}  (should be LOW)")

if sim_related > sim_unrelated:
    print(f"\n✅ Semantic space verified! Related ({sim_related:.4f}) > Unrelated ({sim_unrelated:.4f})")
    print("   CLIP correctly clusters similar meanings close together.\n")
else:
    print(f"\n❌ Check FAILED: expected related > unrelated, got {sim_related:.4f} <= {sim_unrelated:.4f}")
    sys.exit(1)

print("=" * 60)
print("All checks passed — ready to build! 🚀")
print("=" * 60)
