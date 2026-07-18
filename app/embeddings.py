"""
app/embeddings.py
─────────────────
CLIP model loading and encoding functions.

This is the CORE of the multimodal search engine. Everything else (vector DB,
API, UI) is infrastructure around these two functions: encode_text and encode_image.

Learning notes:
  - @lru_cache ensures the ~600MB model loads only ONCE per process lifetime.
    Without it, every API request would reload the model from disk (~3s each).
  - L2 normalization converts raw feature vectors to unit vectors.
    After normalization: cosine_similarity(a, b) == dot_product(a, b)
    Qdrant uses cosine distance by default, so this is required for correct results.
  - torch.no_grad() disables gradient tracking during inference.
    Gradients are only needed during training (backprop). Disabling them
    saves ~50% memory and speeds up inference.
"""

from functools import lru_cache

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import settings


@lru_cache(maxsize=1)
def get_model():
    """
    Load CLIP model and processor, cached for the process lifetime.

    Returns:
        tuple: (model, processor, device)
            model     - the CLIP model (Vision + Text encoders)
            processor - handles tokenization (text) and pixel normalization (images)
            device    - "cuda" if GPU available, else "cpu"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP] Loading {settings.clip_model} on {device}...")
    model = CLIPModel.from_pretrained(settings.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(settings.clip_model)
    model.eval()  # put in inference mode (disables dropout layers)
    print("[CLIP] Model loaded ✅")
    return model, processor, device


def encode_text(text: str) -> list[float]:
    """
    Convert a text string into a normalized 512-dim embedding vector.

    How it works:
      1. Tokenize the text (split into subword tokens, add special [CLS]/[SEP] tokens)
      2. Pass tokens through CLIP's Text Transformer
      3. Extract the [CLS] token's hidden state as the embedding
      4. L2-normalize so the vector lies on the unit hypersphere

    Args:
        text: Any natural language string, e.g. "a dog running on a beach"

    Returns:
        List of 512 floats in range [-1, 1]
    """
    model, processor, device = get_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    # L2 normalization: divide by magnitude so ||vector|| = 1.0
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().tolist()


def encode_image(image: Image.Image) -> list[float]:
    """
    Convert a PIL Image into a normalized 512-dim embedding vector.

    How it works:
      1. Resize image to 224×224, normalize pixel values to model's expected range
      2. Split into 32×32 non-overlapping patches (ViT-B/32 has 7×7 = 49 patches)
      3. Pass patches through CLIP's Vision Transformer
      4. Extract the [CLS] token embedding
      5. L2-normalize

    Args:
        image: PIL Image in RGB mode

    Returns:
        List of 512 floats in range [-1, 1]
    """
    model, processor, device = get_model()
    # Ensure RGB (handles RGBA PNGs, grayscale, etc.)
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two already-normalized vectors.
    Since both are L2-normalized, this is just their dot product.

    Returns a value in [-1, 1]:
      1.0  = identical direction (very similar)
      0.0  = orthogonal (unrelated)
     -1.0  = opposite (very different)
    """
    a = torch.tensor(vec_a)
    b = torch.tensor(vec_b)
    return float(torch.dot(a, b).item())
