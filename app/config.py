"""
app/config.py
─────────────
Loads all configuration from environment variables (with sensible defaults).
Uses python-dotenv to automatically read from a .env file if present.

Learning note:
  Keeping config in one place (not scattered across files) is called the
  "single source of truth" pattern. It makes changing settings (e.g., swapping
  CLIP models) a one-line change instead of hunting across the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if it exists, does nothing otherwise


class Settings:
    # Qdrant connection — local
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

    # Qdrant Cloud (overrides host/port when set)
    qdrant_url:     str = os.getenv("QDRANT_URL", "")       # e.g. https://xxx.cloud.qdrant.io:6333
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    # CLIP model to use - change this to try different models:
    #   "openai/clip-vit-base-patch32"  → 512-dim, fast
    #   "openai/clip-vit-large-patch14" → 768-dim, more accurate, slower
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

    # Number of search results to return
    top_k: int = int(os.getenv("TOP_K", "9"))

    # Qdrant collection name (like a "table" in a SQL database)
    collection_name: str = "multimodal_search"

    # Embedding dimension for CLIP ViT-B/32
    # Must match the model chosen above!
    vector_dim: int = 512

    # Directory where downloaded images are stored
    image_dir: str = os.getenv("IMAGE_DIR", "data/images")


settings = Settings()
