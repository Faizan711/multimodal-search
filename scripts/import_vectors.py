"""
scripts/import_vectors.py
─────────────────────────
Imports pre-exported vectors from data/vectors.json into Qdrant.
Used on HF Spaces startup to avoid re-running CLIP inference.

~3 seconds for 541 vectors vs ~10 minutes full re-index.

Run:
    python scripts/import_vectors.py
"""

import json
import pathlib
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings

DATA = pathlib.Path("data") / "vectors.json"


def import_vectors():
    if not DATA.exists():
        print("❌ data/vectors.json not found — run scripts/export_vectors.py first")
        sys.exit(1)

    client = QdrantClient(
        url=settings.qdrant_url if hasattr(settings, "qdrant_url") else None,
        host=None if (hasattr(settings, "qdrant_url") and settings.qdrant_url) else settings.qdrant_host,
        port=None if (hasattr(settings, "qdrant_url") and settings.qdrant_url) else settings.qdrant_port,
        api_key=settings.qdrant_api_key if hasattr(settings, "qdrant_api_key") else None,
    )

    # Create collection if needed
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in existing:
        print(f"Creating collection '{settings.collection_name}'...")
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=settings.vector_dim, distance=Distance.COSINE),
        )
    else:
        count = client.get_collection(settings.collection_name).points_count
        if count and count >= 500:
            print(f"✅ Collection already has {count} points — skipping import")
            return

    # Load and upsert
    print(f"Loading {DATA} ...")
    data = json.loads(DATA.read_text())
    print(f"Upserting {len(data)} vectors...")

    BATCH = 128
    for i in range(0, len(data), BATCH):
        batch = data[i:i+BATCH]
        client.upsert(
            collection_name=settings.collection_name,
            points=[PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in batch],
        )
        print(f"  {min(i+BATCH, len(data))}/{len(data)}")

    print(f"✅ Import complete — {len(data)} vectors in '{settings.collection_name}'")


if __name__ == "__main__":
    import_vectors()
