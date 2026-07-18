"""
scripts/export_vectors.py
─────────────────────────
Dumps all Qdrant vectors + payloads to data/vectors.json

This file is committed to git so HF Spaces can import vectors on startup
without needing to re-run CLIP (which takes 5+ minutes on CPU).

Import takes ~3 seconds for 541 vectors vs ~10 minutes for full re-indexing.

Run:
    python scripts/export_vectors.py
"""

import json
import pathlib
from qdrant_client import QdrantClient
from app.config import settings

DATA = pathlib.Path("data")
OUT  = DATA / "vectors.json"

def export():
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    print(f"Fetching all points from '{settings.collection_name}'...")
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=settings.collection_name,
            with_vectors=True,
            with_payload=True,
            limit=256,
            offset=offset,
        )
        points.extend(batch)
        print(f"  fetched {len(points)} so far...")
        if offset is None:
            break

    data = [
        {"id": p.id, "vector": p.vector, "payload": p.payload}
        for p in points
    ]

    OUT.write_text(json.dumps(data, indent=None, separators=(",", ":")))
    size = OUT.stat().st_size / 1024
    print(f"\n✅ Exported {len(data)} vectors → {OUT}  ({size:.0f} KB)")

if __name__ == "__main__":
    export()
