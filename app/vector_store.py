"""
app/vector_store.py
────────────────────
Qdrant client wrapper: create collection, upsert vectors, search.

Learning notes:
  - A Qdrant "collection" is analogous to a table in SQL.
    Each "point" = one item (image) stored with its vector + metadata (payload).
  - COSINE distance is correct here because our vectors are L2-normalized.
    Alternatives: DOT (for un-normalized), EUCLID (raw L2 distance).
  - uuid4() generates random IDs — Qdrant requires string or integer IDs.
    Using UUID means we can safely upsert in parallel without ID collisions.
  - The "payload" is arbitrary JSON metadata stored alongside each vector.
    It's returned with search results — no need for a separate metadata DB.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings


def get_client() -> QdrantClient:
    """Create a Qdrant client connected to the configured host."""
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection() -> None:
    """
    Create the Qdrant collection if it doesn't already exist.

    Called at API startup (idempotent — safe to call multiple times).
    Collection stores 512-dim vectors with cosine distance metric.
    """
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.collection_name not in existing:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.vector_dim,
                distance=Distance.COSINE,
            ),
        )
        print(f"[Qdrant] Created collection '{settings.collection_name}'")
    else:
        count = client.count(collection_name=settings.collection_name).count
        print(f"[Qdrant] Collection exists with {count} points")


def upsert_item(vector: list[float], payload: dict) -> str:
    """
    Insert or update a single item in the vector store.

    Args:
        vector:  512-dim float list (from encode_image or encode_text)
        payload: Arbitrary metadata dict — stored alongside the vector.
                 Example: {"filename": "cat.jpg", "caption": "a fluffy cat", "url": "..."}

    Returns:
        The generated UUID string for this point.
    """
    client = get_client()
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=settings.collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )
    return point_id


def search(query_vector: list[float], top_k: int | None = None) -> list[dict]:
    """
    Find the top-K most similar items to the query vector.

    Uses Approximate Nearest Neighbor (ANN) search via Qdrant's HNSW index.
    Much faster than brute-force for large collections.

    Args:
        query_vector: 512-dim float list (from encode_text or encode_image)
        top_k:        Number of results to return (defaults to settings.top_k)

    Returns:
        List of dicts, each containing:
          - "score": cosine similarity [0, 1] (1 = identical)
          - + all payload fields (filename, caption, url, etc.)
    """
    if top_k is None:
        top_k = settings.top_k

    client = get_client()
    results = client.search(
        collection_name=settings.collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": round(r.score, 4), **r.payload} for r in results]


def get_collection_info() -> dict:
    """Return basic stats about the collection (point count, status)."""
    client = get_client()
    info = client.get_collection(collection_name=settings.collection_name)
    return {
        "points_count": info.points_count,
        "status": str(info.status),
        "vector_size": info.config.params.vectors.size,
    }


def delete_collection() -> None:
    """Drop the entire collection — useful for re-indexing from scratch."""
    client = get_client()
    client.delete_collection(collection_name=settings.collection_name)
    print(f"[Qdrant] Deleted collection '{settings.collection_name}'")
