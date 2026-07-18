"""
app/ingest.py
─────────────
One-time script to index images into Qdrant.

Run this ONCE (or whenever you add new images):
  python -m app.ingest

What it does:
  1. Reads data/captions.csv to get the list of images + their captions
  2. For each image: loads it, encodes with CLIP → 512-dim vector
  3. Upserts the vector + metadata into Qdrant

Learning notes:
  - This is an "offline" pipeline (run once, not per-request).
    In production, ingestion is often a separate worker job (e.g., Celery task)
    triggered when new content is uploaded.
  - tqdm gives a progress bar with ETA — crucial for long-running loops.
    Without it, you'd stare at a blank terminal wondering if it crashed.
  - We store the image filename and caption in the payload.
    The filename lets us serve the image file; the caption lets us display it.
"""

import csv
import pathlib

from PIL import Image
from tqdm import tqdm

from app.embeddings import encode_image
from app.vector_store import ensure_collection, get_collection_info, upsert_item

# Paths (relative to project root)
DATA_DIR = pathlib.Path("data")
IMAGES_DIR = DATA_DIR / "images"
CAPTIONS_CSV = DATA_DIR / "captions.csv"


def run(reset: bool = False) -> None:
    """
    Index all images listed in captions.csv into Qdrant.

    Args:
        reset: If True, delete and recreate the collection before indexing.
               Use this to re-index from scratch.
    """
    if not CAPTIONS_CSV.exists():
        print(f"❌ {CAPTIONS_CSV} not found. Run scripts/download_dataset.py first.")
        return

    if reset:
        from app.vector_store import delete_collection
        delete_collection()

    ensure_collection()

    # Read all rows from the CSV
    with open(CAPTIONS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"📂 Found {len(rows)} items in captions.csv")
    print(f"🖼️  Indexing into Qdrant at {IMAGES_DIR}...\n")

    success = 0
    skipped = 0

    for row in tqdm(rows, desc="Indexing", unit="img"):
        img_path = IMAGES_DIR / row["filename"]

        if not img_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            vector = encode_image(img)
            upsert_item(
                vector=vector,
                payload={
                    "filename": row["filename"],
                    "caption": row.get("caption", ""),
                    "photographer": row.get("photographer", ""),
                    "unsplash_url": row.get("unsplash_url", ""),
                },
            )
            success += 1
        except Exception as e:
            print(f"\n⚠️  Skipped {row['filename']}: {e}")
            skipped += 1

    info = get_collection_info()
    print(f"\n✅ Done! Indexed {success} images, skipped {skipped}")
    print(f"📊 Collection now has {info['points_count']} total points")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest images into Qdrant")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before indexing",
    )
    args = parser.parse_args()
    run(reset=args.reset)
