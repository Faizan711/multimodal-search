"""
scripts/download_dataset.py
────────────────────────────
Downloads ~500 open-license images from the Unsplash Lite dataset.

The Unsplash Lite dataset is a freely available CSV of 25,000 photos
published by Unsplash. No API key is needed for the lite version.

What this script does:
  1. Downloads the Unsplash Lite metadata CSV from GitHub
  2. Selects a diverse sample of 500 photos
  3. Downloads each photo at 400px width (small, fast, good for demo)
  4. Saves images to data/images/ and metadata to data/captions.csv

Run with:
  python scripts/download_dataset.py
  python scripts/download_dataset.py --count 200  # smaller sample

Learning notes:
  - We download images at w=400 (not full resolution) to keep file sizes small.
    CLIP resizes everything to 224×224 anyway, so higher resolution is wasted.
  - Using a public dataset (not your own photos) makes this project shareable
    and avoids copyright issues for a public portfolio demo.
"""

import argparse
import csv
import io
import pathlib
import time

import requests
from PIL import Image
from tqdm import tqdm

DATA_DIR = pathlib.Path("data")
IMAGES_DIR = DATA_DIR / "images"
CAPTIONS_CSV = DATA_DIR / "captions.csv"

# Unsplash Lite dataset — 25k photo metadata CSV, no auth needed
UNSPLASH_LITE_CSV_URL = (
    "https://raw.githubusercontent.com/unsplash/datasets/master/lite/photos.tsv000"
)


def download_dataset(count: int = 500) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"⬇️  Downloading Unsplash Lite metadata...")
    resp = requests.get(UNSPLASH_LITE_CSV_URL, timeout=30)
    resp.raise_for_status()

    # Parse TSV (tab-separated)
    lines = resp.text.strip().split("\n")
    reader = csv.DictReader(lines, delimiter="\t")
    all_photos = list(reader)

    # Sample evenly across the dataset for diversity
    step = max(1, len(all_photos) // count)
    selected = all_photos[::step][:count]

    print(f"🎯 Selected {len(selected)} photos to download")

    rows_written = []
    failed = 0

    for photo in tqdm(selected, desc="Downloading images", unit="img"):
        photo_id = photo.get("photo_id", "")
        description = photo.get("photo_description", "") or photo.get(
            "ai_description", ""
        )
        photographer = photo.get("photographer_username", "")
        unsplash_url = photo.get("photo_url", "")

        filename = f"{photo_id}.jpg"
        img_path = IMAGES_DIR / filename

        # Skip if already downloaded
        if img_path.exists():
            rows_written.append(
                {
                    "filename": filename,
                    "caption": description,
                    "photographer": photographer,
                    "unsplash_url": unsplash_url,
                }
            )
            continue

        # Download at 400px width (CLIP will resize to 224 anyway)
        img_url = f"https://images.unsplash.com/photo-{photo_id}?w=400&q=80"
        try:
            r = requests.get(img_url, timeout=15)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(img_path, "JPEG", quality=85)
            rows_written.append(
                {
                    "filename": filename,
                    "caption": description,
                    "photographer": photographer,
                    "unsplash_url": unsplash_url,
                }
            )
            time.sleep(0.05)  # be polite to Unsplash's CDN
        except Exception as e:
            failed += 1
            tqdm.write(f"⚠️  Failed {photo_id}: {e}")

    # Write captions.csv
    with open(CAPTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "caption", "photographer", "unsplash_url"]
        )
        writer.writeheader()
        writer.writerows(rows_written)

    print(f"\n✅ Downloaded {len(rows_written)} images to {IMAGES_DIR}/")
    print(f"📄 Metadata saved to {CAPTIONS_CSV}")
    if failed:
        print(f"⚠️  {failed} images failed to download (network errors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Unsplash demo dataset")
    parser.add_argument(
        "--count", type=int, default=500, help="Number of images to download (default: 500)"
    )
    args = parser.parse_args()
    download_dataset(count=args.count)
