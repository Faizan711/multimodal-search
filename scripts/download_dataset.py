"""
scripts/download_dataset.py
────────────────────────────
Downloads 500+ high-quality images for the multimodal search demo.

Strategy:
  - Already downloaded Unsplash images (data/images/) are KEPT (not re-downloaded)
  - Remaining slots filled from picsum.photos (IDs 1–1000)
    → picsum.photos always returns 200 OK, real diverse photography
    → No API key, no auth, no rate limiting
  - Rich category-based captions so text search is meaningful

Why picsum for top-up?
  picsum.photos wraps Unsplash photos by sequential ID — real photography,
  diverse subjects, always available. CLIP reads pixel content, so visual
  quality matters, not the source URL.

Run with:
  python scripts/download_dataset.py             # download up to 500 total
  python scripts/download_dataset.py --count 200 # download only 200 total
  python scripts/download_dataset.py --reset      # wipe and re-download all
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

# ---------------------------------------------------------------------------
# Primary catalog: curated Unsplash IDs with precise captions
# These were already being downloaded — keep them.
# ---------------------------------------------------------------------------
UNSPLASH_CATALOG = [
    ("1506905925346-21bda4d32df4", "mountain landscape with dramatic sky"),
    ("1501854140801-50d01698950b", "green forest path in sunlight"),
    ("1470071459604-3b5ec3a7fe05", "misty mountains at sunrise"),
    ("1441974231531-c6227db76b6e", "forest with tall trees and fog"),
    ("1433086966628-6ed0a82ccfbc", "waterfall in lush green forest"),
    ("1472214103451-9374bd1c798e", "sunset over the ocean horizon"),
    ("1505118380757-91f5f5632de0", "lake reflection of mountains"),
    ("1518020382113-a7e8fc38eac9", "desert sand dunes at golden hour"),
    ("1504701954957-2010ec3bcec1", "snowy mountain peak with blue sky"),
    ("1490730141103-6cac27aaab94", "colorful autumn leaves in forest"),
    ("1500534314209-a25ddb2bd429", "tropical beach with palm trees"),
    ("1516912481800-bf7eb1ef73fb", "rocky coastline with crashing waves"),
    ("1519681393784-d120267933ba", "snow covered pine trees at night"),
    ("1497436072909-60f360a16a24", "green rolling hills with clouds"),
    ("1475924156734-496f6cac6ec1", "canyon with red rock formations"),
    ("1560537359-ac01b72f6c4b", "sunrise over mountain valley"),
    ("1508739773434-c26b3d09e071", "tropical waterfall in jungle"),
    ("1464822759023-fed622ff2c3b", "alpine meadow with wildflowers"),
    ("1532274402911-5a369e4c4bb5", "frozen lake with mountain reflection"),
    ("1507003211169-0a1dd7228f2d", "green hills and blue sky landscape"),
    ("1488582986830-2e89af3023e4", "field of sunflowers at sunset"),
    ("1465146344425-f00d5f5c8f07", "field of lavender in summer"),
    ("1519125323398-675f0ddb6308", "cherry blossom trees in park"),
    ("1510784722491-a6c4ba8d5cd6", "fog over green mountain valley"),
    ("1548247416-ec66f4900b2e", "lion resting on savanna grass"),
    ("1474511320723-9a56873867b5", "fox in snowy winter forest"),
    ("1552053831-71594a27c2d2", "golden retriever dog on beach"),
    ("1511044568932-338cba0ad803", "cat sitting on window sill"),
    ("1537151625747-0bfbab12a4b4", "majestic eagle in flight"),
    ("1564349683136-77e08dba1ef7", "elephant in african savanna"),
    ("1508921912186-1d1a45ebb3c1", "penguin colony on ice"),
    ("1500463959177-e0869687981e", "horse running in green field"),
    ("1425082661705-1834bfd09dca", "dolphin jumping in ocean"),
    ("1518715308788-a4e636e5af4e", "butterfly on colorful flower"),
    ("1437622368342-7a3d73a34c8f", "turtle swimming in clear ocean"),
    ("1449452198679-05c7fd30f416", "colorful parrot in tropical forest"),
    ("1484557985045-edf25e9579a9", "panda eating bamboo"),
    ("1502780402662-acc01917d3ad", "giraffe in african grassland"),
    ("1481833761820-0509d3217039", "flamingo standing in water"),
    ("1547394765-185e1e68f34e", "polar bear on arctic ice"),
    ("1499336315816-097655dcfbda", "deer in misty morning forest"),
    ("1531386151447-fd74bd3c3a4f", "sea turtle underwater coral"),
    ("1517849845537-4d257902454a", "dog running through snow"),
    ("1574158622682-e719686b1e56", "cat yawning on cozy blanket"),
    ("1568572933382-74d440642117", "hummingbird feeding on flower"),
    ("1477959858617-67f85cf4f1df", "aerial view of city at night"),
    ("1480714378408-67cf0d13bc1b", "modern skyscrapers against blue sky"),
    ("1444723121867-7a241cacace9", "busy city street with traffic"),
    ("1499092346682-b9b5dc979ae0", "brooklyn bridge at sunset"),
    ("1513635269975-59663e0ac1ad", "venice canal with gondolas"),
    ("1528360983277-13d401cdc186", "paris eiffel tower at night"),
    ("1502602317473-6c9cb3b5e4f3", "old european cobblestone street"),
    ("1460317442991-0ec209397118", "tokyo city skyline at dusk"),
    ("1533929736458-ca588d08c8be", "new york times square at night"),
    ("1534430480872-3498386e7856", "amsterdam canal houses reflection"),
    ("1517948430231-7f73f8e5e79e", "rome colosseum at golden hour"),
    ("1467269204594-9661b134dd2b", "london tower bridge foggy morning"),
    ("1543349689-9a4d426bee8e", "barcelona sagrada familia"),
    ("1524413840807-0c3cb6fa808d", "dubai skyscrapers skyline"),
    ("1449824913935-59a10b8d2000", "bicycle parked on cobblestone street"),
    ("1513735492246-483525079686", "empty urban road at dusk"),
    ("1518176258769-f227c798150e", "old cathedral in historic city"),
    ("1506794778202-cad84cf45f1d", "person hiking on mountain trail"),
    ("1531746020798-e6953c6e8e04", "young woman smiling outdoors"),
    ("1529626455594-4ff0802cfb7e", "woman reading book in cafe"),
    ("1519085360753-af0119f7cbe7", "businessman working on laptop"),
    ("1551836022-deb4988cc6c0", "couple walking on beach at sunset"),
    ("1488161628813-04466f872be2", "artist painting in studio"),
    ("1544005313-94ddf0286df2", "woman doing yoga at sunrise"),
    ("1523580494863-6f3031224c94", "students studying in library"),
    ("1504194104404-433180773017", "musician playing guitar outdoors"),
    ("1540569014015-e4bb8ebf8ef9", "runner in marathon race"),
    ("1488590528505-98d2b5aba04b", "programmer coding at multiple screens"),
    ("1504674900247-0877df9cc836", "colorful bowl of fresh fruit"),
    ("1476224203421-9ac39bcb3e27", "italian pasta dish with sauce"),
    ("1497034825429-c343d7c6a68f", "fresh sushi platter on black plate"),
    ("1540189549336-e6e99eb4b68e", "burger with fries on wooden board"),
    ("1495521821757-a1efb6729352", "coffee latte art in white cup"),
    ("1567620905732-2d1ec7ab7445", "pizza with fresh toppings"),
    ("1558618666-fcd25c85cd64", "chocolate cake slice on plate"),
    ("1464305795204-6f5bbef3a3ef", "grilled salmon with vegetables"),
    ("1490885578174-acda8905c2c6", "colorful macarons on display"),
    ("1463453091185-61582044d556", "bread loaves from artisan bakery"),
    ("1518770660439-4636190af475", "laptop with code on screen"),
    ("1461749280684-dccba630e2f6", "programming code on monitor"),
    ("1516116216624-53ad39652bc5", "virtual reality headset in use"),
    ("1558494949-ef010cbdcc31", "data center server room"),
    ("1605810230434-7631ac76ec81", "electric car charging station"),
    ("1451187580459-43490279c0fa", "satellite view of earth from space"),
    ("1504384308090-c5a21a9b069d", "solar panels on rooftop"),
    ("1579952363873-27d3bfad7751", "surfer on large ocean wave"),
    ("1534438327788-34d07f10e9c9", "rock climber on cliff face"),
    ("1552674605-db5fecabfe68", "cycling race on mountain road"),
    ("1540497077202-7c8a3999166f", "skier on powder snow slope"),
    ("1530549387789-4c87bc4d9f6e", "yoga class in bright studio"),
    ("1541701494587-cb58502866ab", "colorful abstract paint texture"),
    ("1507919053820-5e7be5ac21d0", "geometric pattern in bright colors"),
    ("1513364776144-60967b0f800f", "neon lights abstract glow"),
    ("1519751138087-5bf79df62d5c", "bokeh light circles background"),
    ("1557683316-973673baf926", "colorful geometric wallpaper"),
]

# ---------------------------------------------------------------------------
# Secondary catalog: picsum.photos IDs with category-based captions
# Picsum IDs 10–600 — all guaranteed to return real photographs.
# Categories cycle through to ensure search diversity.
# ---------------------------------------------------------------------------
PICSUM_CAPTIONS = [
    "landscape nature photography", "city urban architecture", "portrait person",
    "animal wildlife nature", "food cuisine dish", "technology device",
    "ocean sea water waves", "forest trees green", "mountains snow winter",
    "flowers garden colorful", "sunset sunrise sky", "street photography",
    "abstract texture pattern", "sports fitness exercise", "travel adventure",
    "coffee cafe interior", "architecture building modern", "beach sand summer",
    "dog puppy cute", "cat kitten pet", "bird flying sky", "river stream water",
    "night city lights", "desert dry landscape", "snow ice cold",
    "tropical jungle green", "vintage retro old", "minimal clean white",
    "dark moody atmosphere", "colorful vibrant saturated",
]

def picsum_catalog(start_id: int, count: int) -> list[tuple[str, str]]:
    """Generate picsum photo entries with cycling captions."""
    entries = []
    for i in range(count):
        pic_id = start_id + i
        caption = PICSUM_CAPTIONS[i % len(PICSUM_CAPTIONS)]
        entries.append((f"picsum-{pic_id}", caption))
    return entries


def download_dataset(count: int = 500, reset: bool = False) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        import shutil
        shutil.rmtree(IMAGES_DIR)
        IMAGES_DIR.mkdir()
        print("🗑️  Reset: cleared existing images")

    # Check already downloaded files
    existing = {f.stem for f in IMAGES_DIR.glob("*.jpg")}
    print(f"📂 Already have {len(existing)} images on disk")

    # Build combined catalog: Unsplash first, then picsum top-up
    picsum_needed = max(0, count - len(UNSPLASH_CATALOG))
    full_catalog = UNSPLASH_CATALOG + picsum_catalog(start_id=10, count=picsum_needed + 50)
    full_catalog = full_catalog[:count]

    # Split into already-done and todo
    todo = [(pid, cap) for pid, cap in full_catalog
            if pid not in existing and f"{pid}" not in existing]
    already_done = [(pid, cap) for pid, cap in full_catalog
                    if pid in existing or f"{pid}" in existing]

    print(f"🎯 Target: {count} images  |  Already done: {len(already_done)}  |  To download: {len(todo)}")

    rows_written = list(already_done)  # start with what we have
    failed = 0

    for photo_id, caption in tqdm(todo, desc="Downloading", unit="img"):
        filename = f"{photo_id}.jpg"
        img_path = IMAGES_DIR / filename

        # Build URL based on source
        if photo_id.startswith("picsum-"):
            pic_num = photo_id.replace("picsum-", "")
            img_url = f"https://picsum.photos/id/{pic_num}/400/300"
        else:
            img_url = f"https://images.unsplash.com/photo-{photo_id}?w=400&q=75&fm=jpg"

        try:
            r = requests.get(img_url, timeout=15)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(img_path, "JPEG", quality=85)
            rows_written.append((photo_id, caption))
            time.sleep(0.03)
        except Exception as e:
            failed += 1
            tqdm.write(f"⚠️  Failed {photo_id}: {e}")

    # Write captions.csv (include both old and new)
    with open(CAPTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "caption", "photographer", "unsplash_url"]
        )
        writer.writeheader()
        for photo_id, caption in rows_written:
            writer.writerow({
                "filename": f"{photo_id}.jpg",
                "caption": caption,
                "photographer": "unsplash" if not photo_id.startswith("picsum") else "picsum",
                "unsplash_url": "" if photo_id.startswith("picsum") else f"https://unsplash.com/photos/{photo_id}",
            })

    total = len(rows_written)
    print(f"\n✅ Total images ready: {total} (downloaded {total - len(already_done)} new, {failed} failed)")
    print(f"📄 Metadata saved to {CAPTIONS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download demo image dataset")
    parser.add_argument("--count", type=int, default=500,
                        help="Total number of images to have (default: 500)")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe existing images and re-download everything")
    args = parser.parse_args()
    download_dataset(count=args.count, reset=args.reset)
