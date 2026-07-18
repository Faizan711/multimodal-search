"""
app/api.py
──────────
FastAPI application — the HTTP interface for multimodal search.

Endpoints:
  GET  /health              → liveness check
  GET  /info                → collection stats
  GET  /search/text?q=...   → text-to-image search
  POST /search/image        → image-to-image search

Learning notes:
  - FastAPI uses Python type hints to:
      a) validate request data automatically (Pydantic)
      b) generate OpenAPI docs at /docs (Swagger UI)
  - @app.on_event("startup") runs once when the server starts.
    We use it to ensure Qdrant collection exists before handling requests.
  - UploadFile + File(...) tells FastAPI to expect a multipart/form-data upload.
    The `python-multipart` package handles the actual parsing.
  - The API and Streamlit UI run as separate processes. They communicate over HTTP,
    just like any client-server architecture (mobile app ↔ backend API).
"""

import io
import pathlib

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.config import settings
from app.embeddings import encode_image, encode_text
from app.vector_store import ensure_collection, get_collection_info, search

UI_HTML = pathlib.Path("ui/app.html")
IMAGES_DIR = pathlib.Path(settings.image_dir)

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multimodal Search API",
    description="Search images using text or image queries powered by CLIP embeddings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images from data/images/ at /images/<filename>
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ── Lifecycle ────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    """Run once on server start — ensure the Qdrant collection exists."""
    ensure_collection()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    """Serve the main search UI."""
    if UI_HTML.exists():
        return FileResponse(str(UI_HTML), media_type="text/html")
    return HTMLResponse("<h1>UI not found — run uvicorn from project root</h1>", status_code=404)


@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — returns 200 OK if the API is running."""
    return {"status": "ok", "service": "multimodal-search-api"}


@app.get("/info", tags=["System"])
def collection_info():
    """
    Return stats about the vector collection.
    Use this to check how many images are indexed.
    """
    try:
        return get_collection_info()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {e}")


@app.get("/search/text", tags=["Search"])
def search_by_text(
    q: str = Query(..., description="Natural language search query", min_length=1),
    top_k: int = Query(9, description="Number of results to return", ge=1, le=50),
):
    """
    Text-to-image search: encode the text query with CLIP, find nearest image vectors.

    Example: GET /search/text?q=a sunset over the mountains&top_k=6
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    vector = encode_text(q.strip())
    results = search(vector, top_k=top_k)
    return {
        "query": q,
        "modality": "text",
        "count": len(results),
        "results": results,
    }


@app.post("/search/image", tags=["Search"])
async def search_by_image(
    file: UploadFile = File(..., description="Image file (JPEG or PNG)"),
    top_k: int = Query(9, description="Number of results to return", ge=1, le=50),
):
    """
    Image-to-image search: encode the uploaded image with CLIP, find nearest vectors.

    Accepts: JPEG, PNG, WebP
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    # Read and decode the uploaded image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    vector = encode_image(image)
    results = search(vector, top_k=top_k)
    return {
        "filename": file.filename,
        "modality": "image",
        "count": len(results),
        "results": results,
    }
