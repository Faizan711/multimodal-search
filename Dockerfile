# Multimodal Search — HF Spaces Dockerfile
# ─────────────────────────────────────────
# Port 7860 is required by HF Spaces.
# CPU-only torch keeps the image ~2GB instead of ~6GB.

FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow + git-lfs (needed if HF clones with LFS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch from extra index)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY app/        ./app/
COPY ui/         ./ui/
COPY scripts/    ./scripts/
COPY data/       ./data/

# Startup script
COPY start.sh ./
RUN chmod +x start.sh

# HF Spaces requires non-root user with uid 1000
RUN useradd -m -u 1000 user && chown -R user /app
USER user

# Pre-download CLIP model weights into the image at build time
# so first search isn't slow on cold start
RUN python -c "from transformers import CLIPProcessor, CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); print('CLIP cached OK')"

EXPOSE 7860

CMD ["./start.sh"]
