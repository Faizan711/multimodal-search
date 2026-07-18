# Multimodal Search — HF Spaces Dockerfile
# ─────────────────────────────────────────
# HF Spaces Docker type runs on port 7860.
# CPU-only — CLIP inference works fine without GPU for search.

FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached if requirements don't change)
COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/        ./app/
COPY ui/         ./ui/
COPY scripts/    ./scripts/
COPY data/       ./data/

# Make startup script executable
COPY start.sh ./
RUN chmod +x start.sh

# HF Spaces runs as non-root user
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

CMD ["./start.sh"]
