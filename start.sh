#!/bin/bash
# start.sh — HF Spaces entry point
# 1. Import pre-exported vectors into Qdrant Cloud (fast, ~3 sec)
# 2. Start FastAPI server

set -e

echo "=== Multimodal Search — Startup ==="
echo "Importing vectors into Qdrant Cloud..."
PYTHONPATH=/app python scripts/import_vectors.py

echo "Starting FastAPI server..."
exec uvicorn app.api:app --host 0.0.0.0 --port 7860
