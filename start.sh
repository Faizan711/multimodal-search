#!/bin/bash
# start.sh — HF Spaces container entry point
# 1. Import pre-exported vectors into Qdrant Cloud (~3 seconds)
# 2. Start FastAPI server on port 7860

set -e

echo "======================================"
echo " Multimodal Search — Starting up"
echo "======================================"

if [ -z "$QDRANT_URL" ]; then
  echo "ERROR: QDRANT_URL is not set."
  echo "Add it in HF Space → Settings → Repository secrets"
  exit 1
fi

echo "Qdrant URL: $QDRANT_URL"
echo "Importing vectors into Qdrant Cloud..."
PYTHONPATH=/app python scripts/import_vectors.py

echo ""
echo "Starting FastAPI on port 7860..."
exec uvicorn app.api:app --host 0.0.0.0 --port 7860
