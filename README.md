---
title: Multimodal Search
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
app_file: gradio_app.py
pinned: false
license: mit
short_description: Search 541 images by text or image using CLIP + Qdrant
---

# Multimodal Search

Search images using natural language or image upload — powered by OpenAI CLIP and Qdrant vector search.

## How it works

1. Type a description (e.g. *"sunset over mountains"*) or upload an image
2. Watch the **live AI pipeline animation** — each step executes and updates in real time:
   - Tokenization → CLIP Encoder → 512-dim Vector → Qdrant HNSW Search → Ranked Results
3. Results appear as a grid ranked by cosine similarity score

## Stack

| Component | Technology |
|-----------|-----------|
| Embedding model | OpenAI CLIP ViT-B/32 |
| Vector database | Qdrant Cloud |
| Frontend | Gradio (HF Spaces) |
| Backend | Python — direct CLIP + Qdrant calls |
