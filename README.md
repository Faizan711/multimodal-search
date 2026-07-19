# Multimodal Search

Search images using natural language or image upload — powered by OpenAI CLIP and Qdrant vector search.

This project was built from scratch to explore how modern image search works under the hood. Instead of relying on manual tags or filenames, this app "understands" visual content by embedding both text and images into a shared 512-dimensional vector space.

## Features

- **Text-to-Image Search:** Describe what you are looking for (e.g., "sunset over mountains").
- **Image-to-Image Search:** Upload an image to find visually similar ones.
- **Pipeline Visualization:** A real-time UI animation that shows the AI processing your query step-by-step: Tokenization → CLIP Encoder → Vector Embedding → Qdrant Search → Ranked Results.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Embedding Model** | OpenAI CLIP (ViT-B/32) |
| **Vector Database** | Qdrant (HNSW indexing) |
| **Frontend/UI** | Gradio |
| **Backend** | Python (PyTorch, Transformers) |

---

## Local Setup Instructions

Follow these steps to run the project locally on your machine.

### 1. Prerequisites
- **Python 3.10+**
- **Docker** (to run the local Qdrant database)

### 2. Start Qdrant (Vector Database)
Open a terminal and start a local Qdrant instance using Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 3. Install Python Dependencies
Open a new terminal window in the project directory, create a virtual environment, and install the required packages:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Application
The repository includes a pre-exported file of vectors (`data/vectors.json`) containing 541 sample images. 

Start the Gradio app:
```bash
python gradio_app.py
```
*Note: On the very first run, the app will automatically read `data/vectors.json` and import the vectors into your local Qdrant database. It will also download the CLIP model weights.*

### 5. Use the App
Open your browser and navigate to:
**http://localhost:7860**

Type a query or upload an image, and watch the pipeline animate as it searches!

---

## Project Structure
- `gradio_app.py`: Main Gradio frontend and search execution logic.
- `app/embeddings.py`: CLIP model initialization and inference code.
- `app/vector_store.py`: Qdrant client connection and search logic.
- `app/config.py`: Configuration settings and environment variables.
- `scripts/`: Utilities for downloading datasets and importing/exporting vectors.
- `data/`: Contains the image dataset and the exported vectors JSON.
