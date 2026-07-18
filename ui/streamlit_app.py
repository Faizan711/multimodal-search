"""
ui/streamlit_app.py
────────────────────
Streamlit frontend for multimodal search.

Learning notes:
  - Streamlit re-runs the entire script top-to-bottom on every user interaction.
    Use st.session_state to persist state between runs (e.g., search results).
  - st.cache_resource caches the API URL check so it doesn't re-run every interaction.
  - The UI calls the FastAPI backend via HTTP — same as any frontend would.
    This separation of concerns (UI ↔ API) is production-grade architecture.
"""

import io
import os

import requests
import streamlit as st
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
IMAGES_DIR = "data/images"

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for a polished look ───────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }
    .result-card {
        border-radius: 12px;
        overflow: hidden;
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔍 Multimodal Search</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Search images using natural language or an example image — powered by CLIP embeddings</p>',
    unsafe_allow_html=True,
)

# ── API health check ─────────────────────────────────────────────────────────
@st.cache_resource
def check_api():
    try:
        r = requests.get(f"{API_URL}/info", timeout=3)
        return r.json()
    except Exception:
        return None

info = check_api()
if info is None:
    st.error(f"⚠️ Cannot connect to API at {API_URL}. Is the FastAPI server running?")
    st.code("source .venv/bin/activate && uvicorn app.api:app --reload")
    st.stop()

st.success(f"✅ Connected to API — **{info.get('points_count', 0)}** images indexed")

# ── Search UI ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔤 Text Search", "🖼️ Image Search"])


def display_results(results: list[dict]):
    """Render search results as a 3-column image grid."""
    if not results:
        st.info("No results found. Try a different query.")
        return

    cols = st.columns(3)
    for i, res in enumerate(results):
        with cols[i % 3]:
            img_path = f"{IMAGES_DIR}/{res.get('filename', '')}"
            caption = res.get("caption", "No caption")
            score = res.get("score", 0)

            try:
                st.image(img_path, use_column_width=True)
            except Exception:
                st.warning(f"Image not found: {res.get('filename')}")

            st.markdown(
                f'<span class="score-badge">score: {score:.3f}</span>', unsafe_allow_html=True
            )
            st.caption(caption[:80] + "..." if len(caption) > 80 else caption)


# Tab 1: Text search
with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Describe what you're looking for...",
            placeholder="e.g. a calm lake surrounded by mountains at sunset",
            key="text_query",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.selectbox("Results", [6, 9, 12, 18], index=1, key="text_topk")

    if st.button("🔍 Search", key="text_search_btn", use_container_width=True) and query:
        with st.spinner("Encoding query and searching..."):
            try:
                r = requests.get(
                    f"{API_URL}/search/text",
                    params={"q": query, "top_k": top_k},
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                st.session_state["text_results"] = data["results"]
                st.session_state["last_text_query"] = query
            except Exception as e:
                st.error(f"Search failed: {e}")

    if "text_results" in st.session_state:
        st.markdown(f"**Results for:** _{st.session_state.get('last_text_query', '')}_")
        display_results(st.session_state["text_results"])


# Tab 2: Image search
with tab2:
    uploaded = st.file_uploader(
        "Upload an image to find visually similar ones",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_upload",
    )
    top_k_img = st.selectbox("Results", [6, 9, 12, 18], index=1, key="img_topk")

    if uploaded:
        col_prev, col_btn = st.columns([1, 2])
        with col_prev:
            st.image(uploaded, caption="Your query image", width=200)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Find Similar Images", key="img_search_btn"):
                with st.spinner("Encoding image and searching..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/search/image",
                            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                            params={"top_k": top_k_img},
                            timeout=30,
                        )
                        r.raise_for_status()
                        data = r.json()
                        st.session_state["img_results"] = data["results"]
                    except Exception as e:
                        st.error(f"Search failed: {e}")

    if "img_results" in st.session_state:
        st.markdown("**Similar images:**")
        display_results(st.session_state["img_results"])

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with [CLIP](https://openai.com/research/clip) · "
    "[Qdrant](https://qdrant.tech) · "
    "[FastAPI](https://fastapi.tiangolo.com) · "
    "[Streamlit](https://streamlit.io)"
)
