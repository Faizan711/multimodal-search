"""
app.py — Gradio frontend for Multimodal Search (HF Spaces entry point)
───────────────────────────────────────────────────────────────────────
Self-contained: no separate FastAPI server needed.
- CLIP model loaded once at startup
- Vectors imported from data/vectors.json into Qdrant Cloud on first run
- Pipeline animation streamed step-by-step via Python generator → gr.HTML
- Results shown in gr.Gallery
"""

import pathlib
import time

import gradio as gr
from PIL import Image

# spaces is only available inside HF Spaces — mock it for local dev
try:
    import spaces
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn

from app.config import settings
from app.embeddings import encode_image, encode_text
from app.vector_store import ensure_collection, get_client, search

# ── Startup ──────────────────────────────────────────────────────────────────

IMAGES_DIR = pathlib.Path("data/images")


def _startup():
    """Run once: ensure Qdrant collection exists and has data."""
    ensure_collection()
    client = get_client()
    info = client.get_collection(settings.collection_name)
    count = info.points_count or 0
    if count < 10:
        print("Collection empty — importing from data/vectors.json ...")
        from scripts.import_vectors import import_vectors
        import_vectors()
        info = client.get_collection(settings.collection_name)
        count = info.points_count or 0
    print(f"[Ready] {count} vectors in '{settings.collection_name}'")
    return count


VECTOR_COUNT = _startup()


# ── GPU-accelerated encoding (ZeroGPU allocates GPU only for these calls) ──────

@spaces.GPU
def _clip_encode_text(query: str):
    return encode_text(query)


@spaces.GPU
def _clip_encode_image(image):
    return encode_image(image)


# ── Pipeline HTML renderer ────────────────────────────────────────────────────

_STYLES = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
  .pipe-wrap { font-family: 'Inter', sans-serif; display: flex; flex-direction: column; gap: 4px; }
  .pipe-label { font-size: 10px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
                color: #444; margin-bottom: 6px; }
  .step { border: 1px solid transparent; border-radius: 6px; overflow: hidden; }
  .step.idle  { border-color: transparent; opacity: .45; }
  .step.active { border-color: rgba(255,255,255,.13); opacity: 1; }
  .step.done  { border-color: transparent; opacity: 1; }
  .step-head  { display: flex; align-items: center; gap: 8px; padding: 7px 10px;
                background: #161618; }
  .dot { width: 16px; height: 16px; border-radius: 50%; border: 1px solid #444;
         display: flex; align-items: center; justify-content: center;
         font-size: 9px; color: #444; flex-shrink: 0; }
  .active .dot { border-color: #3b82f6; color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,.1); }
  .done   .dot { background: #10b981; border-color: #10b981; color: #fff; font-size: 11px; }
  .step-name { flex: 1; font-size: 12.5px; font-weight: 500; color: #555; }
  .active .step-name, .done .step-name { color: #f0f0f0; }
  .step-ms { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #444; }
  .step-body { padding: 8px 10px 10px; border-top: 1px solid rgba(255,255,255,.06);
               background: #0c0c0e; font-size: 11px; color: #666; line-height: 1.5; }
  .chips { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 5px; }
  .chip { font-family: 'JetBrains Mono', monospace; font-size: 10px;
          background: rgba(59,130,246,.08); border: 1px solid rgba(59,130,246,.18);
          color: #93c5fd; border-radius: 3px; padding: 1px 5px; }
  .bars { display: flex; align-items: flex-end; gap: 2px; height: 24px; margin-top: 5px; }
  .bar  { flex: 1; border-radius: 1px 1px 0 0; min-height: 2px; }
  .srow { display: flex; align-items: center; gap: 6px; margin-top: 4px; }
  .strack { flex: 1; height: 3px; background: rgba(255,255,255,.06); border-radius: 2px; overflow: hidden; }
  .sfill  { height: 100%; border-radius: 2px; background: #10b981; }
  .sval   { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #10b981; width: 32px; text-align: right; }
</style>
"""


def _step_html(n, label, status, body_html="", ms=None):
    dot = "✓" if status == "done" else str(n)
    ms_str = f"{ms}ms" if ms else ""
    body = f'<div class="step-body">{body_html}</div>' if body_html and status in ("active", "done") else ""
    return f"""
    <div class="step {status}">
      <div class="step-head">
        <div class="dot">{dot}</div>
        <div class="step-name">{label}</div>
        <div class="step-ms">{ms_str}</div>
      </div>
      {body}
    </div>"""


def _token_chips(query):
    words = query.split()[:9]
    tokens = ["[SOS]"] + words + ["[EOS]"]
    chips = "".join(f'<span class="chip">{t}</span>' for t in tokens)
    return f'BPE subword tokens:<div class="chips">{chips}</div>'


def _layer_bars():
    import random
    bars = ""
    colors = ["#3b82f6", "#818cf8", "#a78bfa", "#c084fc", "#e879f9",
              "#f472b6", "#34d399", "#22d3ee", "#60a5fa", "#a78bfa", "#34d399", "#fb923c"]
    for i, c in enumerate(colors):
        h = 15 + random.random() * 85
        bars += f'<div class="bar" style="height:{h}%;background:{c};opacity:.7"></div>'
    return f'12-layer transformer activations:<div class="bars">{bars}</div>'


def _vec_bars():
    import random
    bars = ""
    for i in range(40):
        h = random.random() * 100
        hue = 210 + i * 4
        bars += f'<div class="bar" style="height:{h}%;background:hsl({hue},70%,60%);opacity:.6"></div>'
    return f'512-dim L2-normalized vector:<div class="bars">{bars}</div>'


def _score_rows(results):
    html = "Top matches by cosine similarity:"
    for r in results[:4]:
        pct = min(int((r["score"] / 0.4) * 100), 100)
        html += f"""
        <div class="srow">
          <div class="strack"><div class="sfill" style="width:{pct}%"></div></div>
          <span class="sval">{r["score"]:.3f}</span>
        </div>"""
    return html


def _build_html(steps):
    inner = "".join(
        _step_html(s["n"], s["label"], s["status"], s.get("body", ""), s.get("ms"))
        for s in steps
    )
    return f'{_STYLES}<div class="pipe-wrap"><div class="pipe-label">Pipeline</div>{inner}</div>'


# ── Search logic (generator — streams HTML updates) ───────────────────────────

def _run_search(query_text, query_image, top_k):
    """
    Generator: yields (pipeline_html, gallery_images) tuples as each step runs.
    Gradio streams these to the browser in real time.
    """
    is_text = query_text and query_text.strip()
    query   = query_text.strip() if is_text else ""

    steps = [
        {"n": 1, "label": "Tokenization",      "status": "idle"},
        {"n": 2, "label": "CLIP Encoder",       "status": "idle"},
        {"n": 3, "label": "512-dim Vector",     "status": "idle"},
        {"n": 4, "label": "Qdrant HNSW Search", "status": "idle"},
        {"n": 5, "label": "Ranked Results",     "status": "idle"},
    ]

    t0 = time.perf_counter()
    def ms(): return int((time.perf_counter() - t0) * 1000)

    # ── Step 1: Tokenize ──
    steps[0]["status"] = "active"
    steps[0]["body"]   = _token_chips(query if is_text else "image patches (7×7 grid)")
    yield _build_html(steps), []

    time.sleep(0.45)
    steps[0]["status"] = "done"
    steps[0]["ms"] = ms()

    # ── Step 2: Encode ──
    steps[1]["status"] = "active"
    steps[1]["body"]   = _layer_bars()
    yield _build_html(steps), []

    # Actual encoding — runs on ZeroGPU when on HF Spaces
    if is_text:
        vector = _clip_encode_text(query)
    else:
        vector = _clip_encode_image(query_image)

    steps[1]["status"] = "done"
    steps[1]["ms"] = ms()

    # ── Step 3: Vector ──
    steps[2]["status"] = "active"
    steps[2]["body"]   = _vec_bars()
    yield _build_html(steps), []

    time.sleep(0.3)
    steps[2]["status"] = "done"
    steps[2]["ms"] = ms()

    # ── Step 4: Search ──
    steps[3]["status"] = "active"
    steps[3]["body"]   = f"Scanning {VECTOR_COUNT:,} vectors · O(log n) HNSW"
    yield _build_html(steps), []

    results = search(vector, top_k=top_k)

    steps[3]["status"] = "done"
    steps[3]["ms"] = ms()

    # ── Step 5: Results ──
    steps[4]["status"] = "active"
    steps[4]["body"]   = _score_rows(results)
    yield _build_html(steps), []

    time.sleep(0.2)
    steps[4]["status"] = "done"
    steps[4]["ms"] = ms()

    # Build gallery: list of (PIL.Image, caption_string)
    gallery = []
    for r in results:
        img_path = IMAGES_DIR / r["filename"]
        caption  = f"{r['score']:.4f} · {r['caption']}"
        try:
            gallery.append((str(img_path), caption))
        except Exception:
            pass

    yield _build_html(steps), gallery


def search_text(query, top_k):
    if not query or not query.strip():
        yield _build_html([{"n": i, "label": l, "status": "idle"}
                           for i, l in enumerate(["Tokenization","CLIP Encoder","512-dim Vector","Qdrant HNSW Search","Ranked Results"], 1)]), []
        return
    yield from _run_search(query, None, int(top_k))


def search_image(image, top_k):
    if image is None:
        return
    yield from _run_search("", image, int(top_k))


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0c0c0e !important; }
.gradio-container { max-width: 100% !important; }
footer { display: none !important; }

/* Topbar */
#topbar { background: #111113; border-bottom: 1px solid rgba(255,255,255,.07);
          padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; }
#logo   { display: flex; align-items: center; gap: 10px; }
#logo-mark { width: 22px; height: 22px; border-radius: 5px; background: #3b82f6;
             display: flex; align-items: center; justify-content: center;
             font-size: 11px; font-weight: 700; color: #fff; }
#logo-name { font-weight: 600; font-size: 14px; color: #f0f0f0; letter-spacing: -.01em; }
.tag { padding: 2px 8px; border: 1px solid rgba(255,255,255,.07); border-radius: 4px;
       font-size: 11px; color: #666; font-family: monospace; }

/* Panels */
#left-col  { background: #111113; border-right: 1px solid rgba(255,255,255,.07);
              min-height: calc(100vh - 48px); padding: 16px; }
#right-col { padding: 16px; }

/* Inputs */
.search-label { font-size: 11px !important; font-weight: 600 !important;
                text-transform: uppercase; letter-spacing: .08em; color: #444 !important;
                margin-bottom: 8px !important; }
textarea, input[type=text] {
  background: #161618 !important; border: 1px solid rgba(255,255,255,.07) !important;
  border-radius: 6px !important; color: #f0f0f0 !important;
  font-family: 'Inter', sans-serif !important; font-size: 13px !important;
}
textarea:focus, input:focus {
  border-color: rgba(255,255,255,.13) !important;
  box-shadow: none !important;
}
button.primary { background: #3b82f6 !important; border: none !important;
                 border-radius: 6px !important; font-size: 13px !important;
                 font-weight: 500 !important; }
button.primary:hover { opacity: .85 !important; }

/* Gallery */
.gallery-item { border-radius: 4px !important; }
"""


IDLE_HTML = _build_html([
    {"n": i, "label": l, "status": "idle"}
    for i, l in enumerate(
        ["Tokenization", "CLIP Encoder", "512-dim Vector", "Qdrant HNSW Search", "Ranked Results"], 1
    )
])


with gr.Blocks(css=CSS, title="Multimodal Search") as demo:

    # Topbar
    gr.HTML(f"""
    <div id="topbar">
      <div id="logo">
        <div id="logo-mark">◆</div>
        <span id="logo-name">Multimodal Search</span>
      </div>
      <div style="display:flex;gap:6px">
        <span class="tag">CLIP</span>
        <span class="tag">ViT-B/32</span>
        <span class="tag">Qdrant</span>
        <span class="tag">{VECTOR_COUNT:,} images</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT ──
        with gr.Column(scale=1, elem_id="left-col"):
            gr.Markdown("**Search**{.search-label}")

            with gr.Tab("Text"):
                text_in  = gr.Textbox(
                    placeholder='Describe an image — e.g. "sunset over mountains"',
                    lines=2, show_label=False
                )
                top_k_t  = gr.Dropdown([6, 9, 12], value=9, label="Results", scale=0)
                text_btn = gr.Button("→  Search", variant="primary")

            with gr.Tab("Image"):
                img_in   = gr.Image(type="pil", label="Upload image", height=160)
                top_k_i  = gr.Dropdown([6, 9, 12], value=9, label="Results", scale=0)
                img_btn  = gr.Button("→  Find Similar", variant="primary")

            pipe_html = gr.HTML(value=IDLE_HTML)

        # ── RIGHT ──
        with gr.Column(scale=2, elem_id="right-col"):
            gr.HTML('<div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#444;margin-bottom:12px">Results</div>')
            gallery = gr.Gallery(
                label="", columns=3, height="auto",
                show_label=False, object_fit="cover",
            )

    # Wire up
    text_btn.click(
        fn=search_text,
        inputs=[text_in, top_k_t],
        outputs=[pipe_html, gallery],
    )
    img_btn.click(
        fn=search_image,
        inputs=[img_in, top_k_i],
        outputs=[pipe_html, gallery],
    )
    # Also trigger on Enter in text box
    text_in.submit(
        fn=search_text,
        inputs=[text_in, top_k_t],
        outputs=[pipe_html, gallery],
    )

# HF Spaces imports this file and calls demo directly — set show_api here
demo = demo.queue()
demo.show_api = False

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
