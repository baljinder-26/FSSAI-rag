"""Streamlit UI. Talks to the FastAPI backend; never touches OpenAI or the API key."""
import io
import os

import pdfplumber
import requests
import streamlit as st
import streamlit.components.v1 as components

from app.config import settings

API_URL = os.getenv("API_URL", "http://localhost:8000")
NOT_FOUND = "Information not found in the current guidelines."

PDF_FILE = settings.resolve(settings.pdf_path)


@st.cache_data
def page_count() -> int:
    with pdfplumber.open(PDF_FILE) as pdf:
        return len(pdf.pages)


@st.cache_data
def render_page(page: int) -> bytes:
    """Render one PDF page as a PNG (browsers such as Chrome block inline PDF iframes)."""
    with pdfplumber.open(PDF_FILE) as pdf:
        buffer = io.BytesIO()
        pdf.pages[page - 1].to_image(resolution=220).original.save(buffer, format="PNG")
        return buffer.getvalue()


@st.dialog("Source document", width="large")
def show_document(page: int = 1) -> None:
    """Show the PDF from data/ so the user can verify an answer against the original."""
    total = page_count()
    page = st.number_input(f"Go to page (1-{total})", min_value=1, max_value=total, value=page, step=1)
    st.download_button("Download PDF", PDF_FILE.read_bytes(), file_name=PDF_FILE.name,
                       mime="application/pdf")
    st.image(render_page(page), caption=f"{PDF_FILE.name} - page {page} of {total}")


st.set_page_config(page_title="FSSAI Regulatory Assistant", page_icon="🧪")
st.markdown("""
<style>
.stApp {
    background-color: #0b0f14;
    background-image:
        linear-gradient(rgba(74, 222, 128, 0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(74, 222, 128, 0.07) 1px, transparent 1px);
    background-size: 32px 32px;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background-color: rgba(18, 25, 35, 0.92); }
</style>
""", unsafe_allow_html=True)

st.title("FSSAI Regulatory Assistant")
st.caption("Answers come only from the selected FSSAI document, with page and table citations.")

with st.sidebar:
    st.header("Knowledge base")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.write(f"**Document:** {health['document']}")
        if health["indexed"]:
            st.success(f"Indexed: {health['table_rows']} table rows, {health['chunks']} chunks")
        else:
            st.warning("Not indexed. Run `uv run python -m scripts.ingest`.")
        if not health["api_key_set"]:
            st.warning("OPENAI_API_KEY is not set in .env.")
    except requests.RequestException:
        st.error(f"Backend not reachable at {API_URL}. Start it with "
                 "`uv run uvicorn app.main:app --reload`.")
    if PDF_FILE.exists():
        if st.button("📄 Show document", use_container_width=True):
            show_document()
    else:
        st.error(f"PDF not found: {PDF_FILE.name}")
    st.markdown("**Try:**\n- Lead limit in tea?\n- Cadmium in polished rice?\n- Lead limit in rice?")

QUESTION_LABEL = "Ask about a contaminant limit"

# A plain mic button. It uses the browser's built-in speech recognition (Chrome/Edge) and types
# the words straight into the question box while you speak. No audio is sent to our backend.
MIC_BUTTON = """
<style>
  body { margin: 0; }
  button { width: 44px; height: 40px; border-radius: 8px; border: 1px solid #2a3544;
           background: #121923; color: #e6edf3; font-size: 20px; cursor: pointer; }
  button:hover { border-color: #4ade80; }
  button.on { background: #dc2626; border-color: #dc2626; animation: pulse 1s infinite; }
  @keyframes pulse { 50% { opacity: .6; } }
</style>
<button id="mic" title="Click, speak, and the text appears in the box">🎤</button>
<script>
  const win = window.parent, btn = document.getElementById("mic");
  const SR = win.SpeechRecognition || win.webkitSpeechRecognition;
  const box = () => win.document.querySelector('input[aria-label="__LABEL__"]');
  if (!SR) {
    btn.disabled = true; btn.style.opacity = .4;
    btn.title = "Speech input needs Chrome or Edge";
  } else {
    const rec = new SR();
    rec.lang = "en-IN"; rec.interimResults = true; rec.continuous = false;
    const write = (text) => {                       // type into the Streamlit input
      const el = box(); if (!el) return;
      Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, "value").set.call(el, text);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    };
    rec.onresult = (e) => write(Array.from(e.results).map(r => r[0].transcript).join(" "));
    rec.onend = () => btn.classList.remove("on");
    rec.onerror = () => btn.classList.remove("on");
    btn.onclick = () => {
      if (btn.classList.contains("on")) { rec.stop(); return; }
      btn.classList.add("on"); rec.start();
    };
  }
</script>
""".replace("__LABEL__", QUESTION_LABEL)

st.markdown(f"**{QUESTION_LABEL}**")
# A form sends whatever is in the box when Ask is clicked (or Enter is pressed), so a question
# that was just typed or spoken is never lost.
with st.form("ask", border=False):
    text_col, mic_col = st.columns([12, 1], vertical_alignment="center")
    with text_col:
        question = st.text_input(QUESTION_LABEL, key="question", label_visibility="collapsed",
                                 placeholder="e.g. What is the maximum limit of lead in tea?")
    with mic_col:
        components.html(MIC_BUTTON, height=44)
    asked = st.form_submit_button("Ask", type="primary")

if asked and question.strip():
    with st.spinner("Searching the document..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=90)
        except requests.RequestException:
            st.error("Could not reach the backend. Is it running?")
            st.stop()

    if resp.status_code != 200:
        detail = resp.json().get("detail", "Unexpected error") if resp.headers.get(
            "content-type", "").startswith("application/json") else "Unexpected error"
        st.error(f"Request failed: {detail}")
        st.stop()

    data = resp.json()
    st.subheader("Answer")
    if data["answer"] == NOT_FOUND:
        st.warning(NOT_FOUND)
    else:
        st.success(data["answer"])

    if data["sources"]:
        st.subheader("Sources")
        for i, s in enumerate(data["sources"], start=1):
            with st.container(border=True):
                st.markdown(f"**Source {i}** - 📄 {s['document']} · **Page {s['page']}**"
                            + (f" · {s['table']}" if s["table"] else ""))
                st.code(s["evidence"], language=None, wrap_lines=True)
