import os
import io
import json
import time
import base64
import numpy as np
import streamlit as st
from huggingface_hub import InferenceClient

# ---------- Optional imports ----------
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    SentenceTransformer = None
    faiss = None

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pandas as pd
except Exception:
    pd = None


# ============================================================
#  CONFIG
# ============================================================
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
MODEL_ID = st.secrets.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
VISION_MODEL_ID = st.secrets.get("VISION_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
vision_client = InferenceClient(model=VISION_MODEL_ID, token=HF_TOKEN)

SYSTEM_PROMPT = """You are DR. NASIR ABBAS ARAIN AI PRO, an elite multilingual AI assistant.

CAPABILITIES:
- Fluent in English, Urdu, Spanish, Portuguese
- Expert in coding, writing, research, analysis
- Document analysis with citations
- Professional tone, clear structure

RULES:
1. Reply in the SAME language the user wrote in.
2. Use markdown formatting (headings, bullets, code blocks).
3. When document context is given, base answers ONLY on it and cite chunks.
4. Never invent facts. If unsure, say so.
5. For code, provide clean, commented, runnable examples.
6. Be concise but complete.
"""

MODE_PROMPTS = {
    "🧠 General": "Answer as a helpful, knowledgeable assistant.",
    "✍️ Writer": "You are a professional writer. Produce polished, structured content with strong openings and clean flow.",
    "🔬 Researcher": "You are a research analyst. Provide detailed, well-sourced, structured answers with comparisons where useful.",
    "📄 PDF Expert": "Answer ONLY from the provided document context. Cite chunk numbers. If not found, say clearly.",
    "💻 Coding": "You are a senior software engineer. Provide clean, well-commented, production-ready code with explanations.",
    "🌍 Translator": "Translate accurately between English, Urdu, Spanish, Portuguese. Preserve meaning, tone, and idioms.",
    "📊 Analyst": "You are a data analyst. Structure answers with tables, bullet points, and clear insights.",
    "🎓 Teacher": "You are a patient teacher. Explain concepts step-by-step with examples, analogies, and simple language.",
}


# ============================================================
#  HELPERS
# ============================================================
@st.cache_resource(show_spinner=False)
def get_embedder():
    if SentenceTransformer is not None:
        return SentenceTransformer(EMBED_MODEL)
    return None


def extract_text(uploaded_file):
    text = ""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf") and PdfReader is not None:
            reader = PdfReader(uploaded_file)
            for i, page in enumerate(reader.pages):
                text += f"\n[Page {i+1}]\n" + (page.extract_text() or "")
        elif name.endswith(".docx") and docx is not None:
            d = docx.Document(uploaded_file)
            for p in d.paragraphs:
                text += p.text + "\n"
        elif name.endswith((".txt", ".md")):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".csv") and pd is not None:
            df = pd.read_csv(uploaded_file)
            text = df.to_string()
        elif name.endswith(".json"):
            data = json.loads(uploaded_file.read().decode("utf-8"))
            text = json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        text = f"[Extraction error: {e}]"
    return text.strip()


def chunk_text(text, chunk_size=700, overlap=120):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def build_index(chunks):
    embedder = get_embedder()
    if embedder is None or faiss is None or not chunks:
        return None, None
    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def search_index(index, chunks, query, top_k=4):
    embedder = get_embedder()
    if index is None or embedder is None:
        return []
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q, top_k)
    results = []
    for rank, (score, i) in enumerate(zip(scores[0], idxs[0]), 1):
        if 0 <= i < len(chunks):
            results.append(f"[Chunk {rank} | score {score:.3f}]\n{chunks[i]}")
    return results


def web_search(query, max_results=5):
    if requests is None or BeautifulSoup is None:
        return ""
    try:
        url = "https://html.duckduckgo.com/html/"
        r = requests.post(
            url, data={"q": query}, timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for res in soup.select(".result")[:max_results]:
            title_el = res.select_one(".result__title")
            snippet_el = res.select_one(".result__snippet")
            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if title or snippet:
                results.append(f"- **{title}**\n  {snippet}")
        return "\n\n".join(results)
    except Exception:
        return ""


def call_llm(messages, max_tokens=1500, temperature=0.7, stream=False):
    try:
        if stream:
            return client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                stream=True,
            )
        resp = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ **Error:** {e}\n\nCheck `HF_TOKEN` secret and model availability."


def analyze_image(image_bytes, prompt):
    """Send image to vision model."""
    try:
        b64 = base64.b64encode(image_bytes).decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }]
        resp = vision_client.chat_completion(messages=messages, max_tokens=800)
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Vision error: {e}"


# ============================================================
#  PAGE SETUP — ChatGPT-style CSS
# ============================================================
st.set_page_config(
    page_title="Dr. Nasir AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark theme */
    .stApp { background: #0f1117; color: #ececf1; }
    section[data-testid="stSidebar"] {
        background: #1a1d29;
        border-right: 1px solid #2a2d3a;
    }
    /* Header */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px; border-radius: 16px;
        margin-bottom: 16px; text-align: center;
        box-shadow: 0 10px 40px rgba(102,126,234,0.3);
    }
    .hero h1 { color: white; margin: 0; font-size: 30px; font-weight: 700; }
    .hero p { color: #f0f0f0; margin: 6px 0 0 0; font-size: 15px; }
    /* Chat bubbles */
    .stChatMessage {
        border-radius: 14px;
        padding: 12px;
        margin: 6px 0;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    /* Input box */
    .stChatInput textarea {
        background: #1e2130 !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid #2e3140 !important;
    }
    /* Buttons */
    .stButton button {
        border-radius: 10px !important;
        transition: all 0.2s !important;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    /* Stats cards */
    .stat-card {
        background: #1e2130;
        border: 1px solid #2e3140;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    .stat-card h3 { color: #667eea; margin: 0; font-size: 22px; }
    .stat-card p { color: #9ca3af; margin: 4px 0 0 0; font-size: 12px; }
    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🧠 DR. NASIR ABBAS ARAIN AI PRO</h1>
    <p>Advanced Multilingual AI Assistant</p>
    <p>🌐 English • اردو • Español • Português</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
#  SESSION STATE
# ============================================================
defaults = {
    "messages": [],
    "doc_index": None,
    "doc_chunks": None,
    "doc_name": None,
    "total_messages": 0,
    "chat_title": "New Chat",
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")

    mode = st.selectbox(
        "🎯 Mode",
        list(MODE_PROMPTS.keys()),
        index=0,
        help="Choose how the AI should respond",
    )

    st.markdown("---")
    st.markdown("### 🔌 Features")

    use_doc = st.checkbox("📄 Document RAG", value=False,
                          help="Answer from uploaded document")
    use_web = st.checkbox("🌐 Web Search", value=False,
                          help="Search the web for current info")
    show_sources = st.checkbox("📚 Show sources", value=True,
                               help="Display retrieved chunks")

    st.markdown("---")
    st.markdown("### 📎 Upload Document")
    uploaded_file = st.file_uploader(
        "PDF • DOCX • TXT • CSV • MD • JSON",
        type=["pdf", "docx", "txt", "csv", "md", "json"],
    )

    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        with st.spinner("Indexing document..."):
            text = extract_text(uploaded_file)
            if not text:
                st.error("Could not extract text.")
            else:
                chunks = chunk_text(text)
                index, chunks = build_index(chunks)
                if index is None:
                    st.error("Embedding failed.")
                else:
                    st.session_state.doc_index = index
                    st.session_state.doc_chunks = chunks
                    st.session_state.doc_name = uploaded_file.name
                    st.success(f"✅ {uploaded_file.name}\n\n{len(chunks)} chunks indexed")

    if st.session_state.doc_name:
        st.caption(f"📄 Active: **{st.session_state.doc_name}**")
        if st.button("🗑️ Remove document", use_container_width=True):
            st.session_state.doc_index = None
            st.session_state.doc_chunks = None
            st.session_state.doc_name = None
            st.rerun()

    st.markdown("---")
    st.markdown("### 🖼️ Image Analysis")
    image_file = st.file_uploader(
        "Upload image", type=["png", "jpg", "jpeg", "webp"],
        key="img_uploader",
    )
    if image_file:
        img_prompt = st.text_input(
            "What to ask about the image?",
            value="Describe this image in detail.",
            key="img_prompt",
        )
        if st.button("🔍 Analyze Image", use_container_width=True):
            with st.spinner("Analyzing..."):
                result = analyze_image(image_file.read(), img_prompt)
            st.markdown(result)

    st.markdown("---")
    st.markdown("### 📊 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{st.session_state.total_messages}</h3>
            <p>Messages</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{len(st.session_state.messages)}</h3>
            <p>In chat</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💾 Export")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 JSON", use_container_width=True):
            if st.session_state.messages:
                data = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
                st.download_button(
                    "Download", data,
                    file_name=f"chat_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                )
    with col2:
        if st.button("📝 TXT", use_container_width=True):
            if st.session_state.messages:
                txt = ""
                for m in st.session_state.messages:
                    role = "You" if m["role"] == "user" else "AI"
                    txt += f"\n=== {role} ===\n{m['content']}\n"
                st.download_button(
                    "Download", txt,
                    file_name=f"chat_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.total_messages = 0
        st.rerun()

    st.markdown("---")
    st.caption("👨‍⚕️ **Dr. Nasir Abbas Arain AI Pro**")
    st.caption("v2.0 • Powered by Hugging Face")


# ============================================================
#  WELCOME
# ============================================================
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown("""
### 👋 Khush Amdeed! Welcome!

Main **Dr. Nasir Abbas Arain AI Pro** hoon — aapka advanced multilingual AI assistant.

**Main kya kar sakta hoon:**
- 💬 **Chat** — Urdu, English, Spanish, Portuguese mein
- 📄 **Document Analysis** — PDF/DOCX upload karke sawal poochein
- 🌐 **Web Search** — latest information ke liye
- 🖼️ **Image Analysis** — drawings, diagrams, photos samjhein
- ✍️ **8 Modes** — Writer, Coder, Translator, Teacher aur zyada
- 💾 **Export** — chat ko JSON/TXT mein save karein

**Shuru karne ke liye neeche apna sawal likhein!** 🚀
""")


# ============================================================
#  DISPLAY CHAT HISTORY
# ============================================================
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ============================================================
#  CHAT INPUT
# ============================================================
if prompt := st.chat_input("Apna sawal likhein… (Ask anything in any language)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_messages += 1

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    context_blocks = []

    # RAG
    if use_doc and st.session_state.doc_index is not None:
        with st.spinner("Searching document..."):
            relevant = search_index(
                st.session_state.doc_index,
                st.session_state.doc_chunks,
                prompt,
            )
        if relevant:
            context_blocks.append(
                "=== DOCUMENT CONTEXT ===\n" + "\n\n---\n\n".join(relevant)
            )
            if show_sources:
                with st.expander(f"📚 Retrieved {len(relevant)} chunks from {st.session_state.doc_name}"):
                    for r in relevant:
                        st.markdown(f"```\n{r[:600]}\n```")

    # Web
    if use_web:
        with st.spinner("Searching web..."):
            web_ctx = web_search(prompt)
        if web_ctx:
            context_blocks.append("=== WEB SEARCH RESULTS ===\n" + web_ctx)
            if show_sources:
                with st.expander("🌐 Web results"):
                    st.markdown(web_ctx)

    # Build messages
    final_user = prompt
    if context_blocks:
        final_user = (
            "\n\n".join(context_blocks)
            + "\n\n=== USER QUESTION ===\n"
            + prompt
        )

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + MODE_PROMPTS[mode]}
    ]
    for m in st.session_state.messages[:-1]:
        api_messages.append({"role": m["role"], "content": m["content"]})
    api_messages.append({"role": "user", "content": final_user})

    # Stream response
    with st.chat_message("assistant", avatar="🧠"):
        placeholder = st.empty()
        full_text = ""
        try:
            stream = call_llm(api_messages, stream=True)
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta:
                    full_text += delta
                    placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)
        except Exception:
            # Fallback non-stream
            full_text = call_llm(api_messages)
            placeholder.markdown(full_text)

    st.session_state.messages.append({"role": "assistant", "content": full_text})
    st.session_state.total_messages += 1
