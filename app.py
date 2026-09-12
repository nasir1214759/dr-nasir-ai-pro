import os
import streamlit as st
from huggingface_hub import InferenceClient

HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
MODEL_ID = st.secrets.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

SYSTEM_PROMPT = """You are DR. NASIR ABBAS ARAIN AI PRO, an elite multilingual AI assistant.

RULES:
1. Reply in the SAME language the user wrote in (Urdu → Urdu, English → English, Spanish → Spanish, Portuguese → Portuguese).
2. Use markdown formatting.
3. Never invent facts. If unsure, say so.
4. For code, provide clean, commented examples.
5. Be concise but complete.
"""

MODE_PROMPTS = {
    "🧠 General": "Answer as a helpful assistant.",
    "✍️ Writer": "You are a professional writer. Produce polished content.",
    "🔬 Researcher": "You are a research analyst. Provide detailed answers.",
    "💻 Coding": "You are a senior software engineer. Provide clean code.",
    "🌍 Translator": "Translate accurately between English, Urdu, Spanish, Portuguese.",
    "🎓 Teacher": "You are a patient teacher. Explain step-by-step.",
}

st.set_page_config(page_title="Dr. Nasir AI Pro", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0f1117; color: #ececf1; }
    section[data-testid="stSidebar"] {
        background: #1a1d29;
        border-right: 1px solid #2a2d3a;
    }
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px; border-radius: 16px;
        margin-bottom: 16px; text-align: center;
        box-shadow: 0 10px 40px rgba(102,126,234,0.3);
    }
    .hero h1 { color: white; margin: 0; font-size: 30px; font-weight: 700; }
    .hero p { color: #f0f0f0; margin: 6px 0 0 0; font-size: 15px; }
    .stChatMessage { border-radius: 14px; padding: 12px; margin: 6px 0; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🧠 DR. NASIR ABBAS ARAIN AI PRO</h1>
    <p>Advanced Multilingual AI Assistant</p>
    <p>🌐 English • اردو • Español • Português</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    mode = st.selectbox("🎯 Mode", list(MODE_PROMPTS.keys()), index=0)

    st.markdown("---")
    st.markdown("### 📊 Stats")
    if "total" not in st.session_state:
        st.session_state.total = 0
    st.metric("Messages", st.session_state.total)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total = 0
        st.rerun()

    st.markdown("---")
    st.caption("👨‍⚕️ **Dr. Nasir Abbas Arain AI Pro**")
    st.caption("v1.0 • Powered by Hugging Face")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown("""
### 👋 Khush Amdeed! Welcome!

Main **Dr. Nasir Abbas Arain AI Pro** hoon — aapka advanced multilingual AI assistant.

**Features:**
- 💬 Chat in Urdu, English, Spanish, Portuguese
- ✍️ 6 Modes — Writer, Coder, Translator, Teacher, Researcher, General
- 🎨 ChatGPT-style dark UI

**Neeche apna sawal likh kar shuru karein!** 🚀
""")

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Apna sawal likhein… (Ask anything)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total += 1

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + MODE_PROMPTS[mode]}
    ]
    for m in st.session_state.messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Soch raha hoon..."):
            try:
                resp = client.chat_completion(
                    messages=api_messages,
                    max_tokens=1200,
                    temperature=0.7,
                    top_p=0.95,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"⚠️ **Error:** {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.total += 1
