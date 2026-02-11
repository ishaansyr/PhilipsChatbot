import streamlit as st
import base64
from test_system_lookup import classify, system_lookup, team_lookup
from rag_engine import load_corpus, embed_texts, build_faiss_index, retrieve, generate_answer

from PyPDF2 import PdfReader

def process_uploaded_pdf(file_path):
    text = ""

    reader = PdfReader(file_path)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    # Simple chunking (reuse your existing chunk size logic if different)
    chunk_size = 800
    overlap = 100

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# ------------------ PAGE CONFIG ------------------

st.set_page_config(page_title="Philips Enterprise Knowledge Navigator", layout="wide")


# ------------------ HELPER: LOAD LOGO ------------------

import base64

def load_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ------------------ GLOBAL STYLING ------------------

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">

    <style>
    html, body, [class*="css"]  {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #F5F7FA;
    }

    .stApp {
        background-color: #F5F7FA;
    }

    /* Constrain Main Content Width */
    .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        margin-left: auto;
        margin-right: auto;
    }

    /* Elevated Card Container */
    .assistant-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 60px 60px 50px 60px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.06);
        position: relative;
    }

    /* Divider Under Title */
    .sub-divider {
        width: 80px;
        height: 3px;
        background-color: #0051A8;
        margin: 24px auto 20px auto;
        opacity: 0.85;
    }

    /* Input Styling */
    .stTextInput>div>div>input {
        border-radius: 40px;
        border: 1px solid #E0E0E0;
        padding: 16px 24px;
        font-size: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        transition: all 0.2s ease;
    }

    .stTextInput>div>div>input:focus {
        border: 2px solid #003087;
        box-shadow: 0 12px 28px rgba(0,0,0,0.10);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------ HEADER SECTION ------------------

import streamlit.components.v1 as components

logo_base64 = load_logo("philips logo.png")

header_html = f"""
<div class="assistant-card">

    <div style="position: absolute; top: 30px; right: 40px;">
        <img src="data:image/png;base64,{logo_base64}" width="140">
    </div>

    <div style="text-align: center;">

        <div style="
            font-size: 64px;
            font-weight: 800;
            color: #0051A8;
            letter-spacing: 1.2px;
        ">
            PHILIPS
        </div>

        <div style="
            font-size: 40px;
            font-weight: 600;
            color: #003087;
            margin-top: 12px;
        ">
            Enterprise Knowledge Navigator
        </div>

        <div class="sub-divider"></div>

        <div style="
            font-size: 18px;
            font-weight: 400;
            color: #5F6B7A;
            margin-top: 6px;
        ">
            A unified assistant for projects, operations, and internal systems
        </div>

    </div>

</div>
"""

components.html(header_html, height=360)

# ---------- SIDEBAR ----------

with st.sidebar:

    if st.button("Reset Conversation"):
        st.session_state.chat_history = []

    st.markdown("---")

    st.markdown(
        """
        <div style="color:#0051A8; font-size:20px; font-weight:700;">
        Knowledge Scope
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    • Internal Systems Navigator  
    • Team Responsibility Registry  
    • Curated Project Documentation  
    """)

    st.markdown("---")

    st.markdown("""
    This prototype demonstrates how new employees can quickly navigate internal systems, 
    identify responsible teams, and access project history and lessons learned 
    through a unified knowledge interface.
    """)

    st.markdown("---")

    st.caption("Internal Demonstration Prototype")

    st.markdown("---")
    st.markdown("### Upload Supporting Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF (used for this session only)",
        type=["pdf"]
    )

# ------------------ INITIALISE RAG ------------------

@st.cache_resource
def initialise_rag():
    chunks, metadata = load_corpus()
    embeddings = embed_texts(chunks)
    index = build_faiss_index(embeddings)
    return index, chunks, metadata

index, chunks, metadata = initialise_rag()


# ------------------ SESSION STATE ------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_index" not in st.session_state:
    st.session_state.uploaded_index = None
    st.session_state.uploaded_chunks = None

# ------------------ INPUT SECTION ------------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    user_query = st.text_input("", placeholder="Type your message...")

import tempfile
from rag_engine import load_corpus, embed_texts, build_faiss_index

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Load and chunk uploaded PDF
    uploaded_docs = process_uploaded_pdf(temp_path)
    uploaded_embeddings = embed_texts(uploaded_docs)
    uploaded_index = build_faiss_index(uploaded_embeddings)

    st.session_state.uploaded_index = uploaded_index
    st.session_state.uploaded_chunks = uploaded_docs

    st.success("Document processed and ready for contextual queries.")

# ------------------ ROUTING ------------------

if user_query:

    intent = classify(user_query)

    if intent == "SYSTEM_LOOKUP":
        result = system_lookup(user_query)
        if result:
            task, system, link = result

            response = f"""
            <div style="line-height:1.6; font-size:18px;">
            New employees can complete this action via <b>{system}</b>.

            To {task.lower()}, please access the portal below.

            <br><br>
            <b style="color:#003087;">Link:</b><br>
            <a href="{link}" target="_blank">{link}</a>
            </div>
            """

        else:
            with st.spinner("Searching internal knowledge..."):
                retrieved = retrieve(user_query, index, chunks, metadata, k=3)
                response = generate_answer(user_query, retrieved)

    elif intent == "TEAM_LOOKUP":
        result = team_lookup(user_query)
        if result:
            team, responsibility = result
            response = f"""
            <div style="line-height:1.6; font-size:18px;">
            The <b>{team}</b> team owns this area.

            <br><br>
            {responsibility}
            </div>
            """

        else:
            with st.spinner("Searching internal knowledge..."):
                retrieved = retrieve(user_query, index, chunks, metadata, k=3)
                response = generate_answer(user_query, retrieved)

    else:
        with st.spinner("Searching internal knowledge..."):

            # Retrieve from internal curated corpus
            internal_chunks = retrieve(user_query, index, chunks, metadata, k=3)

            combined_chunks = internal_chunks

            # If a session upload exists, retrieve from it too
            if st.session_state.uploaded_index is not None:
                uploaded_chunks = retrieve(
                    user_query,
                    st.session_state.uploaded_index,
                    st.session_state.uploaded_chunks,
                    None,   # no metadata for uploaded docs
                    k=2
                )
                combined_chunks = internal_chunks + uploaded_chunks

            response = generate_answer(user_query, combined_chunks)

    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", response))


# ------------------ DISPLAY CONVERSATION ------------------

for role, message in st.session_state.chat_history:

    if role == "user":
        st.markdown(
            f"""
            <div style="
                border-left: 5px solid #003087;
                padding: 14px 18px;
                margin: 20px auto;
                max-width: 800px;
                background-color: #f4f7fb;
                font-size: 18px;
            ">
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                margin: 20px auto;
                max-width: 800px;
                background-color: #ffffff;
                border: 1px solid #e6e6e6;
                border-radius: 8px;
                box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
                font-size: 17px;
            ">
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
