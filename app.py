import streamlit as st
import base64
from test_system_lookup import classify, system_lookup, team_lookup
from rag_engine import load_corpus, embed_texts, build_faiss_index, retrieve, generate_answer


# ------------------ PAGE CONFIG ------------------

st.set_page_config(page_title="Philips Knowledge Assistant", layout="wide")


# ------------------ HELPER: LOAD LOGO ------------------

def load_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ------------------ GLOBAL STYLING ------------------

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap" rel="stylesheet">

    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
    }

    .stApp {
        background-color: #ffffff;
    }

    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 40px;
        border: 1px solid #d0d0d0;
        padding: 14px 20px;
        font-size: 18px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }

    .stTextInput>div>div>input:focus {
        border: 2px solid #003087;
        box-shadow: 0px 6px 16px rgba(0,0,0,0.12);
    }

    hr {
        border: 1px solid #003087;
        opacity: 0.2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
.main .block-container {
    max-width: 900px;
    padding-left: 2rem;
    padding-right: 2rem;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)



# ------------------ HEADER SECTION ------------------

import streamlit.components.v1 as components

logo_base64 = load_logo("philips logo.png")

header_html = f"""
<div style="position: relative; padding-top: 60px; padding-bottom: 40px; font-family: 'Inter', sans-serif; background-color: white;">

    <div style="position: absolute; top: -20px; right: 100px;">
        <img src="data:image/png;base64,{logo_base64}" width="200">
    </div>

    <div style="text-align: center;">
    <div style="
    font-size: 78px;
    font-weight: 800;
    color: #0051A8;
    letter-spacing: 2px;
">
    PHILIPS
</div>

<div style="
    font-size: 48px;
    font-weight: 800;
    color: #0051A8;
    margin-top: 8px;
">
    Knowledge Assistant
</div>

    </div>

</div>
"""

components.html(header_html, height=240)

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


# ------------------ INPUT SECTION ------------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    user_query = st.text_input("", placeholder="Type your message...")


# ------------------ ROUTING ------------------

if user_query:

    intent = classify(user_query)

    if intent == "SYSTEM_LOOKUP":
        result = system_lookup(user_query)
        if result:
            task, system, link = result

            response = f"""
            <div style="line-height:1.6;">
            <b style="color:#003087;">Task</b><br>
            {task}<br><br>

            <b style="color:#003087;">System</b><br>
            {system}<br><br>

            <b style="color:#003087;">Access Link</b><br>
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
            <div style="line-height:1.6;">
            <b style="color:#003087;">Team</b><br>
            {team}<br><br>

            <b style="color:#003087;">Responsibility</b><br>
            {responsibility}
            </div>
            """
        else:
            with st.spinner("Searching internal knowledge..."):
                retrieved = retrieve(user_query, index, chunks, metadata, k=3)
                response = generate_answer(user_query, retrieved)

    else:
        with st.spinner("Searching internal knowledge..."):
            retrieved = retrieve(user_query, index, chunks, metadata, k=3)
            response = generate_answer(user_query, retrieved)

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
