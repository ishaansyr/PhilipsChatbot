import os
import re
import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI

client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
PDF_FOLDER = "pdfs"  # folder containing your 6 PDFs


# ---------- TEXT EXTRACTION ----------

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ---------- SEMANTIC CHUNKING ----------

def chunk_text_by_headers(text):
    # Split by section headers
    chunks = re.split(r"\n##\s+", text)
    cleaned = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]
    return cleaned


# ---------- LOAD & PREPARE CORPUS ----------

def load_corpus():
    all_chunks = []
    metadata = []

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, filename)
            text = extract_text_from_pdf(path)
            chunks = chunk_text_by_headers(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append(filename)

    return all_chunks, metadata


# ---------- EMBEDDINGS ----------

def embed_texts(texts):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings).astype("float32")


# ---------- BUILD VECTOR STORE ----------

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# ---------- RETRIEVE ----------

def retrieve(query, index, chunks, metadata, k=3):
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


# ---------- GENERATE ANSWER ----------

SYSTEM_PROMPT = """
You are an enterprise knowledge assistant.

Answer the query using only the provided context as your source of truth.

You may synthesise, organise, and explain the information clearly, but do not introduce facts that are not explicitly supported by the context.

Provide a concise executive-level summary unless the user explicitly requests detailed breakdown.

You may use information from both curated internal documentation and uploaded session documents. Do not fabricate information beyond retrieved context.
If the answer cannot be derived from the context, respond with:
"Information not found in available project documents."
"""

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{query}"}
        ]
    )

    return response.choices[0].message.content.strip()
