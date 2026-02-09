from rag_engine import load_corpus, embed_texts, build_faiss_index, retrieve, generate_answer

# Load corpus
chunks, metadata = load_corpus()

# Embed and index
embeddings = embed_texts(chunks)
index = build_faiss_index(embeddings)

# Test queries
queries = [
    "What went wrong in the packaging transition?",
    "Have we faced alert fatigue issues?",
    "What approaches were tried for CAPA ageing?"
]

for q in queries:
    print("\nQuery:", q)
    retrieved = retrieve(q, index, chunks, metadata, k=3)
    answer = generate_answer(q, retrieved)
    print("Answer:", answer)
    print("-" * 60)
