from rag.reader import extract_text_from_pdf, extract_headings, split_text_by_headings
from rag.chunker import chunk_sections
from sentence_transformers import SentenceTransformer
from rag.embedder import build_dual_faiss_indices, search_index
from rag.retriever import hybrid_retrieve
from config import (
    PDF_PATH,
    EMBED_MODEL,
    SECTION_MATCH_THRESHOLD,
    TOP_K_RETRIEVAL,
    USE_GROQ,
)
from rag.llm import call_groq_llm


# === STEP 1: Read PDF and extract sections ===
print("📄 Loading PDF...")
full_text = extract_text_from_pdf(PDF_PATH)
headings = extract_headings(full_text)
sections = split_text_by_headings(full_text, headings)
print(f"📑 Found {len(sections)} sections")

# === STEP 2: Load embedding model ===
print("📡 Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

# === STEP 3: Chunk + embed each section ===
print("🧩 Creating chunks (1 per section)...")
chunks = chunk_sections(sections, model)

# === STEP 4: Build FAISS indices ===
print("📦 Building FAISS indices...")
section_index, chunk_index, indexed_chunks = build_dual_faiss_indices(chunks)

# === INTERACTIVE QUERY LOOP ===
print("🤖 Ready! Ask anything about the paper.\n")

while True:
    query = input("🔎 Ask a question (or type 'exit'): ").strip()
    if query.lower() == "exit":
        break

    top_chunks = hybrid_retrieve(
        query=query,
        model=model,
        section_index=section_index,
        chunk_index=chunk_index,
        chunks=indexed_chunks,
        section_threshold=0.75,
        top_k=3,
    )

    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    if USE_GROQ:
        answer = call_groq_llm(query, context)
        print("\n💬 Answer:", answer)
