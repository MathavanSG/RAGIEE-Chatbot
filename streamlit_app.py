import streamlit as st
import tempfile
from streamlit_pdf_viewer import pdf_viewer
from sentence_transformers import SentenceTransformer

from rag.reader import extract_text_from_pdf, extract_headings, split_text_by_headings
from rag.chunker import chunk_sections
from rag.embedder import build_dual_faiss_indices
from rag.retriever import hybrid_retrieve
from rag.llm import call_groq_llm

from config import (
    EMBED_MODEL,
    SECTION_MATCH_THRESHOLD,
    TOP_K_RETRIEVAL,
    USE_GROQ,
)


# === Cache model ===
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL)


# === Cache text and section extraction ===
@st.cache_data
def preprocess_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    headings = extract_headings(text)
    sections = split_text_by_headings(text, headings)
    return text, sections


# === Cache chunking + indexing ===
@st.cache_resource
def build_indices(sections, _model):
    chunks = chunk_sections(sections, _model)
    return build_dual_faiss_indices(chunks)


# === UI: Upload PDF ===
uploaded_file = st.sidebar.file_uploader("📄 Upload your PDF", type=["pdf"])
PDF_PATH = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        PDF_PATH = tmp_file.name

# === Sidebar PDF Viewer ===
with st.sidebar:
    st.header("📖 PDF Viewer")
    if PDF_PATH:
        pdf_viewer(PDF_PATH, height=700, width="100%", render_text=True)
    else:
        st.info("Please upload a PDF to begin.")

# === Main App ===
st.title("📚 Academic Paper RAG Assistant")
query = st.text_input(
    "🔎 Ask a question about the paper", placeholder="e.g., What is the abstract?"
)

if query and PDF_PATH:
    model = load_model()
    full_text, sections = preprocess_pdf(PDF_PATH)
    section_index, chunk_index, chunks = build_indices(sections, model)

    # 🔍 Hybrid RAG Retrieval
    top_chunks = hybrid_retrieve(
        query=query,
        model=model,
        section_index=section_index,
        chunk_index=chunk_index,
        chunks=chunks,
        section_threshold=SECTION_MATCH_THRESHOLD,
        top_k=TOP_K_RETRIEVAL,
    )

    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    # 🔎 Show retrieved context
    with st.expander("📄 Retrieved Context", expanded=False):
        for chunk in top_chunks:
            st.markdown(f"**🧩 Section: `{chunk['section']}`**")
            st.write(chunk["text"][:1000] + "...")
            st.divider()

    # 🤖 LLM Answer
    st.markdown("### 💬 Answer")
    with st.spinner("Generating answer using Groq..."):
        answer = call_groq_llm(query, context)
        st.success(answer)

else:
    st.info("⬆️ Upload a PDF and ask a question to get started.")
