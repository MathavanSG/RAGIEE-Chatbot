# 🤖 RAGIEE-Chatbot: Academic PDF Assistant

**RAGIEE-Chatbot** is a Streamlit-based application that allows users to upload research papers (PDFs), visualize them, ask questions, and receive intelligent, LLM-powered responses. It uses **Retrieval-Augmented Generation (RAG)** and dynamically highlights matched chunks inside the original PDF document using `streamlit-pdf-viewer`.

---

## 🚀 Features

### 📄 PDF Upload + Viewer
- Upload any academic paper in PDF format
- Render and interact with the document directly in Streamlit
- Integrated `streamlit-pdf-viewer` to support copyable text, scroll, and highlights

### 🧠 RAG with Hybrid Search
- Dual FAISS indexing:
  - `chunk_embedding`: Full content of each section
  - `section_embedding`: Semantic meaning of section headings
- Hybrid retrieval logic:
  1. Tries to match semantically with section headings first
  2. Falls back to full content similarity if no strong section match

### 💬 LLM-Backed Answers
- Query understanding and response generation using **Groq API + LLaMA3**
- Custom prompt template for academic-style answers
- Configurable threshold and top-k chunk retrieval

### ✨ Highlighted Context
- Extracted chunks are matched against the PDF using **PyMuPDF**
- Matching content is highlighted directly inside the rendered PDF in yellow
- Auto-scroll or future annotation click-to-jump support coming soon

---

## 🧱 Tech Stack

- [Streamlit](https://streamlit.io/) for UI
- [streamlit-pdf-viewer](https://github.com/lfoppiano/streamlit-pdf-viewer) to render PDFs
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for fast vector search
- [Groq API](https://console.groq.com/) with LLaMA-3 for answer generation


