from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rag.embedder import search_index


def hybrid_retrieve(
    query: str,
    model,
    section_index,
    chunk_index,
    chunks: List[Dict],
    section_threshold: float = 0.75,
    top_k: int = 4,
) -> List[Dict]:
    """
    Hybrid retrieval:
    1. Embed query
    2. Try to match section titles (via section embeddings)
    3. If confident match, return those chunks
    4. Else fallback to chunk content search
    """
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    # Search section index
    section_scores, section_ids = section_index.search(
        query_embedding.reshape(1, -1), top_k
    )
    best_section_score = section_scores[0][0]
    best_section_id = section_ids[0][0]
    best_section_name = chunks[best_section_id]["section"]

    if best_section_score >= section_threshold:
        print(
            f"📌 Matched section: {best_section_name} (score: {best_section_score:.2f})"
        )

        # Return all chunks from that section
        selected_chunks = [
            chunk for chunk in chunks if chunk["section"] == best_section_name
        ]
        return selected_chunks[:top_k]

    else:
        print("⚠️ No strong section match. Falling back to full content similarity.")
        # Fallback to content chunk similarity
        content_results = search_index(chunk_index, query_embedding, chunks, top_k)
        return content_results
