import faiss
import numpy as np
from typing import List, Dict, Tuple


def build_dual_faiss_indices(
    chunks: List[Dict],
) -> Tuple[faiss.IndexFlatIP, faiss.IndexFlatIP, List[Dict]]:
    """
    Create two FAISS indices:
    - section_index: from section embeddings
    - chunk_index: from full chunk embeddings
    Returns:
    - section_index: FAISS index for section titles
    - chunk_index: FAISS index for full content chunks
    - chunks: original chunk data with metadata
    """
    section_embeddings = np.array(
        [chunk["section_embedding"] for chunk in chunks]
    ).astype("float32")

    chunk_embeddings = np.array([chunk["chunk_embedding"] for chunk in chunks]).astype(
        "float32"
    )

    dim = chunk_embeddings.shape[1]

    section_index = faiss.IndexFlatIP(dim)  # cosine sim (normalized embeddings)
    chunk_index = faiss.IndexFlatIP(dim)

    section_index.add(section_embeddings)
    chunk_index.add(chunk_embeddings)

    return section_index, chunk_index, chunks


def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    chunks: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """
    Perform a search on the given FAISS index and return top-k matching chunks.
    """
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return [chunks[i] for i in indices[0]]
