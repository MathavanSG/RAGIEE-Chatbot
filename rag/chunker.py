from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer


def chunk_sections(
    sections: List[Tuple[str, str]], model: SentenceTransformer
) -> List[Dict]:
    """
    For each section, treat the entire section as one chunk.
    Embed both the chunk text and the heading.
    """
    chunks = []

    for section_name, section_text in sections:
        if not section_text.strip():
            continue

        chunk_embedding = model.encode([section_text], normalize_embeddings=True)[0]
        section_embedding = model.encode([section_name], normalize_embeddings=True)[0]

        chunks.append(
            {
                "text": section_text,
                "section": section_name,
                "chunk_embedding": chunk_embedding,
                "section_embedding": section_embedding,
            }
        )

    return chunks
