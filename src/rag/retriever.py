import numpy as np
import faiss
from typing import List, Tuple

from src.vectorstore.schema import Chunk

def retrieve_top_k(
    index: faiss.Index,
    query_embedding: np.ndarray,
    chunks: List[Chunk],
    top_k: int = 4,
) -> List[Tuple[Chunk, float]]:
    """
    Returns top-k (Chunk, score) pairs
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    scores, indices = index.search(query_embedding, top_k)  # type: ignore

    results: List[Tuple[Chunk, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        results.append((chunks[idx], float(score)))

    return results
