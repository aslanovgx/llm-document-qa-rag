from __future__ import annotations
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from src.vectorstore.schema import Chunk

class CEReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = 4
    ) -> List[Tuple[Chunk, float]]:
        # ✅ Pyright üçün list[list[str]] veririk
        pairs: List[List[str]] = [[question, c.text] for (c, _s) in candidates]
        scores = self.model.predict(pairs)

        ranked = list(zip([c for (c, _s) in candidates], [float(s) for s in scores]))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
